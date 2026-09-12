import asyncio
import itertools
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal


DEFAULT_MAX_CONCURRENT = int(os.getenv("KEY_ROTATOR_MAX_CONCURRENT", "50"))


class NoAvailableAPIKeysError(RuntimeError):
    """No configured API keys remain available in the current process."""


class KeyRotator:
    """API key selector with process-local concurrency and failover control."""

    def __init__(
        self,
        keys: list[str],
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        strategy: Literal["sticky", "round_robin"] = "sticky",
    ) -> None:
        if not keys:
            raise ValueError("At least one key must be provided")
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be positive")
        if strategy not in {"sticky", "round_robin"}:
            raise ValueError(f"Unknown key selection strategy: {strategy}")
        # De-duplicate keys without changing their configured order.
        self._keys = list(dict.fromkeys(keys))
        self.strategy = strategy
        self._cycle = itertools.cycle(self._keys)
        self._disabled_keys: set[str] = set()
        self._current_key: str | None = self._keys[0]
        self._switching_from: str | None = None
        self._in_flight = {key: 0 for key in self._keys}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._state_changed = asyncio.Condition()

    @classmethod
    def from_env(
        cls,
        env_var: str,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        strategy: Literal["sticky", "round_robin"] = "sticky",
    ) -> "KeyRotator":
        raw = os.getenv(env_var)
        if not raw:
            raise ValueError(f"{env_var} is not set")
        keys = [k.strip() for k in raw.split(";") if k.strip()]
        if not keys:
            raise ValueError(f"{env_var} is empty after parsing")
        return cls(keys, max_concurrent, strategy)

    def _next_sticky_key(self, previous: str | None) -> str | None:
        if previous in self._keys:
            start = self._keys.index(previous) + 1
        else:
            start = 0
        ordered = self._keys[start:] + self._keys[:start]
        return next((key for key in ordered if key not in self._disabled_keys), None)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[str]:
        """Acquire a request slot and yield a currently available key.

        Sticky mode never exposes a replacement key until every request using
        the disabled key has left this context. This prevents two configured
        keys from being used concurrently by one process during failover.
        """
        await self._semaphore.acquire()
        key: str | None = None
        try:
            async with self._state_changed:
                while True:
                    if self.strategy == "sticky":
                        if self._switching_from is not None:
                            await self._state_changed.wait()
                            continue
                        key = self._current_key
                        if key is not None and key not in self._disabled_keys:
                            break
                    else:
                        for _ in range(len(self._keys)):
                            candidate = next(self._cycle)
                            if candidate not in self._disabled_keys:
                                key = candidate
                                break
                        if key is not None:
                            break
                    raise NoAvailableAPIKeysError(
                        "All configured API keys are unavailable in this process"
                    )
                self._in_flight[key] += 1
            yield key
        finally:
            if key is not None:
                async with self._state_changed:
                    self._in_flight[key] -= 1
                    if (
                        self.strategy == "sticky"
                        and self._switching_from == key
                        and self._in_flight[key] == 0
                    ):
                        self._current_key = self._next_sticky_key(key)
                        self._switching_from = None
                        self._state_changed.notify_all()
            self._semaphore.release()

    async def disable(self, key: str) -> bool:
        """Disable a failed key for the rest of the current process.

        Returns True only when this call changed the key's state.
        """
        async with self._state_changed:
            if key not in self._keys or key in self._disabled_keys:
                return False
            self._disabled_keys.add(key)
            if self.strategy == "sticky" and key == self._current_key:
                if self._in_flight[key]:
                    self._switching_from = key
                else:
                    self._current_key = self._next_sticky_key(key)
                self._state_changed.notify_all()
            return True

    @property
    def key_count(self) -> int:
        return len(self._keys)

    @property
    def active_key_count(self) -> int:
        return len(self._keys) - len(self._disabled_keys)


_rotators: dict[tuple[str, int, str], KeyRotator] = {}


def get_rotator(
    env_var: str,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    strategy: Literal["sticky", "round_robin"] = "sticky",
) -> KeyRotator:
    """Return a process-shared key selector for the requested configuration."""
    cache_key = (env_var, max_concurrent, strategy)
    if cache_key not in _rotators:
        _rotators[cache_key] = KeyRotator.from_env(env_var, max_concurrent, strategy)
    return _rotators[cache_key]
