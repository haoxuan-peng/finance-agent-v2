import os
import unittest
from unittest.mock import patch

from finance_agent.key_rotator import KeyRotator, NoAvailableAPIKeysError


async def _take_keys(rotator: KeyRotator, count: int) -> list[str]:
    keys: list[str] = []
    for _ in range(count):
        async with rotator.acquire() as key:
            keys.append(key)
    return keys


class KeyRotatorTest(unittest.IsolatedAsyncioTestCase):
    async def test_sticky_strategy_keeps_using_first_key(self) -> None:
        keys = ["key-a", "key-b", "key-c"]
        actual = await _take_keys(KeyRotator(keys), len(keys) * 2)

        self.assertEqual(keys, ["key-a", "key-b", "key-c"])
        self.assertEqual(actual, ["key-a"] * (len(keys) * 2))

    async def test_round_robin_strategy_cycles_available_keys(self) -> None:
        rotator = KeyRotator(["key-a", "key-b", "key-c"], strategy="round_robin")

        actual = await _take_keys(rotator, 6)

        self.assertEqual(actual, ["key-a", "key-b", "key-c"] * 2)

    async def test_disabled_sticky_key_is_replaced(self) -> None:
        rotator = KeyRotator(["key-a", "key-b"])

        await rotator.disable("key-a")

        self.assertEqual(rotator.active_key_count, 1)
        self.assertEqual(await _take_keys(rotator, 1), ["key-b"])

    async def test_all_keys_disabled_raise_safe_error(self) -> None:
        rotator = KeyRotator(["key-a"])
        await rotator.disable("key-a")

        with self.assertRaises(NoAvailableAPIKeysError):
            await _take_keys(rotator, 1)

    def test_from_env_parses_semicolon_separated_keys(self) -> None:
        with patch.dict(os.environ, {"TEST_API_KEYS": " key-a ; key-b ; key-a "}):
            rotator = KeyRotator.from_env("TEST_API_KEYS")

        self.assertEqual(rotator.key_count, 2)


if __name__ == "__main__":
    unittest.main()
