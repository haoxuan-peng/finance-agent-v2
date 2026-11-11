import logging
import unittest
from typing import TypedDict
from unittest.mock import AsyncMock, MagicMock, patch

from finance_agent.key_rotator import KeyRotator
from finance_agent.prompt import SYSTEM_PROMPT
from finance_agent.tools import MAX_END_DATE, EDGARSearch, PriceHistory, TavilyWebSearch


CUTOFF = "2025-05-31"


class SearchArguments(TypedDict, total=False):
    start_date: str
    end_date: str | None


class DateCutoffConfigurationTests(unittest.TestCase):
    def test_prompt_and_tool_schemas_use_the_requested_cutoff(self):
        self.assertEqual(MAX_END_DATE, CUTOFF)
        self.assertIn("Today's date is May 31, 2025.", SYSTEM_PROMPT)
        for tool_class in (TavilyWebSearch, EDGARSearch):
            with self.subTest(tool=tool_class.name):
                self.assertEqual(tool_class.parameters["end_date"]["default"], CUTOFF)
        for tool_class in (TavilyWebSearch, EDGARSearch, PriceHistory):
            with self.subTest(tool=tool_class.name):
                self.assertIn(CUTOFF, tool_class.parameters["end_date"]["description"])


class SearchDateCutoffTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.web_client = MagicMock()
        self.web_client.search = AsyncMock(return_value={"results": []})
        self.web = TavilyWebSearch(
            key_rotator=KeyRotator(["offline-test-key"]),
            client_factory=lambda key: self.web_client,
        )
        self.edgar = EDGARSearch(key_rotator=KeyRotator(["offline-test-key"]))

        response = MagicMock()
        response.__aenter__.return_value = response
        response.json = AsyncMock(return_value={"filings": []})
        self.session = MagicMock()
        self.session.__aenter__.return_value = self.session
        self.session.post.return_value = response
        session_patch = patch(
            "finance_agent.tools.aiohttp.ClientSession", return_value=self.session
        )
        session_patch.start()
        self.addCleanup(session_patch.stop)

    async def test_missing_or_empty_end_dates_use_the_cutoff(self):
        cases: tuple[SearchArguments, ...] = ({}, {"end_date": None}, {"end_date": ""})
        for arguments in cases:
            for tool in (self.web, self.edgar):
                with self.subTest(tool=tool.name, arguments=arguments):
                    await tool._execute_search("test query", **arguments)
                    if tool is self.web:
                        self.assertEqual(
                            self.web_client.search.call_args.kwargs["end_date"], CUTOFF
                        )
                    else:
                        payload = self.session.post.call_args.kwargs["json"]
                        self.assertEqual(payload["startDate"], "1900-01-01")
                        self.assertEqual(payload["endDate"], CUTOFF)

    async def test_date_ranges_are_clipped_before_being_sent(self):
        cases = (
            ("2025-01-01", "2026-06-30", "2025-01-01", CUTOFF),
            ("2026-06-01", "2026-06-30", CUTOFF, CUTOFF),
            ("2024-01-01", "2024-12-31", "2024-01-01", "2024-12-31"),
            (CUTOFF, CUTOFF, CUTOFF, CUTOFF),
        )
        for start, end, expected_start, expected_end in cases:
            for tool in (self.web, self.edgar):
                with self.subTest(tool=tool.name, start=start, end=end):
                    await tool._execute_search(
                        "test query", start_date=start, end_date=end
                    )
                    if tool is self.web:
                        payload = self.web_client.search.call_args.kwargs
                        self.assertEqual(payload["start_date"], expected_start)
                        self.assertEqual(payload["end_date"], expected_end)
                    else:
                        payload = self.session.post.call_args.kwargs["json"]
                        self.assertEqual(payload["startDate"], expected_start)
                        self.assertEqual(payload["endDate"], expected_end)

    async def test_invalid_dates_are_rejected_before_searching(self):
        cases: tuple[tuple[SearchArguments, str], ...] = (
            (SearchArguments(start_date="2025/01/01"), "start_date"),
            (SearchArguments(end_date="not-a-date"), "end_date"),
            (
                SearchArguments(start_date="2025-05-02", end_date="2025-05-01"),
                "later than end_date",
            ),
        )
        for arguments, message in cases:
            for tool in (self.web, self.edgar):
                with self.subTest(tool=tool.name, arguments=arguments):
                    with self.assertRaisesRegex(ValueError, message):
                        await tool._execute_search("test query", **arguments)
        self.web_client.search.assert_not_awaited()
        self.session.post.assert_not_called()


class PriceDateCutoffTests(unittest.IsolatedAsyncioTestCase):
    async def test_date_ranges_are_clipped_before_fetching_prices(self):
        tool = PriceHistory(api_key="offline-test-key")
        cases = (
            ("2025-01-01", "2026-06-30", "2025-01-01", CUTOFF),
            ("2026-06-01", "2026-06-30", CUTOFF, CUTOFF),
            ("2024-01-01", "2024-12-31", "2024-01-01", "2024-12-31"),
            (CUTOFF, CUTOFF, CUTOFF, CUTOFF),
        )
        for start, end, expected_start, expected_end in cases:
            with self.subTest(start=start, end=end):
                with patch.object(tool, "_fetch", new=AsyncMock(return_value=[])) as fetch:
                    result = await tool.execute(
                        {
                            "ticker": "AAPL",
                            "asset_class": "equity",
                            "start_date": start,
                            "end_date": end,
                        },
                        {},
                        logging.getLogger(__name__),
                    )
                self.assertIsNone(result.error)
                fetch.assert_awaited_once_with(
                    "equity", "AAPL", expected_start, expected_end
                )

    async def test_invalid_dates_are_rejected_before_fetching_prices(self):
        tool = PriceHistory(api_key="offline-test-key")
        cases = (
            ("2025/01/01", "2025-05-01", "start_date"),
            ("2025-01-01", "not-a-date", "end_date"),
            ("2025-05-02", "2025-05-01", "later than end_date"),
        )
        for start, end, message in cases:
            with self.subTest(start=start, end=end):
                with patch.object(tool, "_fetch", new=AsyncMock()) as fetch:
                    result = await tool.execute(
                        {
                            "ticker": "AAPL",
                            "asset_class": "equity",
                            "start_date": start,
                            "end_date": end,
                        },
                        {},
                        logging.getLogger(__name__),
                    )
                self.assertIsNotNone(result.error)
                self.assertIn(message, result.error or "")
                fetch.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
