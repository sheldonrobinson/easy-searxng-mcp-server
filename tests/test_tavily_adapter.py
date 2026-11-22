"""Import-time regression tests for the Tavily adapter."""

import sys
import unittest
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tavily.adapter as adapter  # noqa: E402


class TavilyAdapterImportTests(unittest.TestCase):
    def test_aiohttp_is_bound_for_annotations_and_runtime_use(self):
        self.assertIs(adapter.aiohttp, aiohttp)


if __name__ == "__main__":
    unittest.main()
