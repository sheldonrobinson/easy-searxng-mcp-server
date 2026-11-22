"""Regression tests for the embedded SearXNG configuration."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import main  # noqa: E402


class EmbeddedSearXNGSettingsTests(unittest.TestCase):
    def test_embedded_settings_enable_json_search(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_path = main._write_embedded_settings(
                "127.0.0.1",
                11999,
                Path(temp_dir),
            )
            settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))

        self.assertIn("json", settings["search"]["formats"])
        self.assertEqual(settings["server"]["bind_address"], "127.0.0.1")
        self.assertEqual(settings["server"]["port"], 11999)
        self.assertEqual(settings["server"]["method"], "GET")
        self.assertFalse(settings["server"]["limiter"])
        self.assertNotEqual(
            settings["server"]["secret_key"],
            "konnek-searxng-local-key-change-for-production",
        )
        self.assertGreaterEqual(len(settings["server"]["secret_key"]), 64)

    def test_embedded_mode_starts_searxng_backend(self):
        argv = [
            "easy-searxng-mcp-server",
            "--embedded",
            "--sxng-port",
            "11999",
        ]
        with (
            patch.object(main.sys, "argv", argv),
            patch.object(main.TavilyAdapter, "configure_tavily_adapter"),
            patch.object(main, "_start_embedded_searxng") as start_embedded,
            patch.object(main, "KonnekSearchPort") as search_port_type,
            patch.object(main, "KonnekMCPServer"),
            patch.object(
                main.asyncio,
                "run",
                side_effect=lambda coroutine: coroutine.close(),
            ),
        ):
            main.main()

        start_embedded.assert_called_once_with("127.0.0.1", 11999)
        search_port_type.assert_called_once_with(
            instance_url="http://127.0.0.1:11999"
        )


if __name__ == "__main__":
    unittest.main()
