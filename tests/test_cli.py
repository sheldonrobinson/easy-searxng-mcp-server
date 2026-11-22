"""Tests for command-line version reporting."""

import sys
import unittest
from contextlib import redirect_stdout
from importlib.metadata import PackageNotFoundError
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import main  # noqa: E402


class CommandLineTests(unittest.TestCase):
    def _invoke_cli(self, arguments):
        stdout = StringIO()
        with (
            patch.object(main, "_get_package_version", return_value="2030.1.2.3"),
            patch.object(sys, "argv", ["easy-searxng-mcp-server", *arguments]),
            redirect_stdout(stdout),
        ):
            with self.assertRaises(SystemExit) as raised:
                main.main()

        self.assertEqual(raised.exception.code, 0)
        return stdout.getvalue()

    def test_help_includes_package_version(self):
        for flag in ("-h", "--help"):
            with self.subTest(flag=flag):
                output = self._invoke_cli([flag])
                self.assertIn("package version: 2030.1.2.3", output)
                self.assertIn("--version", output)

    def test_version_option_prints_package_version(self):
        output = self._invoke_cli(["--version"])
        self.assertEqual(
            output.strip(),
            "easy-searxng-mcp-server 2030.1.2.3",
        )

    def test_missing_distribution_metadata_has_explicit_fallback(self):
        with patch.object(
            main,
            "package_version",
            side_effect=PackageNotFoundError,
        ):
            self.assertEqual(main._get_package_version(), "unknown")


if __name__ == "__main__":
    unittest.main()
