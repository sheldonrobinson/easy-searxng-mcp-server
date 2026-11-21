"""Setup script for easy-searxng-mcp-server.

Provides a custom install command that runs a post-install script to
download the faster-whisper base model when the audio-transcription extra
is enabled.

Works alongside pyproject.toml which defines the project metadata.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import date
from setuptools import setup
from setuptools.command.install import install

_PYPI_JSON_URL = "https://pypi.org/pypi/easy-searxng-mcp-server/json"


def _fetch_latest_pypi_version() -> str | None:
    """Query PyPI for the latest published version of this package.

    Returns the version string (e.g., ``2026.11.21.1``) or ``None`` if
    the request fails or the package does not exist yet.
    """
    try:
        req = urllib.request.Request(
            _PYPI_JSON_URL,
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("info", {}).get("version")
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError):
        return None


def _get_build_date_version() -> str:
    """Generate a PEP 440 compliant version from the build date.

    Format: ``YYYY.mm.dd.{build-number}`` (e.g., ``2026.11.21.0``).
    PyPI does not allow local version identifiers (PEP 440), so we use
    a date-based version with an incrementing build number.

    The function queries PyPI for the latest published version.  If a
    version for today already exists, the build number is incremented by
    1; otherwise it starts at 0.  If the network is unavailable, the
    version falls back to ``YYYY.mm.dd.0``.
    """
    latest = _fetch_latest_pypi_version()
    today = date.today()
    today_prefix = f"{today.year}.{today.month}.{today.day}"

    if latest == today_prefix:
        # Exact date match (e.g., "2026.11.21") — start at build 1
        next_build = 1
    elif latest and latest.startswith(f"{today_prefix}."):
        # Date + build number (e.g., "2026.11.21.3") — increment
        try:
            build_num = int(latest.split(".")[-1])
            next_build = build_num + 1
        except (ValueError, IndexError):
            next_build = 1
    else:
        next_build = 0

    return f"{today_prefix}.{next_build}"


class PostInstallCommand(install):
    """Custom install command that runs post-install steps."""

    def run(self):
        install.run(self)

        # Run post-install model download script
        script_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "src",
            "common",
            "post_install_model.py",
        )
        if os.path.exists(script_path):
            print("[post-install] Running post-install model download script...")
            result = os.system(f"{sys.executable} {script_path}")
            if result != 0:
                print(
                    "[post-install] Warning: post-install script exited with "
                    f"code {result}",
                    file=sys.stderr,
                )
        else:
            print(
                f"[post-install] Script not found: {script_path}",
                file=sys.stderr,
            )


setup(
    version=_get_build_date_version(),
    cmdclass={
        "install": PostInstallCommand,
    },
)
