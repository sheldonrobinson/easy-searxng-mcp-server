"""Setup script for easy-searxng-mcp-server.

Provides a custom install command that runs a post-install script to
download the faster-whisper base model when the audio-transcription extra
is enabled.

Works alongside pyproject.toml which defines the project metadata.
"""

import os
import sys
from datetime import date
from setuptools import setup
from setuptools.command.install import install


def _get_build_date_version() -> str:
    """Generate a PEP 440 compliant version from the build date.

    Format: ``YYYY.mm.dd.{build-number}`` (e.g., ``2026.11.21.0``).
    PyPI does not allow local version identifiers (PEP 440), so we use
    a date-based version with an incrementing build number.

    The ``PACKAGE_VERSION`` environment variable, if set by the CI/CD
    workflow, takes precedence.  Otherwise the version is derived from
    today's date with build number ``0``.
    """
    env_version = os.environ.get("PACKAGE_VERSION")
    if env_version:
        return env_version

    today = date.today()
    return f"{today.year}.{today.month}.{today.day}.0"


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
