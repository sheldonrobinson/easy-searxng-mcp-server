"""Setup script for easy-searxng-mcp-server.

Provides a custom install command that runs a post-install script to
download the faster-whisper base model when the audio-transcription extra
is enabled.

Works alongside pyproject.toml which defines the project metadata.
"""

import os
import sys
from setuptools import setup
from setuptools.command.install import install


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
    cmdclass={
        "install": PostInstallCommand,
    },
)
