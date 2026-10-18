"""
SimpleXNG: A simple way to run SearXNG locally for Konnek MCP Server.

This module embeds a SearXNG instance within the Konnek MCP server,
allowing local search without requiring a remote SearXNG instance.
Uses the searxng package's MCP server as the embedded backend.
"""

import asyncio
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from common.config import setup_logger

# Web crawler configuration
CONTENT_FILTER_THRESHOLD = float(os.getenv("CONTENT_FILTER_THRESHOLD", "0.6"))
WORD_COUNT_THRESHOLD = int(os.getenv("WORD_COUNT_THRESHOLD", "10"))

_settings_path: Optional[Path] = None


class SearXNGServerConfig(BaseModel):
    """SearXNG Server Configuration.

    Attributes:
        log_level: logging level of server (default: logging.INFO)
        host: Host to bind to (default: 127.0.0.1).
        port: Port to run on (default: 11888).
        settings: Path to custom settings.yml file.
        verbose: Enable verbose logging.
    """

    log_level: int = logging.INFO
    host: str = "127.0.0.1"
    port: int = 11888
    app_name: str = "Konnek"
    settings: Optional[str] = None
    verbose: bool = False


class SimpleXNGServer:
    """Embedded SearXNG server for Konnek MCP Server.

    Wraps the searxng MCP server to provide an embedded
    SearXNG instance for local search.
    """

    def __init__(self, server_config: SearXNGServerConfig):
        self.searxng_server_config = server_config
        self.log = logging.getLogger(self.searxng_server_config.app_name)
        self.log.setLevel(level=self.searxng_server_config.log_level)
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server_task: Optional[asyncio.Task] = None

    @staticmethod
    def get_bundled_template() -> Path:
        """Return path to bundled settings template."""
        return Path(__file__).parent / "settings" / "settings_template.yml"

    @staticmethod
    def get_settings_path() -> Optional[Path]:
        """Return current settings path."""
        return _settings_path

    def configure_server(self) -> None:
        """One-time initialization of SearXNG settings."""
        global _settings_path

        settings_path = Path(self.searxng_server_config.settings)

        if _settings_path is not None:
            raise RuntimeError("Settings already initialized")

        if settings_path.exists():
            self.log.warning(
                "Using specified settings file: %s", settings_path
            )
        else:
            # Create settings directory if needed
            if not settings_path.parent.exists():
                settings_path.parent.mkdir(parents=True, exist_ok=True)

            # Load template and customise
            template_path = SimpleXNGServer.get_bundled_template()
            import yaml

            settings = yaml.safe_load(template_path.read_text())

            settings["general"]["debug"] = False
            settings["server"]["port"] = self.searxng_server_config.port
            settings["server"][
                "bind_address"
            ] = self.searxng_server_config.host
            import secrets

            settings["server"]["secret_key"] = secrets.token_hex(16)

            content = (
                f"# Generated from template {template_path.name}\n"
                f"# Port: {self.searxng_server_config.port}, "
                f"Host: {self.searxng_server_config.host}\n"
                f"# Random secret key generated automatically\n\n"
                f"{yaml.dump(settings, default_flow_style=False)}"
            )
            settings_path.write_text(content)

            self.log.warning(
                "Wrote new settings file (including random secret key): %s",
                settings_path,
            )

        _settings_path = settings_path

        # Set environment variables for SearXNG
        os.environ["SEARXNG_SETTINGS_PATH"] = str(settings_path)
        os.environ["SEARXNG_HOST"] = self.searxng_server_config.host
        os.environ["SEARXNG_PORT"] = str(self.searxng_server_config.port)

        self.log.info(
            "SEARXNG_SETTINGS_PATH=%s", os.environ["SEARXNG_SETTINGS_PATH"]
        )
        self.log.info("SEARXNG_HOST=%s", os.environ["SEARXNG_HOST"])
        self.log.info("SEARXNG_PORT=%s", os.environ["SEARXNG_PORT"])

    def start_server(self) -> None:
        """Start the embedded SearXNG server in a background thread."""
        self.configure_server()

        url = f"http://{self.searxng_server_config.host}:{self.searxng_server_config.port}"

        self.log.info(
            "Starting %s embedded SearXNG server...",
            self.searxng_server_config.app_name,
        )
        self.log.info("URL: %s", url)

        def _run_server():
            """Run the searxng MCP server in a new event loop."""
            try:
                from searxng_mcp.server import serve

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop

                loop.run_until_complete(
                    serve(
                        instance_url=url,
                        timeout=30,
                        transport="http",
                        host=self.searxng_server_config.host,
                        port=self.searxng_server_config.port,
                    )
                )
            except Exception as exc:  # pylint: disable=broad-except
                self.log.error(
                    "Embedded SearXNG server error: %s", exc, exc_info=True
                )

        self._thread = threading.Thread(target=_run_server, daemon=True)
        self._thread.start()
        self.log.info(
            "Embedded SearXNG server thread started at %s", url
        )

    def stop_server(self) -> None:
        """Stop the embedded SearXNG server."""
        if self._loop and self._server_task:
            self._server_task.cancel()
        if self._thread:
            self._thread.join(timeout=5)
        self.log.info("Embedded SearXNG server stopped.")


def main() -> None:
    """CLI entry point for embedded SearXNG server."""

    def signal_handler(_signum: int, _frame: Any) -> None:
        """Handle Ctrl+C gracefully."""
        log.warning("Shutting down SearXNG...")
        sys.exit(0)

    log = setup_logger(level="INFO")
    import signal

    signal.signal(signal.SIGINT, signal_handler)

    try:
        server = SimpleXNGServer(server_config=SearXNGServerConfig())
        server.start_server()
        # Keep the main thread alive
        try:
            while True:
                import time as _time

                _time.sleep(1)
        except KeyboardInterrupt:
            server.stop_server()
    except Exception as exc:  # pylint: disable=broad-except
        log.error("Error: %s: %s", exc.__class__.__name__, exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
