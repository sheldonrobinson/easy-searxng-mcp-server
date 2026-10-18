#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Konnek SearXNG MCP Server Agent.

Main entry point for the Konnek SearXNG MCP Server.
Provides search and web_search methods via MCP protocol.
Supports --embedded to start searxng in-process (no separate Flask server),
and --http / --searxng-url for connecting to a remote SearXNG instance.

Transports:
- stdio:  MCP over Standard Input/Output (default)
- streamable-http:  MCP over Streamable HTTP (Starlette + uvicorn)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
import time as _time
from typing import Any, Dict, List, Optional

import uvicorn

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
)

from searxng.client import SearchError, SearchParameters, SearchPort, SearchQuery
from searxng.adapters import HttpSearchAdapter

from common.config import setup_logger
from tavily.adapter import TavilyAdapter, TavilyAdapterConfig


logger = setup_logger(level="INFO")


# ---------------------------------------------------------------------------
# MCP Tool schema
# ---------------------------------------------------------------------------

WEB_SEARCH_TOOL = Tool(
    name="web_search",
    title="Konnek SearXNG Web Search",
    description=(
        "Use SearXNG to search the web for information. "
        "Returns relevant web content with customizable parameters."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string",
                "minLength": 1,
            },
            "categories": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Search categories, e.g. ['general', 'images', 'news']"
                ),
            },
            "engines": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Search engines, e.g. ['google', 'bing', 'duckduckgo']"
                ),
            },
            "language": {
                "type": "string",
                "description": "Search language code (default: 'auto')",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default: 5)",
                "minimum": 1,
                "maximum": 50,
            },
            "time_range": {
                "type": "string",
                "description": "Time range filter",
                "enum": ["day", "week", "month", "year", "all"],
            },
            "safe_search": {
                "type": "integer",
                "description": (
                    "Safe-Search filter (0:normal, 1:moderate, 2:strict)"
                ),
                "minimum": 0,
                "maximum": 2,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)


# ---------------------------------------------------------------------------
# Search port wrapper
# ---------------------------------------------------------------------------

class KonnekSearchPort(SearchPort):
    """SearchPort that delegates to an HttpSearchAdapter."""

    def __init__(self, instance_url: str):
        self._adapter = HttpSearchAdapter(
            instance_url=instance_url,
            timeout=30,
        )

    async def search(
        self, query: SearchQuery, parameters: SearchParameters
    ):
        return await self._adapter.search(query, parameters)

    async def close(self):
        await self._adapter.close()


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

class KonnekMCPServer:
    """MCP server wrapping Konnek SearXNG search functionality."""

    def __init__(self, search_port: SearchPort):
        self._search_port = search_port
        self._server = Server("easy-searxng-mcp-server")
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register MCP protocol handlers."""

        @self._server.list_tools()
        async def handle_list_tools(
            request: ListToolsRequest,
        ) -> ListToolsResult:
            """List available MCP tools."""
            return ListToolsResult(tools=[WEB_SEARCH_TOOL])

        @self._server.call_tool()
        async def handle_call_tool(
            request: CallToolRequest,
        ) -> CallToolResult:
            """Handle MCP tool calls."""
            if request.name != WEB_SEARCH_TOOL.name:
                raise ValueError(f"Unsupported tool: {request.name}")

            arguments = request.arguments or {}
            query = arguments.get("query")

            if not query or not isinstance(query, str) or not query.strip():
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text="Error: Missing required parameter: query",
                        )
                    ],
                    isError=True,
                )

            try:
                # Build search parameters
                categories = arguments.get("categories")
                engines = arguments.get("engines")
                language = arguments.get("language", "auto")
                max_results = arguments.get("max_results", 5)
                time_range = arguments.get("time_range", "all")
                safe_search = arguments.get("safe_search", 1)

                valid_time_ranges = {"day", "week", "month", "year"}
                tr = time_range if time_range in valid_time_ranges else None

                params = SearchParameters(
                    categories=tuple(categories) if categories else ("general",),
                    engines=tuple(engines)
                    if engines
                    else ("google", "bing", "duckduckgo"),
                    language=language,
                    max_results=max_results,
                    time_range=tr,
                )

                result = await self._search_port.search(
                    SearchQuery(text=query), parameters=params
                )

                if not result or not result.results:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"No results found for {query!r}.",
                            )
                        ]
                    )

                # Format results as text
                formatted = []
                for i, r in enumerate(result.results):
                    formatted.append(
                        f"[{i + 1}] {r.title.value}\n"
                        f"URL: {r.url.value}\n"
                        f"{r.content.value}\n"
                    )

                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text="\n".join(formatted),
                        )
                    ]
                )

            except SearchError as exc:
                logger.error("Search execution failed: %s", exc)
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Search failed: {str(exc)}",
                        )
                    ],
                    isError=True,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Tool execution error: %s", exc, exc_info=True)
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Search failed: {str(exc)}",
                        )
                    ],
                    isError=True,
                )

    @property
    def server(self) -> Server:
        """Return the underlying MCP Server instance."""
        return self._server


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point for Konnek SearXNG MCP Server."""
    parser = argparse.ArgumentParser(
        description="Konnek SearXNG MCP Server Agent"
    )
    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio", "streamable-http"],
        help="MCP transport to use (default: stdio).",
    )
    parser.add_argument(
        "--embedded",
        action="store_true",
        help="Start searxng MCP server in-process (background thread) on "
        "localhost, then connect via HttpSearchAdapter. No separate Flask "
        "server is required.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind for streamable-http transport (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind for streamable-http transport (default: 8000).",
    )
    parser.add_argument(
        "--searxng-url",
        type=str,
        default=None,
        help="URL of the remote SearXNG instance "
        "(e.g., http://127.0.0.1:11888).",
    )
    parser.add_argument(
        "--sxng-port",
        type=int,
        default=11888,
        help="Port for embedded SearXNG server (default: 11888).",
    )
    parser.add_argument(
        "--sxng-host",
        type=str,
        default="127.0.0.1",
        help="Host for embedded SearXNG server (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--embeddings-url",
        type=str,
        default=None,
        help="OpenAI-compatible embeddings endpoint URL for VectorRAG reranking.",
    )
    parser.add_argument(
        "--embeddings-model",
        type=str,
        default=None,
        help="Embedding model name (default: uses server's loaded model).",
    )
    parser.add_argument(
        "--embeddings-dim",
        type=int,
        default=None,
        help="Embedding dimension (default: uses model's n_embd).",
    )

    args = parser.parse_args()

    # Determine embeddings configuration
    if args.embeddings_url:
        embeddings_url = args.embeddings_url
    elif os.getenv("EMBEDDINGS_URL"):
        embeddings_url = os.getenv("EMBEDDINGS_URL")
    else:
        embeddings_url = None

    if args.embeddings_model:
        embeddings_model = args.embeddings_model
    elif os.getenv("EMBEDDINGS_MODEL"):
        embeddings_model = os.getenv("EMBEDDINGS_MODEL")
    else:
        embeddings_model = None

    if args.embeddings_dim is not None:
        embeddings_dim = args.embeddings_dim
    elif os.getenv("EMBEDDINGS_DIM"):
        embeddings_dim = int(os.getenv("EMBEDDINGS_DIM"))
    else:
        embeddings_dim = 0

    # Determine SearXNG URL
    if args.embedded:
        searxng_url = f"http://{args.sxng_host}:{args.sxng_port}"
    elif args.searxng_url:
        searxng_url = args.searxng_url
    elif os.getenv("SEARXNG_URL"):
        searxng_url = os.getenv("SEARXNG_URL")
    else:
        searxng_url = "https://searx.party"

    # Configure Tavily adapter
    TavilyAdapter.configure_tavily_adapter(
        tavily_config=TavilyAdapterConfig(
            settings=None,
            embeddings_url=embeddings_url,
            embeddings_model=embeddings_model,
            embeddings_dim=embeddings_dim,
        )
    )

    # For embedded mode, start searxng MCP server in-process
    if args.embedded:
        logger.info("Starting embedded SearXNG server in-process...")
        import searxng.server as sxng_server_module

        def _run_searxng_server():
            try:
                asyncio.run(
                    sxng_server_module.serve(
                        instance_url=searxng_url,
                        timeout=30,
                        transport="http",
                        host=args.sxng_host,
                        port=args.sxng_port,
                    )
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.error(
                    "Embedded SearXNG server error: %s", exc, exc_info=True
                )

        searxng_thread = threading.Thread(
            target=_run_searxng_server, daemon=True
        )
        searxng_thread.start()

        logger.info(
            "Embedded SearXNG server starting at %s", searxng_url
        )

        # Give the server a moment to start
        _time.sleep(2)

    # Create search port and MCP server
    search_port = KonnekSearchPort(instance_url=searxng_url)
    konnek_server = KonnekMCPServer(search_port=search_port)
    mcp_server = konnek_server.server

    # Run based on transport
    if args.transport == "stdio":
        logger.info("Starting MCP server on stdio transport...")

        async def _run_stdio():
            async with stdio_server() as (read_stream, write_stream):
                await mcp_server.run(
                    read_stream,
                    write_stream,
                    mcp_server.create_initialization_options(),
                )

        asyncio.run(_run_stdio())

    elif args.transport == "streamable-http":
        logger.info(
            "Starting MCP server on streamable-http transport at %s:%d ...",
            args.host, args.port,
        )

        app = mcp_server.streamable_http_app(
            streamable_http_path="/mcp",
            host=args.host,
        )

        config = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            log_config=None,
        )
        server = uvicorn.Server(config)
        asyncio.run(server.serve())

    else:
        logger.error(
            "Unknown transport: %s. Use 'stdio' or 'streamable-http'.",
            args.transport,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
