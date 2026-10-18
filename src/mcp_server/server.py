"""MCP server implementation for Konnek SearXNG integration.

Provides the web_search MCP tool using SearXNG as the backend.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

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

logger = logging.getLogger(__name__)


# Define the web_search tool schema
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
                "description": "Safe-Search filter (0:normal, 1:moderate, 2:strict)",
                "minimum": 0,
                "maximum": 2,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
)


class KonnekMCPServer:
    """MCP server wrapper for Konnek SearXNG integration."""

    def __init__(self, search_handler: Any):
        self._server = Server("easy-searxng-mcp-server")
        self._search_handler = search_handler
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
                # Call the search handler (synchronous)
                result = self._search_handler(
                    query=query,
                    categories=arguments.get("categories"),
                    engines=arguments.get("engines"),
                    language=arguments.get("language", "auto"),
                    max_results=arguments.get("max_results", 5),
                    time_range=arguments.get("time_range", "all"),
                    safe_search=arguments.get("safe_search", 1),
                )

                if result.get("status") == "error":
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"Search failed: {result.get('message', 'Unknown error')}",
                            )
                        ],
                        isError=True,
                    )

                # Format results as text
                results = result.get("results", [])
                if not results:
                    return CallToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"No results found for {query!r}.",
                            )
                        ]
                    )

                formatted = []
                for i, r in enumerate(results):
                    if isinstance(r, dict):
                        content = r.get("content", str(r))
                        url = r.get("url", "")
                        score = r.get("score", 0)
                        formatted.append(
                            f"[{i + 1}] {content}\n"
                            f"URL: {url}\n"
                            f"Score: {score}\n"
                        )
                    else:
                        formatted.append(f"[{i + 1}] {r}\n")

                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text="\n".join(formatted),
                        )
                    ]
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

    async def run_stdio(self) -> None:
        """Run the MCP server over stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options(),
            )

    @property
    def server(self) -> Server:
        """Return the underlying MCP Server instance."""
        return self._server
