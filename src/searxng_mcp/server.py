"""MCP server - serve() function for running an MCP server over HTTP."""

import asyncio
import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server


logger = logging.getLogger(__name__)


async def serve(
    instance_url: str,
    timeout: int = 30,
    transport: str = "http",
    host: str = "127.0.0.1",
    port: int = 8000,
):
    """Run an MCP server for SearXNG search.

    Args:
        instance_url: URL of the SearXNG instance to proxy.
        timeout: HTTP request timeout in seconds.
        transport: Transport protocol ("http" or "stdio").
        host: Host to bind to (for HTTP transport).
        port: Port to bind to (for HTTP transport).
    """
    from .adapters import HttpSearchAdapter
    from .client import SearchPort

    adapter = HttpSearchAdapter(instance_url=instance_url, timeout=timeout)

    class MCPSearchPort(SearchPort):
        def __init__(self, adapter: HttpSearchAdapter):
            self._adapter = adapter

        async def search(self, query, parameters):
            return await self._adapter.search(query, parameters)

        async def close(self):
            await self._adapter.close()

    search_port = MCPSearchPort(adapter)
    server = Server("easy-searxng-mcp-server")

    @server.list_tools()
    async def handle_list_tools():
        from mcp.types import ListToolsResult, Tool

        return ListToolsResult(
            tools=[
                Tool(
                    name="web_search",
                    description="Search the web via SearXNG",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "categories": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": ["general"],
                            },
                            "engines": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": ["google", "bing", "duckduckgo"],
                            },
                            "language": {"type": "string", "default": "auto"},
                            "max_results": {"type": "integer", "default": 10},
                            "time_range": {
                                "type": "string",
                                "enum": ["day", "week", "month", "year"],
                                "default": "all",
                            },
                        },
                        "required": ["query"],
                    },
                )
            ]
        )

    @server.call_tool()
    async def handle_call_tool(name, arguments):
        from mcp.types import CallToolResult, TextContent

        if name != "web_search":
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {name}")],
                isError=True,
            )

        query = arguments.get("query", "")
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
            from .client import SearchQuery, SearchParameters

            params = SearchParameters(
                categories=tuple(arguments.get("categories", ["general"])),
                engines=tuple(
                    arguments.get(
                        "engines", ["google", "bing", "duckduckgo"]
                    )
                ),
                language=arguments.get("language", "auto"),
                max_results=arguments.get("max_results", 10),
                time_range=arguments.get("time_range"),
            )

            result = await search_port.search(
                SearchQuery(text=query), parameters=params
            )

            formatted = []
            for i, r in enumerate(result.results):
                formatted.append(
                    f"[{i + 1}] {r.title}\nURL: {r.url}\n{r.content}\n"
                )

            return CallToolResult(
                content=[TextContent(type="text", text="\n".join(formatted))]
            )
        except Exception as exc:
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

    if transport == "stdio":
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    else:
        import uvicorn
        from mcp.server.streamable_http import StreamableHTTPServerTransport

        # For streamable HTTP transport, we need to create an ASGI app
        # This is a simplified version - production should use proper MCP server
        async def asgi_app(scope, receive, send):
            if scope["type"] == "lifespan":
                while True:
                    message = await receive()
                    if message["type"] == "lifespan.startup":
                        await send({"type": "lifespan.startup.complete"})
                    elif message["type"] == "lifespan.shutdown":
                        await send({"type": "lifespan.shutdown.complete"})
                        return
            elif scope["type"] == "http":
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [[b"content-type", b"text/plain"]],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b"MCP server running",
                    }
                )

        config = uvicorn.Config(app=asgi_app, host=host, port=port, log_level="info")
        server_instance = uvicorn.Server(config)
        await server_instance.serve()
