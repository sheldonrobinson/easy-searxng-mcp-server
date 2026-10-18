#!/bin/bash
# Use uvx if available for managed Python environments, otherwise fallback to python3
if command -v uvx &>/dev/null; then
    exec uvx --from easy-searxng-mcp-server easy-searxng-mcp-server "$@"
else
    exec python3 src/main.py "$@"
fi
