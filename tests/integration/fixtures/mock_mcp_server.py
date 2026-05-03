"""Minimal MCP server fixture for integration tests.

Run directly to start the server::

    python tests/integration/fixtures/mock_mcp_server.py

The server exposes two tools:
- ``echo(message: str) -> str``   — returns "ECHO: {message}"
- ``add(a: int, b: int) -> str``  — returns the sum as a string
"""
from mcp.server.fastmcp import FastMCP

app = FastMCP("mock-test-server")


@app.tool()
def echo(message: str) -> str:
    """Echo back the message with a prefix."""
    return f"ECHO: {message}"


@app.tool()
def add(a: int, b: int) -> str:
    """Add two integers and return the result."""
    return str(a + b)


if __name__ == "__main__":
    app.run()
