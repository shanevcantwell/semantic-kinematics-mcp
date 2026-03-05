"""
MCP Server for semantic-kinematics tools.

Entry point for JSON-RPC over stdio communication.
Follows surf-mcp patterns for tool registration and dispatch.

Usage:
    semantic-kinematics-mcp  # starts server, waits for JSON-RPC
"""

from dotenv import load_dotenv
load_dotenv()  # Must run before any StateManager construction reads env vars

import asyncio
import json
from typing import Any, Dict, List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from semantic_kinematics.mcp.state_manager import StateManager
from semantic_kinematics.mcp.commands import embeddings, classification, trajectory, model


# Initialize server and state
server = Server("semantic-kinematics")
state_manager = StateManager()


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List all available tools."""
    tools = []
    tools.extend(embeddings.get_tools())
    tools.extend(classification.get_tools())
    tools.extend(trajectory.get_tools())
    tools.extend(model.get_tools())
    return tools


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """
    Dispatch tool calls to appropriate handlers.

    Returns JSON-serialized results as TextContent.
    """
    try:
        # Embedding tools
        if name == "embed_text":
            result = await embeddings.embed_text(state_manager, arguments)
        elif name == "calculate_drift":
            result = await embeddings.calculate_drift(state_manager, arguments)

        # Classification tools
        elif name == "classify_document":
            result = await classification.classify_document(state_manager, arguments)

        # Trajectory tools
        elif name == "analyze_trajectory":
            result = await trajectory.analyze_trajectory(state_manager, arguments)
        elif name == "compare_trajectories":
            result = await trajectory.compare_trajectories_handler(state_manager, arguments)

        # Model management tools
        elif name == "model_status":
            result = await model.model_status(state_manager, arguments)
        elif name == "model_load":
            result = await model.model_load(state_manager, arguments)
        elif name == "model_unload":
            result = await model.model_unload(state_manager, arguments)

        # Unknown tool
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str)
        )]

    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(e)})
        )]


async def run_server():
    """Run the MCP server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


def main():
    """Entry point for the MCP server."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
