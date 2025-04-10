import asyncio
from pathlib import Path
from textwrap import dedent

from agno.agent import Agent
from agno.models.openai.like import OpenAILike
# from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools
from mcp import StdioServerParameters
import os
import dotenv

dotenv.load_dotenv()

async def run_agent(message: str) -> None:
    """Run the filesystem agent with the given message."""

    file_path = "/Users/stepbystep/Documents/CodeWay"
    print("file_path:", file_path)

    # MCP server to access the filesystem (via `npx`)
    async with MCPTools(f"npx -y @modelcontextprotocol/server-filesystem {file_path}") as mcp_tools:
        agent = Agent(
            # model=OpenAIChat(id="gpt-4o", base_url=os.environ.get("OPENAI_BASE_URL"), api_key=os.environ.get("OPENAI_API_KEY")),
            model=OpenAILike(id="Qwen/Qwen2.5-32B-Instruct", base_url=os.environ.get("LOCAL_API_BASE")),
            tools=[mcp_tools],
            instructions=dedent("""\
                You are a filesystem assistant. Help users explore files and directories.

                - Navigate the filesystem to answer questions
                - Use the list_allowed_directories tool to find directories that you can access
                - Provide clear context about files you examine
                - Use headings to organize your responses
                - Be concise and focus on relevant information\
            """),
            markdown=True,
            show_tool_calls=True,
            debug_mode=True
        )

        # Run the agent
        await agent.aprint_response(message, stream=True)


# Example usage
if __name__ == "__main__":
    # Basic example - exploring project license
    print("OPENAI_BASE_URL:", os.environ.get("OPENAI_BASE_URL"))
    print("LOCAL_API_BASE:", os.environ.get("LOCAL_API_BASE"))
    asyncio.run(run_agent("What is the license for this project?"))