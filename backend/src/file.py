import asyncio
from pathlib import Path
from textwrap import dedent

from agno.agent import Agent
# from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
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
    # https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem
    async with MCPTools(f"npx -y @modelcontextprotocol/server-filesystem {file_path}") as mcp_tools:
        agent = Agent(
            # model=OpenAIChat(id="gpt-4o", base_url=os.environ.get("OPENAI_BASE_URL"), api_key=os.environ.get("OPENAI_API_KEY")),
            model=Ollama(id="qwen2.5:32b"),
            tools=[mcp_tools],
            instructions=dedent(f"""\
You are a helpful filesystem assistant. You can interact with a virtual filesystem using a set of predefined tools. 

⚠️ Your access is strictly limited to the following directory:
- {file_path}

You MUST follow these principles:
- 🗂️ Before accessing any file or directory, use `list_allowed_directories` to confirm your allowed root.
- 🔍 To find files, always use `search_files` with the correct base path (e.g., `{file_path}`), never use `/`.
- 📄 Use `list_directory`, `read_file`, and `get_file_info` to explore file contents or metadata.
- ✍️ For writing or editing files, use `write_file` or `edit_file` only within the allowed path.

🚫 You are not allowed to access or reference any path outside the allowed directory. Do NOT use absolute root paths like `/`.

🧠 Use markdown formatting in your responses.
- Use **bold** for filenames or directories
- Use `code blocks` for command results
- Use bullet lists for file listings
- Use headings (##) to organize your output

🎯 Your goal is to help the user understand, search, edit, or analyze the file structure within the allowed path. Be clear and concise.

Example tasks you can help with:
- “List all files in the folder”
- “Show me the README.md content”
- “Search for license-related files”
- “Edit a file and replace a word”

"""),
            markdown=True,
            show_tool_calls=True,
            debug_mode=True
        )

        # Run the agent
        await agent.aprint_response(message, stream=False)


# Example usage
if __name__ == "__main__":
    # Basic example - exploring project license
    print("OPENAI_BASE_URL:", os.environ.get("OPENAI_BASE_URL"))
    print("LOCAL_API_BASE:", os.environ.get("LOCAL_API_BASE"))
    asyncio.run(run_agent("What is the /Users/stepbystep/Documents/CodeWay/LICENSE for CodeWay project? not sub package license"))