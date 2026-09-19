import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

async def main():
    client = MultiServerMCPClient(
        {
            "weather": {
                "command": "python",
                "args": ["weather_server.py"],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()
    print(f"Discovered {len(tools)} tool(s):")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    # Call the tool directly to confirm it works
    weather_tool = tools[0]
    result = await weather_tool.ainvoke({"days": 3})
    print("\nTool result:")
    print(result)

asyncio.run(main())