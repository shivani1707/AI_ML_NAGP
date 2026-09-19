import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

async def main():
    client = MultiServerMCPClient(
        {
            "currency": {
                "command": "python",
                "args": ["currency_server.py"],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()
    print(f"Discovered {len(tools)} tool(s):")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    currency_tool = tools[0]
    result = await currency_tool.ainvoke(
        {"amount": 50000, "from_currency": "INR", "to_currency": "SGD"}
    )
    print("\nTool result:")
    print(result)

asyncio.run(main())