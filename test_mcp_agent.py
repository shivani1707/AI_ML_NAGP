import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

async def main():
    client = MultiServerMCPClient(
        {
            "weather": {
                "command": "python",
                "args": ["weather_server.py"],
                "transport": "stdio",
            },
            "currency": {
                "command": "python",
                "args": ["currency_server.py"],
                "transport": "stdio",
            },
        }
    )

    tools = await client.get_tools()
    print(f"Loaded {len(tools)} tools: {[t.name for t in tools]}\n")

    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
    agent = create_react_agent(llm, tools)

    async def ask(question):
        print(f"Q: {question}")
        result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
        final_message = result["messages"][-1]
        print(f"A: {final_message.content}\n")
        print("-" * 80)

    await ask("What's the weather like in Singapore right now?")
    await ask("Convert 200 SGD to INR")
    await ask("What are must-visit attractions in Singapore?")  # should NOT call a tool

asyncio.run(main())