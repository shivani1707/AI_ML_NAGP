import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# --- Load the RAG vector store ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)

# --- Turn RAG retrieval into a LangChain tool the agent can call ---
@tool
def search_travel_knowledge_base(query: str) -> str:
    """
    Search the Singapore travel knowledge base for destination information:
    attractions, neighbourhoods, transportation, culture, food, and itinerary ideas.
    Use this for any question about what to see, do, or know about Singapore
    that does NOT require current/live data like weather or exchange rates.
    """
    results = vector_store.similarity_search(query, k=4)
    if not results:
        return "No relevant information found in the knowledge base."

    output = ""
    for doc in results:
        output += f"[Source: {doc.metadata['source_title']}]\n{doc.page_content}\n\n"
    return output


def extract_text(content):
    """Gemini sometimes returns content as a list of blocks (text + internal signatures)
    instead of a plain string. This pulls out just the readable text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item["text"] for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content)


SYSTEM_PROMPT = """You are a Singapore travel planning assistant.

Rules you must follow:
- For destination facts (attractions, transport, culture, food, itineraries), use the
  search_travel_knowledge_base tool. Do not invent destination facts from your own knowledge.
- For weather, use the weather tool. For currency conversion, use the currency tool.
- Use MCP tools (weather, currency) only for current/live data, never for destination facts
  already covered by the knowledge base.
- When you combine information from multiple sources, clearly label each part:
  e.g. "According to the knowledge base..." / "According to the live weather forecast..."
  / "My recommendation:" for anything you're suggesting rather than stating as fact.
- If the knowledge base and tools don't have enough information to answer, say so
  clearly instead of guessing.
- Remember earlier parts of this conversation and use them for follow-up questions.
"""


async def main():
    mcp_client = MultiServerMCPClient(
        {
            "weather": {"command": "python", "args": ["weather_server.py"], "transport": "stdio"},
            "currency": {"command": "python", "args": ["currency_server.py"], "transport": "stdio"},
        }
    )
    mcp_tools = await mcp_client.get_tools()
    all_tools = mcp_tools + [search_travel_knowledge_base]

    print(f"Agent has {len(all_tools)} tools: {[t.name for t in all_tools]}\n", flush=True)

    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

    checkpointer = InMemorySaver()
    agent = create_react_agent(
        llm, all_tools, prompt=SYSTEM_PROMPT, checkpointer=checkpointer
    )

    # A fixed thread_id keeps all turns in this session connected as one conversation
    config = {"configurable": {"thread_id": "session-1"}}

    print("Singapore Travel Assistant — type 'quit' to exit\n")
    while True:
        question = input("You: ")
        if question.strip().lower() == "quit":
            break

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]},
            config=config,
        )
        final_message = result["messages"][-1]
        print(f"\nAssistant: {extract_text(final_message.content)}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()