import asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

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


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item["text"] for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content)


async def build_agent():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )

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

    mcp_client = MultiServerMCPClient(
        {
            "weather": {"command": "python", "args": ["weather_server.py"], "transport": "stdio"},
            "currency": {"command": "python", "args": ["currency_server.py"], "transport": "stdio"},
        }
    )
    mcp_tools = await mcp_client.get_tools()
    all_tools = mcp_tools + [search_travel_knowledge_base]

    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
    checkpointer = InMemorySaver()
    agent = create_react_agent(
        llm, all_tools, prompt=SYSTEM_PROMPT, checkpointer=checkpointer
    )
    return agent


# --- One-time setup, persisted across Streamlit re-runs ---
if "loop" not in st.session_state:
    st.session_state.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.loop)

if "agent" not in st.session_state:
    with st.spinner("Setting up the travel assistant (loading tools and knowledge base)..."):
        st.session_state.agent = st.session_state.loop.run_until_complete(build_agent())
    st.session_state.config = {"configurable": {"thread_id": "streamlit-session"}}
    st.session_state.messages = []

# --- UI ---
st.title("Singapore Travel Planning Assistant")
st.caption("Ask about attractions, transport, culture, weather, or currency conversion.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask me about your Singapore trip...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.loop.run_until_complete(
                st.session_state.agent.ainvoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config=st.session_state.config,
                )
            )
            answer = extract_text(result["messages"][-1].content)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})