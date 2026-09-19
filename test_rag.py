from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
def extract_text(content):
    """Gemini sometimes returns content as a list of blocks (text + internal signatures)
    instead of a plain string. This pulls out just the readable text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)

def ask(question, k=4):
    # Step 1: Retrieve the most relevant chunks
    results = vector_store.similarity_search(question, k=k)

    # Step 2: Build context text + collect sources for citation
    context = "\n\n---\n\n".join(doc.page_content for doc in results)
    sources = {doc.metadata["source_title"] for doc in results}  # set = unique

    # Step 3: Ask the LLM to answer USING ONLY this retrieved context
    prompt = f"""Answer the question using ONLY the information in the context below.
    If the question uses subjective language like "must-visit," "best," or "top,"
    treat it as asking for recommendations drawn from what's in the context -
    you don't need an exact keyword match for words like "must-visit" to answer.
    Only say the context is insufficient if it genuinely lacks relevant information
    to address the question, not merely because it doesn't use the exact same phrasing.


Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)

    print(f"\nQ: {question}")
    print(f"\nA: {extract_text(response.content)}")
    print(f"\nSources used: {', '.join(sources)}")
    print("=" * 80)

# Try a few sample questions from the brief
ask("What are the must-visit attractions in Singapore?")
ask("How can a tourist travel around Singapore?")
ask("Suggest activities for a family with children.")