import os
import frontmatter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()
KB_FOLDER = "knowledge_base"

# --- Step A: Load all markdown files and parse their metadata ---
raw_documents = []
for filename in os.listdir(KB_FOLDER):
    if filename.endswith(".md"):
        path = os.path.join(KB_FOLDER, filename)
        post = frontmatter.load(path)  # splits header (metadata) from body
        raw_documents.append(
            Document(
                page_content=post.content,
                metadata={
                    "source_title": post.get("title", filename),
                    "source_url": post.get("source_url", ""),
                },
            )
        )
        print(f"Loaded {filename}: {len(post.content)} characters")

print(f"\nTotal source documents: {len(raw_documents)}")

# --- Step B: Split into chunks ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # characters per chunk
    chunk_overlap=150,    # overlap so context isn't lost at boundaries
)
chunks = splitter.split_documents(raw_documents)
print(f"Total chunks created: {len(chunks)}")

# --- Step C: Embed and store in FAISS ---
import time

def embed_with_retry(batch, embeddings, max_retries=5):
    """Try to build a FAISS index from a batch of chunks, retrying on rate limits."""
    for attempt in range(max_retries):
        try:
            return FAISS.from_documents(batch, embeddings)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 30 * (attempt + 1)  # 30s, 60s, 90s...
                print(f"  Rate limited. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                raise  # some other error — don't hide it, let it surface
    raise RuntimeError("Max retries exceeded for embedding batch")

# --- Step C: Embed in small batches with pauses, then merge ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

BATCH_SIZE = 10
vector_store = None

for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i : i + BATCH_SIZE]
    print(f"Embedding chunks {i} to {i + len(batch)} of {len(chunks)}...")

    batch_store = embed_with_retry(batch, embeddings)

    if vector_store is None:
        vector_store = batch_store
    else:
        vector_store.merge_from(batch_store)

    time.sleep(5)  # small pause between batches, even on success


# --- Step D: Save the vector store to disk so we don't have to rebuild it every run ---
vector_store.save_local("faiss_index")
print("Saved vector store to ./faiss_index")