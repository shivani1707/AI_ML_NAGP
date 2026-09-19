import os
from langchain_community.document_loaders import WebBaseLoader

os.makedirs("knowledge_base", exist_ok=True)

title = "Wikivoyage: Singapore Travel Guide"
url = "https://en.wikivoyage.org/wiki/Singapore"

print(f"Fetching: {title}")
loader = WebBaseLoader(url)
docs = loader.load()
content = docs[0].page_content

lines = [line.strip() for line in content.split("\n") if line.strip()]
cleaned = "\n".join(lines)

header = f"---\ntitle: {title}\nsource_url: {url}\n---\n\n"

path = "knowledge_base/wikivoyage-singapore.md"
with open(path, "w", encoding="utf-8") as f:
    f.write(header + cleaned)

print(f"Saved {path} ({len(cleaned)} characters)")