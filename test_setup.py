import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load the API key from .env
load_dotenv()

# Create a connection to Gemini
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# Send a simple test message
response = llm.invoke("Say hello and confirm you're working in one short sentence.")

print(response.content)