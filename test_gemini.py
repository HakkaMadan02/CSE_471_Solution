import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(Path(__file__).resolve().parent / ".env")

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY is missing from .env")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key,
    temperature=0,
    max_retries=0,
    timeout=60,
)

response = llm.invoke(
    "In one short sentences, explain what an evidence-seeking "
    "web research agent does. Do not search the web."
)

print(response.text)
print("\nToken usage:", response.usage_metadata)