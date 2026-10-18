import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_tavily import TavilySearch

project_dir = Path(__file__).resolve().parent
load_dotenv(project_dir / ".env")

if not os.getenv("TAVILY_API_KEY"):
    raise RuntimeError("TAVILY_API_KEY is missing from .env")

search = TavilySearch(
    max_results=3,
    topic="general",
    search_depth="basic",
    include_answer=False,
    include_raw_content=False,
)

result = search.invoke({
    "query": "site.fdic.gov deposit insurance coverage limit"
})

if isinstance(result, str):
    result = json.loads(result)

output_dir = project_dir / "output"
output_dir.mkdir(exist_ok=True)

with (output_dir / "search_test.json").open("w", encoding="utf-8") as file:
    json.dump(result, file, indent=2, ensure_ascii=False)

for index, item in enumerate(result.get("results", []), start=1):
    print(f"\nResult {index}")
    print("Title:", item.get("title", ""))
    print("URL:", item.get("url", ""))
    print("Excerpt:", item.get("content", "")[:500])

print("\nFull response saved to output/search_test.json")