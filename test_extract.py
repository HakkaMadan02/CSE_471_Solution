import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_tavily import TavilyExtract

project_dir = Path(__file__).resolve().parent
load_dotenv(project_dir / ".env")

if not os.getenv("TAVILY_API_KEY"):
    raise RuntimeError("TAVILY_API_KEY is missing from .env")

output_dir = project_dir / "output"

with (output_dir / "search_test.json").open(encoding="utf-8") as file:
    search_result = json.load(file)

results = search_result.get("results", [])
if not results:
    raise RuntimeError("No search results found. Run test_search.py first.")

url = results[0]["url"]
print("Extracting:", url)

extract = TavilyExtract(extract_depth="basic")
result = extract.invoke({"urls": [url]})

if isinstance(result, str):
    result = json.loads(result)

with (output_dir / "extract_test.json").open("w", encoding="utf-8") as file:
    json.dump(result, file, indent=2, ensure_ascii=False)

pages = result.get("results", [])

if not pages:
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise RuntimeError("No page content was extracted. See response above.")

for page in pages:
    text = page.get("raw_content", "")
    print("\nURL:", page.get("url", ""))
    print("Extracted characters:", len(text))
    print("\nFirst 1,500 characters:\n")
    print(text[:1500])

print("\nFull response saved to output/extract_test.json")