import json
from pathlib import Path

project_dir = Path(__file__).resolve().parent
output_dir = project_dir / "output"

runs = sorted(
    path for path in output_dir.iterdir()
    if path.is_dir() and (path / "trace.json").exists()
)

if not runs:
    raise RuntimeError("No agent trace found.")

run_dir = runs[-1]

with (run_dir / "trace.json").open(encoding="utf-8") as file:
    trace = json.load(file)

extracted_pages = []

for message in trace["messages"]:
    if message.get("type") != "tool":
        continue

    if message.get("name") != "tavily_extract":
        continue

    content = message.get("content")

    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            print("Could not parse extraction response as JSON:")
            print(content[:1000])
            continue
    elif isinstance(content, dict):
        payload = content
    else:
        print("Unexpected tool-content format:", type(content).__name__)
        continue

    extracted_pages.extend(payload.get("results", []))

if not extracted_pages:
    raise RuntimeError("No extracted pages found in the trace.")

sections = []

for index, page in enumerate(extracted_pages, start=1):
    url = page.get("url", "")
    text = page.get("raw_content", "")

    sections.append(
        f"SOURCE {index}\nURL: {url}\n\n{text}"
    )

    print(f"\nSource {index}: {url}")
    print(f"Characters: {len(text)}")

destination = run_dir / "retrieved_pages.txt"
destination.write_text(
    "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(sections),
    encoding="utf-8",
)

print(f"\nSaved retrieved text to: {destination}")