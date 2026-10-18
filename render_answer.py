import json
from collections import Counter
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"

runs = sorted(
    path for path in OUTPUT_DIR.iterdir()
    if path.is_dir()
    and (path / "trace.json").exists()
    and (path / "verification.json").exists()
)

if not runs:
    raise RuntimeError("No verified run found.")

#run_dir = runs[-1]
run_dir = (
    Path(sys.argv[1]).resolve()
    if len(sys.argv) > 1
    else runs[-1]
)

trace = json.loads(
    (run_dir / "trace.json").read_text(encoding="utf-8")
)
verification = json.loads(
    (run_dir / "verification.json").read_text(encoding="utf-8")
)

claims = verification["claims"]

accepted = []
flagged = []

for claim in claims:
    passes = (
        claim.get("checked_label") == "supported"
        and claim.get("quote_found") is True
        and bool(claim.get("source_url"))
    )

    if passes:
        accepted.append(claim)
    else:
        flagged.append(claim)

# Assign citation IDs from actual stored source URLs.
source_ids = {}

for claim in accepted:
    url = claim["source_url"]
    if url not in source_ids:
        source_ids[url] = f"S{len(source_ids) + 1}"

lines = [
    "# Evidence-filtered answer",
    "",
    "## Question",
    "",
    trace["question"],
    "",
    "## Supported findings",
    "",
]

if accepted:
    for claim in accepted:
        source_id = source_ids[claim["source_url"]]
        lines.append(f"- {claim['claim']} [{source_id}]")
else:
    lines.append(
        "The verification stage did not establish enough support "
        "to present any of the reviewed claims as supported findings."
    )

if flagged:
    lines.extend([
        "",
        "## Findings not accepted as supported",
        "",
    ])

    for claim in flagged:
        label = claim.get("checked_label", "insufficient")
        reason = claim.get("explanation", "")

        lines.append(
            f"- {claim['claim']}\n"
            f"  - Verification label: {label}.\n"
            f"  - Reason: {reason}"
        )

        if (
            claim.get("label") in {"supported", "contradicted"}
            and not claim.get("quote_found")
        ):
            lines.append(
                "  - Programmatic check: the proposed quotation "
                "was not found in the referenced source."
            )

if source_ids:
    lines.extend(["", "## Sources", ""])

    for url, source_id in source_ids.items():
        lines.append(f"- [{source_id}] {url}")

lines.extend([
    "",
    "## Scope of verification",
    "",
    "These findings were checked against the retrieved text saved "
    "for this run. Matching quotations and model-assessed support "
    "do not independently establish source freshness, completeness, "
    "or real-world correctness.",
    "",
])

destination = run_dir / "final_answer.md"
destination.write_text("\n".join(lines), encoding="utf-8")

label_counts = Counter(
    claim.get("checked_label", "insufficient")
    for claim in claims
)

summary = {
    "run": run_dir.name,
    "reviewed_claims": len(claims),
    "accepted_claims": len(accepted),
    "flagged_claims": len(flagged),
    "checked_label_counts": dict(label_counts),
    "verification_seconds": verification.get("verification_seconds"),
    "note": "These are evidence-audit counts, not an accuracy score.",
}

(run_dir / "verification_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("Run:", run_dir.name)
print("Reviewed claims:", len(claims))
print("Accepted claims:", len(accepted))
print("Flagged claims:", len(flagged))
print("\nSaved:", destination)