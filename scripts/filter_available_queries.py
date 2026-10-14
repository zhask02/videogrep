import json
from pathlib import Path

video_dir = Path("data/videos")
query_path = Path("data/eval/queries.jsonl")
output_path = Path("data/eval/queries_available.jsonl")
report_path = Path("data/eval/coverage_report.json")

available = {p.stem for p in video_dir.glob("*.mp4")}

queries = [
    json.loads(line)
    for line in query_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

usable = [q for q in queries if q["clip_id"] in available]
missing = [q for q in queries if q["clip_id"] not in available]

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(
    "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in usable),
    encoding="utf-8",
)

report = {
    "downloaded_clips": len(available),
    "total_queries": len(queries),
    "usable_queries": len(usable),
    "missing_queries": len(missing),
    "query_coverage": round(len(usable) / len(queries), 4) if queries else 0,
    "missing_clip_ids": sorted({q["clip_id"] for q in missing}),
}
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(f"Downloaded clips : {len(available)}")
print(f"Total queries    : {len(queries)}")
print(f"Usable queries   : {len(usable)}")
print(f"Missing queries  : {len(missing)}")
print(f"Coverage         : {report['query_coverage']:.1%}")
print(f"\nFiltered queries: {output_path}")
print(f"Coverage report : {report_path}")
