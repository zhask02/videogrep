"""Convert QVHighlights val annotations into VideoGrep's queries.jsonl format."""
import argparse
import json
import pathlib
import random
import urllib.request

URL = ("https://raw.githubusercontent.com/jayleicn/moment_detr/main/"
       "data/highlight_val_release.jsonl")


def convert(rows, n_videos, seed=0):
    vids = sorted({r["vid"] for r in rows})
    random.Random(seed).shuffle(vids)
    keep = set(vids[:n_videos])
    out = []
    for r in rows:
        if r["vid"] not in keep:
            continue
        out.append({
            "query_id": f"qvh_{r['qid']}",
            "clip_id": r["vid"],
            "query": r["query"],
            "windows": r["relevant_windows"],  # [[start_s, end_s], ...] within the clip
            "query_type": "visual",
            "tolerance_s": 2.0,
            "source": "qvhighlights-val",
        })
    return sorted(keep), out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-videos", type=int, default=60)
    p.add_argument("--raw", default="data/raw/qvh_val.jsonl")
    p.add_argument("--out-dir", default="data/eval")
    a = p.parse_args()

    raw = pathlib.Path(a.raw)
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        urllib.request.urlretrieve(URL, raw)
    rows = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]

    vids, queries = convert(rows, a.n_videos)
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries) + "\n")
    (out / "videos.txt").write_text("\n".join(vids) + "\n")
    print(f"{len(vids)} videos, {len(queries)} queries -> {out}")


if __name__ == "__main__":
    main()