
"""Search video clips using text-to-CLIP similarity and a FAISS index."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPModel

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "cache" / "clip_index"
MODEL_ID = "openai/clip-vit-base-patch32"


def load_index():
    """Load the saved FAISS index and its corresponding clip metadata."""
    index_path = INDEX_DIR / "clip_index.faiss"
    clips_path = INDEX_DIR / "clips.jsonl"
    summary_path = INDEX_DIR / "summary.json"

    for path in (index_path, clips_path, summary_path):
        if not path.is_file():
            raise SystemExit(
                f"Missing index file: {path.relative_to(ROOT)}. "
                "Build the index first with scripts/build_clip_index.py."
            )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["model_id"] != MODEL_ID:
        raise SystemExit(
            f"Index model is {summary['model_id']}, but search expects {MODEL_ID}."
        )

    with clips_path.open("r", encoding="utf-8") as file:
        clips = [json.loads(line) for line in file if line.strip()]

    index = faiss.read_index(str(index_path))

    if len(clips) != index.ntotal:
        raise SystemExit(
            f"Metadata/index mismatch: {len(clips)} clips but "
            f"{index.ntotal} FAISS entries."
        )

    if index.d != summary["embedding_dimension"]:
        raise SystemExit(
            f"Index dimension {index.d} does not match the saved summary."
        )

    return index, clips


def embed_query(query: str, processor: AutoProcessor, model: CLIPModel):
    """Encode one text query as a normalized CLIP vector."""
    inputs = processor(text=[query], return_tensors="pt", padding=True)

    with torch.inference_mode():
        text_outputs = model.get_text_features(**inputs)
        text_features = (
            text_outputs.pooler_output
            if hasattr(text_outputs, "pooler_output")
            else text_outputs
        )
        text_embedding = F.normalize(text_features.float(), p=2, dim=-1)

    return text_embedding.cpu().numpy()


def search(query: str, top_k: int = 5) -> list[dict]:
    """Return the top matching clips and their cosine similarity scores."""
    if not query.strip():
        raise ValueError("Search query cannot be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    index, clips = load_index()

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID)
    model.eval()

    query_embedding = embed_query(query, processor, model)
    k = min(top_k, index.ntotal)

    started = time.perf_counter()
    scores, indices = index.search(query_embedding, k)
    search_seconds = time.perf_counter() - started

    results = []
    for rank, (score, idx) in enumerate(
        zip(scores[0].tolist(), indices[0].tolist()), start=1
    ):
        if idx < 0:
            continue

        clip = clips[idx]
        video_path = ROOT / clip["video_path"]

        results.append(
            {
                "rank": rank,
                "clip_id": clip["clip_id"],
                "score": float(score),
                "video_path": clip["video_path"],
                "video_exists": video_path.is_file(),
                "frame_count": clip["frame_count"],
            }
        )

    print(f"\nQuery: {query}")
    print(f"Search time (FAISS only): {search_seconds * 1000:.3f} ms")
    print("Scores are cosine similarities, not probabilities.\n")

    for result in results:
        print(
            f"{result['rank']}. {result['clip_id']} "
            f"| score={result['score']:.4f} "
            f"| frames={result['frame_count']} "
            f"| video_exists={result['video_exists']}"
        )
        print(f"   {result['video_path']}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find video clips matching a natural-language query."
    )
    parser.add_argument("query", help="Text describing the visual content.")
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results to return."
    )
    args = parser.parse_args()

    try:
        search(args.query, args.top_k)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
