
"""Search video clips with CLIP and FAISS, with explicit latency timings."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import faiss
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPModel

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "cache" / "clip_index"
MODEL_ID = "openai/clip-vit-base-patch32"


class SearchEngine:
    """Load the model and index once, then reuse them across queries."""

    def __init__(self) -> None:
        index_started = time.perf_counter()
        self.index, self.clips, self.summary = self._load_index()
        self.index_load_s = time.perf_counter() - index_started

        model_started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = CLIPModel.from_pretrained(MODEL_ID)
        self.model.eval()
        self.model_load_s = time.perf_counter() - model_started

    @staticmethod
    def _load_index():
        index_path = INDEX_DIR / "clip_index.faiss"
        clips_path = INDEX_DIR / "clips.jsonl"
        summary_path = INDEX_DIR / "summary.json"

        for path in (index_path, clips_path, summary_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {path.relative_to(ROOT)}. "
                    "Run scripts/build_clip_index.py first."
                )

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["model_id"] != MODEL_ID:
            raise ValueError(
                f"Index model is {summary['model_id']}, expected {MODEL_ID}."
            )

        with clips_path.open("r", encoding="utf-8") as file:
            clips = [json.loads(line) for line in file if line.strip()]

        index = faiss.read_index(str(index_path))

        if len(clips) != index.ntotal:
            raise ValueError(
                f"Metadata/index mismatch: {len(clips)} clips, "
                f"{index.ntotal} FAISS entries."
            )
        if index.d != summary["embedding_dimension"]:
            raise ValueError("FAISS dimension does not match summary.json.")

        return index, clips, summary

    def _embed_query(self, query: str) -> tuple[Any, float]:
        """Time tokenization, text encoding, normalization and CPU transfer."""
        started = time.perf_counter()
        inputs = self.processor(text=[query], return_tensors="pt", padding=True)

        with torch.inference_mode():
            output = self.model.get_text_features(**inputs)
            features = (
                output.pooler_output
                if hasattr(output, "pooler_output")
                else output
            )
            embedding = F.normalize(features.float(), p=2, dim=-1)

        embedding_array = embedding.cpu().numpy()
        encode_ms = (time.perf_counter() - started) * 1000
        return embedding_array, encode_ms

    def search(self, query: str, top_k: int = 5) -> dict:
        """Search clips and return results with component timings."""
        total_started = time.perf_counter()

        if not query.strip():
            raise ValueError("Search query cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_embedding, encode_ms = self._embed_query(query)

        search_started = time.perf_counter()
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)
        search_ms = (time.perf_counter() - search_started) * 1000

        prepare_started = time.perf_counter()
        results = []

        for rank, (score, idx) in enumerate(
            zip(scores[0].tolist(), indices[0].tolist()), start=1
        ):
            if idx < 0:
                continue

            clip = self.clips[idx]
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

        prepare_ms = (time.perf_counter() - prepare_started) * 1000
        total_ms = (time.perf_counter() - total_started) * 1000

        timings = {
            "encode_ms": encode_ms,
            "search_ms": search_ms,
            "prepare_ms": prepare_ms,
            "total_ms": total_ms,
            "unaccounted_ms": total_ms - encode_ms - search_ms - prepare_ms,
        }

        return {
            "query": query,
            "top_k": top_k,
            "timings": timings,
            "results": results,
        }


_DEFAULT_ENGINE: SearchEngine | None = None


def search(query: str, k: int = 5) -> dict:
    """Convenience API; the default engine is loaded only once per process."""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = SearchEngine()
    return _DEFAULT_ENGINE.search(query, k)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find video clips matching a natural-language query."
    )
    parser.add_argument("query", help="Text describing the visual content.")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    try:
        engine = SearchEngine()
        response = engine.search(args.query, args.top_k)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Query: {response['query']}")
    print(f"Index load: {engine.index_load_s:.3f} s")
    print(f"Model load: {engine.model_load_s:.3f} s")
    print("Query timings (milliseconds):")
    print(json.dumps(response["timings"], indent=2))
    print("Scores are cosine similarities, not probabilities.\n")

    for result in response["results"]:
        print(
            f"{result['rank']}. {result['clip_id']} "
            f"| score={result['score']:.4f} "
            f"| frames={result['frame_count']} "
            f"| video_exists={result['video_exists']}"
        )
        print(f"   {result['video_path']}")


if __name__ == "__main__":
    main()
