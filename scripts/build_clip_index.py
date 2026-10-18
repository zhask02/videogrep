
"""Build a pooled-CLIP clip index and retain frame-level embeddings."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, CLIPModel

ROOT = Path(__file__).resolve().parents[1]
FRAME_ROOT = ROOT / "data" / "cache" / "frames"
OUTPUT_DIR = ROOT / "data" / "cache" / "clip_index"
MODEL_ID = "openai/clip-vit-base-patch32"


def read_clip_metadata(limit_clips: int | None = None) -> list[dict]:
    """Load frame metadata and validate the referenced image files."""
    metadata_files = sorted(FRAME_ROOT.glob("*/metadata.json"))
    if limit_clips is not None:
        metadata_files = metadata_files[:limit_clips]

    if not metadata_files:
        raise SystemExit(f"No metadata.json files found under {FRAME_ROOT}")

    clips = []
    for metadata_path in metadata_files:
        with metadata_path.open("r", encoding="utf-8") as file:
            clip = json.load(file)

        valid_frames = []
        for frame in clip.get("frames", []):
            image_path = ROOT / Path(frame["path"])
            if image_path.is_file():
                valid_frames.append(
                    {
                        "path": image_path,
                        "timestamp_seconds": float(frame["timestamp_seconds"]),
                    }
                )

        if not valid_frames:
            print(f"WARNING: skipping clip with no readable frame files: {metadata_path}")
            continue

        clips.append(
            {
                "clip_id": clip["clip_id"],
                "video_path": clip["video_path"],
                "frames": valid_frames,
            }
        )

    if not clips:
        raise SystemExit("No clips with valid extracted frames were found.")

    return clips


def encode_frames(
    clips: list[dict],
    model: CLIPModel,
    processor: AutoProcessor,
    batch_size: int,
) -> tuple[np.ndarray, list[dict]]:
    """Encode every frame and retain its clip ID and timestamp."""
    frame_records = [
        {
            "clip_id": clip["clip_id"],
            "video_path": clip["video_path"],
            "frame_path": str(frame["path"].relative_to(ROOT)).replace("\\", "/"),
            "timestamp_seconds": frame["timestamp_seconds"],
        }
        for clip in clips
        for frame in clip["frames"]
    ]

    embeddings = []
    started = time.perf_counter()
    model.eval()

    for start in range(0, len(frame_records), batch_size):
        batch_records = frame_records[start : start + batch_size]
        images = []

        try:
            for record in batch_records:
                with Image.open(ROOT / record["frame_path"]) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")

            with torch.inference_mode():
                image_outputs = model.get_image_features(**inputs)
                image_features = (
                    image_outputs.pooler_output
                    if hasattr(image_outputs, "pooler_output")
                    else image_outputs
                )
                batch_embeddings = F.normalize(
                    image_features.float(), p=2, dim=-1
                )

            embeddings.append(batch_embeddings.cpu().numpy().astype(np.float32))
        finally:
            for image in images:
                image.close()

        completed = min(start + len(batch_records), len(frame_records))
        if completed == len(frame_records) or completed % 100 < batch_size:
            elapsed = time.perf_counter() - started
            print(
                f"Encoded {completed}/{len(frame_records)} frames "
                f"({elapsed:.1f}s elapsed)"
            )

    frame_embeddings = np.concatenate(embeddings, axis=0)
    return frame_embeddings, frame_records


def pool_clip_embeddings(
    clips: list[dict],
    frame_embeddings: np.ndarray,
    frame_records: list[dict],
) -> tuple[np.ndarray, list[dict]]:
    """Average each clip's frame vectors and normalize the result."""
    clip_embeddings = []
    clip_records = []
    offset = 0

    for clip in clips:
        count = len(clip["frames"])
        vectors = frame_embeddings[offset : offset + count]
        pooled = vectors.mean(axis=0, keepdims=True)
        pooled = pooled / np.maximum(
            np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12
        )
        clip_embeddings.append(pooled.astype(np.float32))
        clip_records.append(
            {
                "clip_id": clip["clip_id"],
                "video_path": clip["video_path"],
                "frame_count": count,
                "frame_start_index": offset,
                "frame_end_index": offset + count,
            }
        )
        offset += count

    if offset != len(frame_records):
        raise RuntimeError("Frame-to-clip mapping is inconsistent.")

    return np.concatenate(clip_embeddings, axis=0), clip_records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the pooled CLIP baseline index."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of images encoded per batch (default: 8).",
    )
    parser.add_argument(
        "--limit-clips",
        type=int,
        default=None,
        help="Optional clip limit for a quick smoke test.",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")
    if args.limit_clips is not None and args.limit_clips < 1:
        raise SystemExit("--limit-clips must be at least 1.")

    clips = read_clip_metadata(args.limit_clips)
    frame_count = sum(len(clip["frames"]) for clip in clips)
    print(f"Clips: {len(clips)}")
    print(f"Frames: {frame_count}")
    print(f"Loading pretrained model: {MODEL_ID}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID)
    model.eval()

    started = time.perf_counter()
    frame_embeddings, frame_records = encode_frames(
        clips, model, processor, args.batch_size
    )
    clip_embeddings, clip_records = pool_clip_embeddings(
        clips, frame_embeddings, frame_records
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(OUTPUT_DIR / "frame_embeddings.npy", frame_embeddings)
    np.save(OUTPUT_DIR / "clip_embeddings.npy", clip_embeddings)
    write_jsonl(OUTPUT_DIR / "frames.jsonl", frame_records)
    write_jsonl(OUTPUT_DIR / "clips.jsonl", clip_records)

    # Exact inner-product search over normalized pooled clip embeddings.
    index = faiss.IndexFlatIP(clip_embeddings.shape[1])
    index.add(np.ascontiguousarray(clip_embeddings))
    faiss.write_index(index, str(OUTPUT_DIR / "clip_index.faiss"))

    summary = {
        "model_id": MODEL_ID,
        "pooling": "mean of normalized frame embeddings, then L2 normalize",
        "index_type": "IndexFlatIP",
        "similarity": "cosine similarity for normalized vectors",
        "embedding_dimension": int(clip_embeddings.shape[1]),
        "clip_count": len(clip_records),
        "frame_count": len(frame_records),
        "batch_size": args.batch_size,
        "device": str(next(model.parameters()).device),
        "build_seconds": round(time.perf_counter() - started, 3),
    }
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nINDEX BUILD SUCCESSFUL")
    print(json.dumps(summary, indent=2))
    print(f"Saved index and metadata to: {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
