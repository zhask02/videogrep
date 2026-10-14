"""Extract uniformly sampled frames from local VideoGrep clips."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

VIDEO_DIR = Path("data/videos")
FRAME_DIR = Path("data/cache/frames")
MANIFEST_PATH = Path("data/cache/frame_manifest.jsonl")
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi"}


def extract_clip(
    video_path: Path,
    output_dir: Path,
    interval: float,
    force: bool = False,
) -> dict:
    """Extract frames and save their timestamps in filenames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"

    if metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("interval_seconds") == interval:
            existing = list(output_dir.glob("frame_*.jpg"))
            if len(existing) == metadata.get("frame_count", -1) and existing:
                return {**metadata, "status": "skipped_existing"}

    # Remove old frames if the interval changed or a forced re-extraction was requested.
    for old_frame in output_dir.glob("frame_*.jpg"):
        old_frame.unlink()
    if metadata_path.exists():
        metadata_path.unlink()

    output_pattern = output_dir / "frame_%06d.jpg"
    filter_expression = (
        f"fps=1/{interval},scale='min(384,iw)':-2"
    )

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        filter_expression,
        "-q:v",
        "3",
        "-start_number",
        "0",
        "-y",
        str(output_pattern),
    ]

    subprocess.run(command, check=True, capture_output=True, text=True)

    frames = sorted(output_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError(f"FFmpeg produced no frames for {video_path.name}")

    metadata = {
        "clip_id": video_path.stem,
        "video_path": str(video_path),
        "interval_seconds": interval,
        "frame_count": len(frames),
        "frames": [
            {
                "path": str(frame),
                "timestamp_seconds": index * interval,
            }
            for index, frame in enumerate(frames)
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {**metadata, "status": "extracted"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=0, help="Limit clips; 0 means all")
    parser.add_argument("--force", action="store_true", help="Re-extract existing frames")
    args = parser.parse_args()

    if args.interval <= 0:
        parser.error("--interval must be greater than zero")

    if not VIDEO_DIR.exists():
        raise SystemExit(f"Video directory not found: {VIDEO_DIR}")

    videos = sorted(
        path for path in VIDEO_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if args.limit > 0:
        videos = videos[:args.limit]

    if not videos:
        raise SystemExit(f"No video files found in {VIDEO_DIR}")

    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for index, video in enumerate(videos, start=1):
        output_dir = FRAME_DIR / video.stem
        print(f"[{index}/{len(videos)}] {video.name}", flush=True)

        try:
            result = extract_clip(video, output_dir, args.interval, args.force)
            results.append(result)
            print(
                f"  {result['status']}: {result['frame_count']} frames",
                flush=True,
            )
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            error = str(exc)
            if isinstance(exc, subprocess.CalledProcessError):
                error = (exc.stderr or str(exc)).strip()
            results.append({
                "clip_id": video.stem,
                "video_path": str(video),
                "status": "failed",
                "error": error,
            })
            print(f"  FAILED: {error}", flush=True)

    # Rewrite a deterministic manifest for this run.
    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result) + "\n")

    successful = sum(result["status"] != "failed" for result in results)
    total_frames = sum(result.get("frame_count", 0) for result in results)
    print("\nFRAME EXTRACTION SUMMARY")
    print(f"Clips processed: {len(results)}")
    print(f"Successful:      {successful}")
    print(f"Failed:          {len(results) - successful}")
    print(f"Frames:          {total_frames}")
    print(f"Manifest:        {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
