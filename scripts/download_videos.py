"""Download the labeled QVHighlights video sections.

Reads data/eval/videos.txt, where each line has the form:

    <youtube_id>_<start_seconds>_<end_seconds>

For each entry, yt-dlp is asked to retrieve only the requested section.
FFmpeg/ffprobe are then used to validate the resulting MP4.

Downloaded media and manifests live under data/ and should not be committed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

VIDEO_ID_RE = re.compile(
    r"^(?P<video_id>.+)_(?P<start>\d+(?:\.\d+)?)_(?P<end>\d+(?:\.\d+)?)$"
)


@dataclass
class DownloadResult:
    clip_id: str
    video_id: str
    start: float
    end: float
    output: str
    status: str
    duration: float | None = None
    size_bytes: int | None = None
    elapsed_seconds: float | None = None
    error: str | None = None


def parse_clip_id(clip_id: str) -> tuple[str, float, float]:
    """Parse <video_id>_<start>_<end>."""
    match = VIDEO_ID_RE.match(clip_id)

    if not match:
        raise ValueError(f"Invalid clip ID: {clip_id}")

    video_id = match.group("video_id")
    start = float(match.group("start"))
    end = float(match.group("end"))

    if end <= start:
        raise ValueError(f"End must be greater than start: {clip_id}")

    return video_id, start, end


def load_clip_ids(path: Path) -> list[str]:
    """Read non-empty, non-comment clip IDs."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    clip_ids = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        clip_ids.append(line)

    return clip_ids


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a command and capture stdout/stderr."""
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )


def validate_video(path: Path) -> tuple[bool, float | None, str | None]:
    """Validate that ffprobe can read a video stream and duration."""
    if not path.exists() or path.stat().st_size == 0:
        return False, None, "output file is missing or empty"

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_type",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]

    result = run_command(command)

    if result.returncode != 0:
        return False, None, result.stderr.strip() or "ffprobe failed"

    try:
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        duration = float(data["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, None, f"invalid ffprobe output: {exc}"

    if not streams:
        return False, duration, "no video stream found"

    return True, duration, None


def download_clip(
    clip_id: str,
    output_dir: Path,
    cookies_from_browser: str | None = None,
) -> DownloadResult:
    """Download one labeled section."""
    video_id, start, end = parse_clip_id(clip_id)

    output_path = output_dir / f"{clip_id}.mp4"

    if output_path.exists():
        valid, duration, error = validate_video(output_path)

        if valid:
            return DownloadResult(
                clip_id=clip_id,
                video_id=video_id,
                start=start,
                end=end,
                output=str(output_path),
                status="skipped_existing",
                duration=duration,
                size_bytes=output_path.stat().st_size,
                elapsed_seconds=0.0,
            )

        # Don't leave a corrupt file blocking a retry.
        output_path.unlink(missing_ok=True)

    url = f"https://www.youtube.com/watch?v={video_id}"

    command = [
        "yt-dlp",
        "--js-runtimes",
        "deno",
        "--no-playlist",
        "--download-sections",
        f"*{start}-{end}",
        "-f",
        "bv*[height<=360]+ba/b[height<=360]/b",
        "--merge-output-format",
        "mp4",
        "--force-keyframes-at-cuts",
        "-o",
        str(output_path),
        url,
    ]

    if cookies_from_browser:
        command[1:1] = ["--cookies-from-browser", cookies_from_browser]

    started = time.perf_counter()
    result = run_command(command)
    elapsed = time.perf_counter() - started

    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip() or "yt-dlp failed"

        return DownloadResult(
            clip_id=clip_id,
            video_id=video_id,
            start=start,
            end=end,
            output=str(output_path),
            status="failed",
            elapsed_seconds=elapsed,
            error=error[-2000:],
        )

    valid, duration, validation_error = validate_video(output_path)

    if not valid:
        output_path.unlink(missing_ok=True)

        return DownloadResult(
            clip_id=clip_id,
            video_id=video_id,
            start=start,
            end=end,
            output=str(output_path),
            status="failed_validation",
            elapsed_seconds=elapsed,
            error=validation_error,
        )

    return DownloadResult(
        clip_id=clip_id,
        video_id=video_id,
        start=start,
        end=end,
        output=str(output_path),
        status="success",
        duration=duration,
        size_bytes=output_path.stat().st_size,
        elapsed_seconds=elapsed,
    )


def write_jsonl(path: Path, results: list[DownloadResult]) -> None:
    """Write results as JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(asdict(result)) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download QVHighlights video sections."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/eval/videos.txt"),
        help="Path to videos.txt",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/videos"),
        help="Directory for downloaded videos",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Download at most N clips",
    )

    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help="Optional browser name for yt-dlp cookies, e.g. chrome",
    )

    args = parser.parse_args()

    clip_ids = load_clip_ids(args.input)

    if args.limit is not None:
        clip_ids = clip_ids[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path("data/eval/download_manifest.jsonl")
    failure_path = Path("data/eval/download_failures.jsonl")

    results: list[DownloadResult] = []

    total = len(clip_ids)

    print(f"Found {total} clips to download.")
    print(f"Output directory: {args.output_dir}")
    print()

    for index, clip_id in enumerate(clip_ids, start=1):
        print(f"[{index}/{total}] {clip_id}")

        result = download_clip(
            clip_id,
            args.output_dir,
            args.cookies_from_browser,
        )

        results.append(result)

        if result.status in {"success", "skipped_existing"}:
            size_mb = (
                result.size_bytes / (1024 * 1024)
                if result.size_bytes is not None
                else 0
            )

            print(
                f"  OK: {result.status} | "
                f"duration={result.duration:.1f}s | "
                f"size={size_mb:.1f} MB"
            )
        else:
            print(f"  FAILED: {result.error}")

        print()

    write_jsonl(manifest_path, results)

    failures = [
        result
        for result in results
        if result.status not in {"success", "skipped_existing"}
    ]

    write_jsonl(failure_path, failures)

    successful = len(results) - len(failures)

    print("=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Requested : {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed    : {len(failures)}")
    print(f"Manifest  : {manifest_path}")
    print(f"Failures  : {failure_path}")


if __name__ == "__main__":
    main()