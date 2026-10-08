from pathlib import Path

import pytest

from scripts.download_videos import (
    load_clip_ids,
    parse_clip_id,
)


def test_parse_clip_id():
    clip_id = "wUgPzvcKK5c_210.0_360.0"

    video_id, start, end = parse_clip_id(clip_id)

    assert video_id == "wUgPzvcKK5c"
    assert start == 210.0
    assert end == 360.0


def test_parse_clip_id_rejects_invalid_range():
    with pytest.raises(ValueError):
        parse_clip_id("wUgPzvcKK5c_360.0_210.0")


def test_load_clip_ids(tmp_path: Path):
    input_file = tmp_path / "videos.txt"

    input_file.write_text(
        """
        # comment
        abc123_0.0_10.0

        xyz789_20.0_30.0
        """,
        encoding="utf-8",
    )

    result = load_clip_ids(input_file)

    assert result == [
        "abc123_0.0_10.0",
        "xyz789_20.0_30.0",
    ]