import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location(
    "conv", pathlib.Path(__file__).parent.parent / "scripts" / "convert_qvhighlights.py")
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)


def test_convert_keeps_n_videos_and_fields():
    rows = [{"qid": i, "query": f"q{i}", "vid": f"v{i}_0.0_150.0",
             "relevant_windows": [[1, 5]]} for i in range(10)]
    vids, qs = conv.convert(rows, 3)
    assert len(vids) == 3
    assert len(qs) == 3
    assert qs[0]["windows"] == [[1, 5]] and qs[0]["query_type"] == "visual"