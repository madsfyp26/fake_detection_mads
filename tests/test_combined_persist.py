import os
import shutil
import tempfile

from orchestrator.combined_runner import _persist_combined_artifacts, _safe_temp_roots_for_cleanup


def test_safe_temp_roots_filters_non_temp():
    tmp = tempfile.gettempdir()
    inside = os.path.join(tmp, "fad_test_sub")
    assert inside in _safe_temp_roots_for_cleanup([inside])
    assert "/usr" not in _safe_temp_roots_for_cleanup(["/usr/bin"])


def test_persist_copies_audio_and_roi():
    root = tempfile.mkdtemp()
    try:
        work = tempfile.mkdtemp(dir=root)
        audio = os.path.join(work, "clip.wav")
        roi = os.path.join(work, "mouth_roi.mp4")
        with open(audio, "wb") as f:
            f.write(b"wavdata")
        with open(roi, "wb") as f:
            f.write(b"roidata")

        persist = os.path.join(root, "out")
        result: dict = {
            "audio_path": audio,
            "roi_path": roi,
            "cam_overlays_dir": None,
            "cam_parent_dir": None,
        }
        _persist_combined_artifacts(
            result,
            persist_run_dir=persist,
            cleanup_volatile_after_persist=False,
        )
        assert result["persist_run_dir"] == os.path.abspath(persist)
        assert os.path.isfile(result["audio_path"])
        assert persist in result["audio_path"]
        assert os.path.isfile(result["roi_path"])
        assert os.path.isfile(audio)  # original kept
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_persist_rewrites_cam_idx_paths():
    root = tempfile.mkdtemp()
    try:
        work = tempfile.mkdtemp(dir=root)
        audio = os.path.join(work, "clip.wav")
        roi = os.path.join(work, "mouth_roi.mp4")
        ov = os.path.join(work, "overlays")
        os.makedirs(ov)
        with open(audio, "wb") as f:
            f.write(b"wavdata")
        with open(roi, "wb") as f:
            f.write(b"roidata")
        cv = os.path.join(work, "cam_volume.npy")
        fh = os.path.join(work, "fused_heatmap.npy")
        with open(cv, "wb") as f:
            f.write(b"np")
        with open(fh, "wb") as f:
            f.write(b"fh")

        persist = os.path.join(root, "persist_out")
        result: dict = {
            "audio_path": audio,
            "roi_path": roi,
            "cam_overlays_dir": ov,
            "cam_parent_dir": work,
            "cam_idx": {
                "overlay_dir": ov,
                "cam_volume_path": cv,
                "fused_heatmap_path": fh,
            },
        }
        _persist_combined_artifacts(
            result,
            persist_run_dir=persist,
            cleanup_volatile_after_persist=False,
        )
        ci = result["cam_idx"]
        assert ci["overlay_dir"] == result["cam_overlays_dir"]
        assert persist in (ci.get("cam_volume_path") or "")
        assert persist in (ci.get("fused_heatmap_path") or "")
        assert os.path.isfile(ci["cam_volume_path"])
        assert os.path.isfile(ci["fused_heatmap_path"])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_persist_cleanup_removes_volatile_under_tmp():
    root = tempfile.mkdtemp()
    try:
        work = tempfile.mkdtemp(dir=root)
        audio = os.path.join(work, "x.wav")
        with open(audio, "wb") as f:
            f.write(b"a")
        roi = os.path.join(work, "mouth_roi.mp4")
        with open(roi, "wb") as f:
            f.write(b"b")
        persist = os.path.join(root, "out2")
        result: dict = {"audio_path": audio, "roi_path": roi, "cam_overlays_dir": None}
        _persist_combined_artifacts(
            result,
            persist_run_dir=persist,
            cleanup_volatile_after_persist=True,
        )
        assert os.path.isdir(persist)
        assert not os.path.isfile(audio)
    finally:
        shutil.rmtree(root, ignore_errors=True)
