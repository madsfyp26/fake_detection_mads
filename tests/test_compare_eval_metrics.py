import json
import tempfile

import pytest

from tools.compare_eval_metrics import compare_metrics


def test_compare_metrics_detects_diffs():
    a = {"baseline_pipeline_p_fused_threshold_search": {"accuracy": 0.8}, "n_samples_ok": 10}
    b = {"baseline_pipeline_p_fused_threshold_search": {"accuracy": 0.9}, "n_samples_ok": 10}
    with tempfile.TemporaryDirectory() as d:
        pa = f"{d}/a.json"
        pb = f"{d}/b.json"
        with open(pa, "w") as fa:
            json.dump(a, fa)
        with open(pb, "w") as fb:
            json.dump(b, fb)
        r = compare_metrics(pa, pb)
    assert r["n_differing"] >= 1
    keys = {x["key"] for x in r["diffs"]}
    assert "baseline_pipeline_p_fused_threshold_search.accuracy" in keys
