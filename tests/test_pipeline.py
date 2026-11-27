"""The detection and k-sweep scripts on fake activations, and the Table 5 analysis on the released outputs."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from synthetic import write_activations

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "tests" / "synthetic_config.yaml"


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)], check=True)


@pytest.fixture(scope="module")
def detection(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipeline")
    write_activations(tmp / "activations", per_dataset=80, d_inter=96)
    run(
        "run_detection.py",
        "--config",
        CONFIG,
        "--model",
        "toy",
        "--activations-dir",
        tmp / "activations",
        "--work-dir",
        tmp / "work",
        "--out",
        tmp / "detection",
    )
    return tmp


def test_detection_finds_planted_signal(detection):
    result = json.loads((detection / "detection" / "toy.json").read_text())
    assert result["section_A"]["n_samples_after_filter"] < 320, "unlabeled examples and the extra dataset are dropped"
    assert set(result["section_A"]["per_dataset"]) == {
        "pubmedqa_labeled",
        "medqa_usmle",
        "medmcqa",
        "pubmedqa_artificial",
    }
    assert result["section_B"]["best_conditions"]["raw_gen0"]["auroc"] > 0.8
    assert set(result["section_D"]["transfer_matrix"]) == set(result["section_A"]["per_dataset"])


def test_ksweep(detection):
    run(
        "run_ksweep.py",
        "--config",
        CONFIG,
        "--model",
        "toy",
        "--activations-dir",
        detection / "activations",
        "--work-dir",
        detection / "work",
        "--detection",
        detection / "detection",
        "--out",
        detection / "ksweep",
    )
    sweep = json.loads((detection / "ksweep" / "toy.json").read_text())
    assert list(sweep["random_k"]) == ["1", "10", "30"]
    assert sweep["top_k"]["10"]["auroc"] > sweep["random_k"]["10"]["mean"]


def test_table5_reproduces_from_released_outputs(tmp_path):
    interventions = ROOT / "outputs" / "interventions"
    if len(list(interventions.glob("*.parquet"))) < 16:
        pytest.skip("run scripts/download_interventions.py first")
    run("analyze_interventions.py", "--interventions", interventions, "--out", tmp_path)
    rerun = json.loads((tmp_path / "analysis.json").read_text())
    saved = json.loads((ROOT / "results" / "interventions" / "analysis.json").read_text())
    for name, pooled in saved["pooled"].items():
        assert rerun["pooled"][name]["mean"] == pytest.approx(pooled["mean"], abs=1e-12)
        assert rerun["pooled"][name]["ci"] == pytest.approx(pooled["ci"], abs=1e-12)
    for key, verdict in saved["verdicts"].items():
        assert rerun["verdicts"][key]["min_mcnemar_p_holm"] == pytest.approx(verdict["min_mcnemar_p_holm"], abs=1e-12)
