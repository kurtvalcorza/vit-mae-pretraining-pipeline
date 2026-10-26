"""Tier C review findings T109-1..T109-4 (2026-10-25)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "vit_mae_pretraining_colab.ipynb"


def _nb():
    return json.loads(NB.read_text(encoding="utf-8"))


def _all_text():
    md = "\n".join("".join(c["source"]) for c in _nb()["cells"])
    return " ".join(md.split())


def test_t109_1_dat9_imagenet_classes():
    text = _all_text()
    assert "out of the checkpoint's ImageNet-1k pretraining domain" not in text
    assert "photographs it was never trained on" not in text
    assert "**Overlap with the training data:**" in text and "`n01532829`" in text


def test_t109_2_runs_named():
    text = _all_text()
    assert "Kaggle T4 release run" not in text and "release run" not in text
    assert "`78c4c1bd`" in text


def test_t109_3_spec_items():
    nb = _nb()
    dimer = nb["metadata"]["dimer"]
    assert dimer["notebook_spec"] == "2.5" and dimer["notebook_kind"] == "TUTORIAL"
    first_code = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    head = "\n".join("".join(c["source"]) for c in nb["cells"][:first_code])
    assert "Notice.** For research, training and evaluation only." in head
    text = _all_text()
    assert "[1] K. He" in text and "`DIMER_ISOLATED_ENV`" in text


def test_t109_4_observer_wording():
    text = _all_text()
    assert "60 per species, one per observer)" not in text
