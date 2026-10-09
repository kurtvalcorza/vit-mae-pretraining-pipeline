"""Regression tests for the 2026-10-05 notebook review findings (MAE-M1..M3, MAE-m1..m3).

Every test needs only CI's dependencies and no model: the notebook's own cell sources are executed with stand-ins
where a model would be needed, and the torch-backed pipeline is checked statically (the package imports torch). Stand-in evidence is plumbing evidence, not model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest


def _load_samples():
    package = types.ModuleType("_mae_review_pkg")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "src" / "vit_mae_pipeline")]
    sys.modules["_mae_review_pkg"] = package
    return importlib.import_module("_mae_review_pkg.samples")


sm = _load_samples()

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "vit_mae_pretraining_colab.ipynb"
LOCK = ROOT / "tutorials" / "requirements-colab.lock.txt"
PIPELINE = ROOT / "src" / "vit_mae_pipeline" / "pipeline.py"
STEM = "vit_mae_pretraining"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in _code_cells(notebook) if marker in c["source"]]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _markdown(notebook: dict) -> str:
    return "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")


# --- MAE-M1: no in-kernel install, no restart, idempotent Section 1 ------------------------------------------


def test_mae_m1_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(notebook):
    code = "\n".join(c["source"] for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "Restart the runtime" not in json.dumps(notebook)
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in c["source"]]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = kernel[0]["source"]
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256", 'MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source


def test_mae_m1_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(notebook):
    source = _cell(notebook, "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    spec = importlib.util.spec_from_file_location("_review_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    build.check_lock(build._pins(ROOT), lock_text)


@pytest.mark.skipif(sys.platform != "linux", reason="the worker protocol uses Linux pass_fds and a symlinked interpreter (as in rtdetr-detection-pipeline 0feefe5)")
def test_mae_m1_section_1_is_idempotent_and_keeps_the_live_worker(notebook, tmp_path, monkeypatch, capsys):
    """The real Section 1 cell, run twice with a stand-in interpreter: the matching environment is reused (no
    download) and the live worker — with every variable later cells created — is kept."""
    source = _cell(notebook, "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = types.SimpleNamespace(input_transformers_cleanup=[])
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        exec(compile(source, "<section 1>", "exec"), namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("print('value', learner_value)\n")
        assert "value 42" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]
    finally:
        runtime.close()


def test_mae_m1_routed_cells_do_not_import_ipython(notebook):
    """Every cell after Section 1 runs in the isolated environment, which has no IPython: a routed cell that imports
    IPython.display would fail or lose its output there (table-transformer-detection-pipeline 67c5153). The worker
    injects `display` into the cell namespace instead."""
    routed = [c["source"] for c in _code_cells(notebook) if "# dimer: kernel cell" not in c["source"]]
    assert routed and not [s for s in routed if re.search(r"^\s*(from|import) IPython", s, re.M)]
    assert "_main.__dict__.update(__builtins__=builtins, display=display)" in _cell(notebook, "# dimer: kernel cell")


# --- MAE-M2: every adaptation starts from the pinned base -------------------------------------------------------


def test_mae_m2_adapt_and_load_artifact_restore_the_base_first():
    """Torch-backed, so checked statically: pre-call state kept, base restored, then epoch 0 (the frozen model)."""
    text = PIPELINE.read_text(encoding="utf-8")
    adapt = text[text.index("    def adapt(") : text.index("    def save_artifact(")]
    order = [adapt.index(m) for m in ("previous_state = {", "restored = self.restore_base()", "self._remember_base(names)", '"note": "frozen model"', "for epoch in range(1, epochs + 1):")]
    assert order == sorted(order)
    failure = adapt[adapt.index("except BaseException:") :]
    assert "**initial_state, **previous_state}" in failure and failure.index("self.adapter, self.probe = previous_adapter, previous_probe") < failure.index("raise")
    assert '"started_from": "pinned base"' in adapt
    load = text[text.index("    def load_artifact(") : text.index("    def from_artifact(")]
    assert load.index("self.restore_base()") < load.index("self._remember_base(sorted(model_tensors))") < load.index("self.model.load_state_dict(merged")
    restore = text[text.index("    def restore_base(") : text.index("    def _trainable_names(")]
    assert "{**state, **self._base_state}" in restore and "self.adapter, self.probe = None, None" in restore


def test_mae_m2_restore_base_reports_only_tensors_that_differ_from_the_base():
    """A first adapt() on the untouched base must say "pinned base", not "restored N tensors" (t5-base 93a578f).
    Runs the real method on a two-tensor stand-in model; skipped where torch is absent (CI)."""
    torch = pytest.importorskip("torch")
    from vit_mae_pipeline.pipeline import ViTMAEPipeline

    pipe = object.__new__(ViTMAEPipeline)
    pipe.model = torch.nn.Linear(2, 2)
    pipe._base_state, pipe.adapter, pipe.probe = {}, None, None
    names = ["bias", "weight"]
    pipe._remember_base(names)
    assert pipe.restore_base() == []  # remembered but unchanged: nothing to report
    with torch.no_grad():
        pipe.model.bias.add_(1.0)
    pipe.adapter = {"started_from": "pinned base"}
    assert pipe.restore_base() == ["bias"] and pipe.adapter is None  # only the changed tensor is reported
    assert torch.equal(pipe.model.bias, pipe._base_state["bias"])
    assert pipe.restore_base() == []  # nothing differs from the base any more


def test_mae_m2_byod_rerun_restores_the_base_and_the_experiment_has_its_own_pipeline(notebook):
    section_4 = _cell(notebook, "USE_BYOD = False")
    assert section_4.index("restored_tensors = pipe.restore_base()") < section_4.index("if USE_BYOD:")
    experiment = _cell(notebook, "RUN_EXPERIMENT = False")
    assert "experiment_pipe = ViTMAEPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=pipe.device)" in experiment
    assert "experiment_control = experiment_pipe.evaluate_reconstruction(test_records, mask_ratio=EXPERIMENT_MASK_RATIO, seed=0)" in experiment
    assert experiment.index("experiment_control =") < experiment.index("experiment_result = experiment_pipe.adapt(")
    assert f"Path('outputs/{STEM}_experiment')" in experiment
    assert "raise RuntimeError(f'the experiment changed a default export: {unchanged}')" in experiment
    assert not re.search(r"(?<!experiment_)pipe\.adapt\(", experiment)
    assert "**Predict → Change one thing → Run → Observe → Explain**" in _markdown(notebook)
    assert "they do not affect the default path" not in _markdown(notebook)


# --- MAE-M3: guided layer and infrastructure labelling ----------------------------------------------------------


def test_mae_m3_guided_layer_is_present(notebook):
    markdown = _markdown(notebook)
    for heading in ("**Who this notebook is for.**", "**Input → Model → Output.**", "**How to use this notebook.**", "**Roadmap:**", "## Troubleshooting", "## Glossary", "## Conclusion (your notes)", "## 10. Change one thing", "**Learner:**"):
        assert heading in markdown, heading
    assert markdown.count("**Predict") >= 7
    assert markdown.count("<details><summary>Check your reasoning</summary>") >= 7
    assert markdown.count("**What to notice:**") >= 6


def test_mae_m3_infrastructure_cells_are_labelled_and_collapsed(notebook):
    infra = [c for c in _code_cells(notebook) if c["metadata"].get("cellView") == "form"]
    assert len([c for c in infra if c["metadata"].get("dimer", {}).get("embedded_module")]) == 6
    titled = [c["source"].splitlines()[0] for c in infra if not c["metadata"].get("dimer")]
    assert len(titled) == 3 and all(t.startswith("# @title Infrastructure:") for t in titled), titled


def test_mae_m3_no_template_placeholders_leak(notebook):
    learner = "\n".join(c["source"] for c in notebook["cells"] if not c.get("metadata", {}).get("dimer", {}).get("embedded_module"))
    for leftover in ("{{", "{MODEL_ID}", "{stem}", "@P:"):
        assert leftover not in learner, leftover
    assert "}}" not in _markdown(notebook)


# --- MAE-m1: substitution tokens and the species count -----------------------------------------------------------


def test_mae_m1_no_substitution_token_and_the_validator_now_fails_on_one(notebook):
    assert "@P:" not in json.dumps(notebook) and "@P:" not in (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    markdown = _markdown(notebook)
    assert "three sparrows, a junco and two finches" in markdown and "four small sparrows" not in markdown
    spec = importlib.util.spec_from_file_location("_review_validator", ROOT / "tools" / "validate_release_assets.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    assert validator.SUBSTITUTION_TOKEN.search("macro F1 @P:FROZEN_PROBE_F1@") and not validator.SUBSTITUTION_TOKEN.search(markdown)


# --- MAE-m2: the held-out reconstruction is read on mean and median ------------------------------------------


def test_mae_m2_prose_reads_mean_and_median_and_scopes_the_guarantee(notebook):
    markdown = _markdown(notebook)
    assert "**mean and its median together**" in markdown and "0.1801 to 0.1827" in markdown
    assert "never returns one worse *on the validation split*" in markdown
    assert "validation selection that never returns a worse epoch than the frozen model, an artifact" not in markdown


# --- MAE-m3: BYOD contract and the probe verdict ------------------------------------------------------------


def _jpeg(i: int, corrupt: bool = False) -> bytes:
    from PIL import Image

    if corrupt:
        return b"not an image"
    image = Image.fromarray(np.random.default_rng(i).integers(0, 255, (40, 48, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG")
    return buffer.getvalue()


def _zip(path: Path, per_label: int, labels=("cat", "dog"), *, drop: str | None = None, corrupt: str | None = None, extra: dict[str, bytes] | None = None, bom: bool = False) -> Path:
    names = [(f"{label}{i}.jpg", label) for label in labels for i in range(per_label)]
    rows = "id,file,label\n" + "".join(f"r{k},{name},{label}\n" for k, (name, label) in enumerate(names))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("labels.csv", ("﻿" if bom else "") + rows)
        for k, (name, _label) in enumerate(names):
            if name != drop:
                archive.writestr(name, _jpeg(k, corrupt=name == corrupt))
        for name, data in (extra or {}).items():
            archive.writestr(name, data)
    return path


def test_mae_m3_stated_minimum_is_what_the_split_accepts(tmp_path, notebook):
    assert sm.min_byod_records(2)["total"] == 12 and sm.min_byod_records(3)["total"] == 15
    split = sm.split_dataset(sm.load_byod_dataset(_zip(tmp_path / "ok.zip", 6)), seed=42)
    assert {k: len(v) for k, v in split.items()} == {"test": 2, "validation": 2, "train": 8}
    with pytest.raises(ValueError, match=r"supply at least 12 distinct photographs, 6 per label"):
        sm.split_dataset(sm.load_byod_dataset(_zip(tmp_path / "small.zip", 5)), seed=42)
    assert "**12 photographs for two labels**" in _markdown(notebook)


def test_mae_m3_missing_and_corrupt_images_name_their_row_and_litter_is_skipped(tmp_path):
    with pytest.raises(ValueError, match=r"labels.csv line 5 \(file 'cat3.jpg'\): that image file is not in the dataset"):
        sm.load_byod_dataset(_zip(tmp_path / "missing.zip", 6, drop="cat3.jpg"))
    with pytest.raises(ValueError, match=r"labels.csv line 3 \(file 'cat1.jpg'\): Pillow cannot decode the image"):
        sm.load_byod_dataset(_zip(tmp_path / "corrupt.zip", 6, corrupt="cat1.jpg"))
    assert len(sm.load_byod_dataset(_zip(tmp_path / "mac.zip", 6, extra={"__MACOSX/._cat0.jpg": b"\0"}, bom=True))) == 12


def _section_4(notebook: dict, path: str) -> str:
    source = _cell(notebook, "USE_BYOD = False")
    source = source.replace("USE_BYOD = False  # @param", "USE_BYOD = True  # @param", 1)
    return source.replace("BYOD_PATH = ''  # @param", f"BYOD_PATH = {path!r}  # @param", 1)


def _section_4_namespace(restored: list) -> dict:
    ns = {k: getattr(sm, k) for k in dir(sm) if not k.startswith("__")}
    pipe = types.SimpleNamespace(adapter={"best_epoch": 3}, restore_base=lambda: restored.append(True) or ["a"])
    ns.update({"os": __import__("os"), "Path": Path, "pipe": pipe, "__name__": "__main__"})
    return ns


def test_mae_m3_byod_path_runs_section_4_outside_colab_from_the_base(notebook, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _zip(tmp_path / "mine.zip", 7)
    restored: list = []
    ns = _section_4_namespace(restored)
    exec(_section_4(notebook, "mine.zip"), ns)
    out = capsys.readouterr().out
    assert restored == [True], "a BYOD re-run must put the model back to the pinned base first"
    assert ns["raw_rows"] == {"byod": 14, "labels": 2, "duplicate_images_dropped": 0, "effective_minimum": 12}
    assert "fewer than 5 per label" in out
    assert (tmp_path / "outputs" / f"{STEM}_train.csv").is_file()


def test_mae_m3_upload_outside_colab_cancelled_and_bad_path_are_explained(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google", None)
    with pytest.raises(RuntimeError, match="upload dialog exists only in Google Colab"):
        exec(_section_4(notebook, ""), _section_4_namespace([]))
    with pytest.raises(FileNotFoundError, match="BYOD_PATH 'nowhere.zip' does not exist"):
        exec(_section_4(notebook, "nowhere.zip"), _section_4_namespace([]))
    for uploaded, message in (({}, "received 0"), ({"a.zip": b"", "b.zip": b""}, "received 2")):
        google, colab, files = (types.ModuleType(n) for n in ("google", "google.colab", "google.colab.files"))
        files.upload = lambda uploaded=uploaded: uploaded
        colab.files, google.colab = files, colab
        for name, module in (("google", google), ("google.colab", colab), ("google.colab.files", files)):
            monkeypatch.setitem(sys.modules, name, module)
        with pytest.raises(ValueError, match=message):
            exec(_section_4(notebook, ""), _section_4_namespace([]))


def test_mae_m3_only_contract_checks_remain_hard(notebook):
    code = "\n".join(c["source"] for c in _code_cells(notebook) if not c["metadata"].get("dimer", {}).get("embedded_module"))
    asserts = re.findall(r"(?m)^\s*assert .*$", code)
    assert len(asserts) == 1 and asserts[0].startswith("assert parity['identical_reconstructions']")
    assert "if val_history[adapt_result['best_epoch']] > val_history[0]:" in code


RK = ("masked_mse", "masked_mse_median", "masked_mse_visible", "psnr_masked")


def _rec(mean, median, ids=("a", "b", "c"), per=None):
    fill = {k: 0.3 for k in RK}
    return {
        "masked_mse": mean, "masked_mse_median": median, "masked_mse_visible": 0.2, "psnr_masked": 19.0, "n": len(ids), "mask_ratio": 0.75, "verdict": "small-sample", "definitions": {},
        "baselines": {"blur_fill": {**fill, "baseline": "blur"}, "mean_patch_fill": {**fill, "masked_mse": 0.4, "baseline": "mean"}},
        "per_image": [{"id": i, "masked_mse": v} for i, v in zip(ids, per or [mean] * len(ids), strict=True)],
    }


def _probe(accuracy):
    per = {c: {"support": 2, "recall": accuracy, "f1": accuracy} for c in ("cat", "dog")}
    return {"accuracy": accuracy, "macro_f1": accuracy, "n": 4, "verdict": "small-sample", "knn": {"accuracy": 0.5, "macro_f1": 0.5, "baseline": "knn"}, "per_class": per, "predictions": []}


def test_mae_m3_misses_are_recorded_and_do_not_stop_the_notebook(notebook, tmp_path, monkeypatch):
    """Sections 6 and 8 with stand-ins: a fill beats the frozen model, the probe sits at the floor, and the median
    worsens while the mean improves. Both cells complete and record the verdicts (no model)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    recs = iter([_rec(0.35, 0.18, per=[0.3, 0.35, 0.4]), _rec(0.34, 0.19, per=[0.29, 0.36, 0.4]), _rec(0.2332, 0.2)])
    probes = iter([_probe(0.5), _probe(0.5)])

    class StandIn:
        def evaluate_reconstruction(self, records, seed=0):
            return next(recs)

        def fit_probe(self, records):
            return {"train_loss": [0.5], "standardisation": "stand-in"}

        def evaluate(self, records):
            return next(probes)

    floor = {"accuracy": 0.5, "macro_f1": 0.33, "n": 4, "baseline": "majority"}
    ns = {
        "pipe": StandIn(), "train_records": [], "val_records": [], "test_records": [], "time": __import__("time"), "json": json, "classes": ["cat", "dog"],
        "majority_baseline": lambda *a: floor, "colour_neighbour_baseline": lambda *a: {**floor, "baseline": "colour"},
        "MODEL_ID": "stand-in", "MODEL_REVISION": "0" * 40, "DEFAULT_MODEL_KEY": "stand-in", "data_source": "stand-in", "dataset_manifests": {"test": {"digest": "d"}},
        "disjoint": {}, "display_names": {}, "val_history": {0: 0.2335, 3: 0.2332}, "adapt_result": {"best_epoch": 3, "history": [], "trainable_names": []}, "adapt_seconds": 0.0,
    }
    exec(_cell(notebook, "frozen_rec = pipe.evaluate_reconstruction(test_records, seed=0)"), ns)
    assert ns["frozen_rec_verdict"] == "a fill matches or beats the frozen model" and ns["frozen_probe_verdict"] == "at or below floor"
    exec(_cell(notebook, "adapted_rec = pipe.evaluate_reconstruction(test_records, seed=0)"), ns)
    comparison = json.loads((tmp_path / "outputs" / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))["comparison"]
    assert comparison["verdicts"]["test_masked_mse_mean"] == "improved" and comparison["verdicts"]["test_masked_mse_median"] == "worse"
    assert comparison["verdicts"]["adapted_below_blur_fill"] is False and comparison["verdicts"]["probe_accuracy"] == "no change"
    assert comparison["paired_per_image"] == {"better": 1, "worse": 1, "same": 1, "of": 3}
