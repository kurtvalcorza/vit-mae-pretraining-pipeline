"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package (six
modules, carried verbatim in dependency order), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

Generator /2 keys in use: ``modules`` lists every module of ``src/vit_mae_pipeline/`` except
``__init__.py``; ``entry_module`` is ``config.py`` (it holds ``MODEL_ID``/``MODEL_REVISION``/
``MODEL_LICENSE`` and the model key under the package's own spelling ``DEFAULT_MODEL_KEY``, mapped by
``identity_names``); ``rewrites`` carries two rules — the fleet ``DEFAULT_WEIGHTS_DIR`` rule and the
``__file__`` use inside ``model.resolve_weights_path`` (a repository-checkout convenience that a
standalone notebook has no checkout for); ``model_load`` lets the pipeline pick CUDA when it is visible
(the continuation stage is where that matters; CPU is the documented fallback).

This template configures an E2E masked-image-modelling workflow: the pinned facebook/vit-mae-base
snapshot is digest-verified and loaded, 360 CC0 iNaturalist photographs of six bird species are fetched
with per-file digests, validated and split by photograph, three drawn shapes are masked and reconstructed
through the inference contract, the frozen model's masked-patch MSE over the held-out photographs is
measured beside two non-neural fills and its features are read through a linear probe beside three
non-neural classifiers, a bounded continuation of the masked-autoencoding objective runs in the kernel,
the held-out split is reconstructed and probed again, the adapted model re-reconstructs the shapes, and
the adapter (with the probe head) is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "vit_mae_pipeline",
    "repo_name": "vit-mae-pretraining-pipeline",
    "stem": "vit_mae_pretraining",
    "notebook_name": "vit_mae_pretraining_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "isolated_runtime": True,
    "infrastructure_labels": True,
    # The fleet's uv isolated-environment mechanism (bioclip2-biodiversity-pipeline): managed CPython, a size- and
    # SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "pipeline_class": "ViTMAEPipeline",
    "weights_key": "vit-mae-base",
    "modules": ["config.py", "model.py", "metrics.py", "samples.py", "pipeline.py", "provenance.py"],
    "entry_module": "config.py",
    "identity_names": {"MODEL_KEY": "DEFAULT_MODEL_KEY"},
    "rewrites": [
        ["^DEFAULT_WEIGHTS_DIR = Path\\(__file__\\)[^\\n]*$", 'DEFAULT_WEIGHTS_DIR = Path.cwd() / "weights" / DEFAULT_MODEL_KEY  # standalone rewrite (build_notebook.py): working-directory snapshot, no repository checkout'],
        ["^    repo_root = Path\\(__file__\\)\\.resolve\\(\\)\\.parents\\[2\\]$", "    repo_root = Path.cwd()  # standalone rewrite (build_notebook.py): no repository checkout to resolve"],
    ],
    "model_load": "ViTMAEPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)",
    "runtime_imports": ["torch", "transformers", "numpy"],
    "title": "ViT-MAE Base Pre-training Pipeline — DIMER E2E masked-image-modelling tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/vit-mae-pretraining-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/vit-mae-pretraining-pipeline/blob/main/tutorials/vit_mae_pretraining_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-facebook%2Fvit--mae--base-ffcc4d?style=flat",
            "https://huggingface.co/facebook/vit-mae-base",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-facebookresearch%2Fmae-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/facebookresearch/mae",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2111.06377-b31b1b.svg", "https://arxiv.org/abs/2111.06377"),
    ],
    "capability": "masked-patch reconstruction (masked image modelling), mean-pooled encoder embeddings, a linear probe on those embeddings and bounded continuation of the masked-autoencoding objective on a photograph set, using the pinned `facebook/vit-mae-base` weights",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (nothing is "
        "installed into the notebook's own Python, so no restart is needed and Run all completes in one pass), stages and digest-verifies the "
        "pinned `facebook/vit-mae-base` snapshot (a 448 MB `model.safetensors`; no pickle is opened anywhere), fetches "
        "the 360 pinned iNaturalist photographs from the project's open-data bucket (about 39 MB, each refused on any byte-size "
        "or SHA-256 mismatch), cuts them per species into 216 / 48 / 96 training, validation and test photographs, masks and "
        "reconstructs three drawn shapes through the inference contract with an input manifest and a rejection probe, measures "
        "the frozen model's masked-patch MSE over the 96 test photographs beside the mean-patch and blur fills and its "
        "linear-probe accuracy beside the majority floor, a colour nearest neighbour and a k-NN on the same features, runs a "
        "bounded continuation of the masked-autoencoding objective on the decoder and the last two encoder blocks with "
        "validation-MSE epoch selection, reconstructs and probes the held-out photographs again, re-reconstructs the drawn shapes "
        "with the adapted model, exports the adapter and the probe head as safetensors with a manifest, and reloads that artifact "
        "into a fresh pipeline to verify reconstruction and probe parity. The default path needs no repository clone, no DIMER "
        "worker or service, no credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5). On CPU the whole "
        "path took about four minutes of model time on the build workstation's CPU after the downloads (expect longer on a 2-vCPU hosted "
        "runtime); a CUDA runtime is used automatically when present and finishes in a few minutes."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and either set `BYOD_PATH` to a zip or folder "
        "in the runtime (Colab, Kaggle or Jupyter) or leave it empty to upload one zip in Colab, then choose **Run after** from "
        "that cell (it first puts the model back to the pinned base) to supply a `labels.csv` (columns `id`, `file`, `label`) "
        "beside the image files — at least **12 photographs for two labels** (6 per label), 15 for three, and 4 per label from "
        "four labels up (`min_byod_records(n_labels)`); the labels feed only the linear probe, the reconstruction objective never sees them. They pass "
        "through the same validation, seeded stratified split, baselines, continuation, held-out evaluation, artifact export "
        "and reload-parity cells as the iNaturalist sample. The expected schema and the ceilings are stated in the "
        "Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD is optional and never part of the "
        "default path."
    ),
    "intro": (
        "`facebook/vit-mae-base` is the ViT-B/16 masked autoencoder of He, Chen, Xie, Li, Dollár and Girshick (CVPR 2022): a "
        "12-layer, 768-wide encoder that sees only the visible 25 % of an image's 196 16×16 patches, and an 8-layer, 512-wide "
        "decoder that predicts the pixels of the hidden 75 % from the encoder's tokens plus a shared mask token — 111,907,840 "
        "parameters in total, pre-trained on ImageNet-1K for 1,600 epochs with per-patch pixel MSE on the hidden patches only, "
        "published under the **Apache-2.0** licence. It has **no user-facing task output**: the model returns a reconstruction "
        "and its own loss, and the encoder's features are what downstream fine-tuning consumes. This notebook therefore reads "
        "the model two ways — the **masked-patch MSE** of its reconstructions (the objective, in the processor's normalised "
        "pixel space, lower is better, not a quality judgement) and the **accuracy of a linear probe** on its mean-pooled "
        "patch tokens (the paper's own readout of representation quality) — and neither is calibrated or a human judgement.\n\n"
        "What this notebook adds to inference is **continuation of the pre-training objective on a photograph set**. The "
        "dataset is real: 360 CC0-licensed, research-grade iNaturalist photographs of six common North American birds "
        "(**CC0 1.0**; three sparrows, a junco and two finches, 60 per species, one per observer), pinned by photo id, "
        "byte size and SHA-256 and fetched from the project's open-data bucket at run time. The honest question is narrow: does "
        "a bounded continuation of masked autoencoding — the decoder and the last two encoder blocks, on 216 photographs, "
        "with the epoch chosen on the validation masked MSE — move the reconstruction error on an image-disjoint test split, "
        "against two **non-neural fills** (the **mean colour of the visible patches** and a **blur of the visible neighbours**), "
        "and does it move the probe against three **non-neural classifiers** (the **majority floor**, a **colour nearest "
        "neighbour** and a **k-NN on the same features**)? The build record's answer is that it barely does — the pre-trained "
        "model is already converged on natural photographs, and 216 more of them at a learning rate small enough not to hurt "
        "move the held-out mean masked MSE from 0.2281 to 0.2280 while the median rose from 0.1801 to 0.1827 — so what the "
        "notebook demonstrates is the adaptation *contract* (bounded training, validation selection that never returns an "
        "epoch worse than the frozen model *on the validation split* — the held-out error can still move either way — an artifact "
        "that reloads with verified parity) and how to read the numbers around it. Nothing here is a quality claim about your "
        "photographs: it is one seeded split of one sample.\n\n"
        "**Snapshot note:** the pinned revision ships `model.safetensors` (a 4-file manifest: model card, config, processor "
        "config and weights) — no pickle is opened anywhere in this notebook. Section 3 stages and digest-verifies those files before "
        "the processor or the model is constructed."
    ),
    "guided": {
        "opening": [
            (
                "**Who this notebook is for.** A learner who knows basic Python, has used Colab or Jupyter, and wants to see what a masked autoencoder does, how to read its reconstruction loss and its features honestly, and what continuing its pre-training on a small photograph set does — and does not — change. No prior experience with masked autoencoders or fine-tuning is assumed; each term is explained where it first matters and again in the **Glossary** at the end. CPU is adequate (about four minutes of model time); a GPU is faster.\n\n**Input → Model → Output.**\n\n| | Reconstruction | Representation | Bounded continuation |\n|---|---|---|---|\n| Input | one image (16..4,096 px), resized to 224 × 224, with 75 % of its 196 patches hidden by a seeded random mask | photographs with nothing hidden | unlabelled photographs (216 training and 48 validation in the sample) |\n| Model | ViT-MAE base: a 12-layer encoder over the visible patches and an 8-layer decoder that predicts the hidden ones | the encoder's mean-pooled patch tokens, read by a linear probe | the decoder and the last two encoder blocks trained on the model's own masked-patch loss; validation MSE chooses the epoch |\n| Output | a reconstruction, its masked and visible MSE and PSNR — the model's own objective, not a perceptual judgement | probe accuracy and macro F1 beside three non-neural classifiers | a safetensors adapter, and held-out reconstruction and probe numbers beside the frozen model |\n\n**How to use this notebook.** Choose a runtime (CPU works; a GPU is faster), then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated environment, the carried package (including the 360-photograph table) and the model snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run. Before each principal result the notebook asks you to **Predict**; after it come **What to notice** and a collapsible **Check your reasoning** with a worked answer that names the run it quotes — the Kaggle T4 release run of 21 September 2026. Section 10 is a **change-one-thing experiment**, off by default. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 the photographs, validation and a stratified split *(evaluation practice)* → 5 masking and reconstruction through the inference contract *(core concept: what an MAE predicts)* → 6 two fills, three classifiers and the frozen model *(evaluation practice)* → 7 bounded continuation and validation selection *(core concept)* → 8 held-out evaluation, mean and median *(evaluation practice)* → 9 re-reconstruct, export and reload *(engineering)* → 10 change one thing (optional) → conclude."
            )
        ]
    },
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the immutable "
        "upstream snapshot; fetch a digest-pinned labelled photograph set, validate it and split it per species without "
        "leakage; mask and reconstruct drawn shapes through the public API and read a reconstruction loss correctly (the "
        "model's own objective, seeded masks, hidden-patch and visible-patch error, uncalibrated); measure the frozen model's "
        "masked-patch MSE beside two non-neural fills and its linear-probe accuracy beside three non-neural classifiers; run a "
        "bounded continuation of the masked-autoencoding objective with explicit hyperparameters and validation-based epoch "
        "selection; evaluate on an image-disjoint test split both ways; re-reconstruct drawings from a different image family "
        "with the adapted model; and export a safetensors adapter with its probe head that reloads against the pinned base "
        "with verified parity."
    ),
    "exclusions": (
        "image classification as a product (the probe is a readout of the features, not a classifier to ship), object "
        "detection, segmentation, generation of new images, inpainting of user-chosen regions (the mask is random, as in "
        "pre-training), full fine-tuning of the encoder for a downstream task, pre-training from scratch, normalised-pixel "
        "targets (`norm_pix_loss` is off in this checkpoint), training on photographs that are not the pinned sample or your "
        "own uploads, evaluation on ImageNet or any benchmark proper (only one seeded 360-photograph sample is scored here), "
        "and any claim that six bird species stand in for your images. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Learner:** basic Python and Colab or Jupyter familiarity; no prior experience with masked autoencoders or fine-tuning. The notebook explains masking, masked and visible MSE, PSNR, the linear probe, k-NN, validation selection and the adapter where they are first used; the Glossary repeats them.",
        "- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or Linux Jupyter). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the kernel's own Python version does not matter and nothing is installed into it. The default path runs on CPU (float32) and uses CUDA automatically when available. CPU is slow but adequate: the build record measured about 4.4 s to reconstruct the 96 test photographs, 28.8 s to embed the 312 photographs and fit the probe, and 159.1 s for the 5 epochs of continuation (216 photographs per epoch through the encoder and decoder, the decoder and the last two encoder blocks training) including the per-epoch validation reconstruction, so the whole default path is about four minutes of model time on the build workstation's CPU with the snapshot and photographs already cached (a 2-vCPU hosted runtime will be several times slower); a hosted T4 finishes it in a few minutes. The pinned `torch==2.14.0` install and the 448 MB checkpoint are the large downloads of the run; the photographs add about 39 MB.",
        "- **Knowledge:** basic Python and PIL; what a mean squared error and a PSNR are; what a masked autoencoder hides and predicts; what accuracy and macro F1 measure and why a linear probe is a readout of features rather than a product classifier.",
        "- **Data contract:** records are `{id, image, label}` — a PIL image or a file decodable by Pillow with sides between `MIN_IMAGE_SIDE` (16) and `MAX_IMAGE_SIDE` (4096) px, and a label of at most 64 plain characters (used by the probe only). Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; a training split needs 8..20,000 records over 2..100 labels, and every label needs one training, one validation and one test photograph, so with the default 15 % + 20 % stratified hold-out the effective BYOD minimum is **12 photographs for two labels** (6 per label), 15 for three, and 4 per label from four labels up (`min_byod_records(n_labels)`); splitting is stratified per label after pixel-digest de-duplication so no photograph lands in two splits. Every image is resized to 224×224 by the processor. BYOD accepts one zip of images plus a `labels.csv` in that shape.",
        "- **Validation is structural, not semantic:** every image is opened and decoded and every label checked, but nothing checks that a label is right — a mislabelled set is probed without complaint, and the reconstruction objective never reads the labels at all.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path fetches 360 JPEG/PNG files from `https://inaturalist-open-data.s3.amazonaws.com/photos/<id>/medium.<ext>` (about 39 MB in total), each pinned by byte size and SHA-256 in the carried `samples.py` and refused on any mismatch; every photograph's iNaturalist observation page and observer login are kept in its record. Each photograph carries the CC0 1.0 licence its observer chose (nothing is committed to the repository).",
    ],
    "cells": [
        {
            "md": (
                "## 4. iNaturalist photographs and split\n\n"
                "`fetch_corpus` returns the 360 pinned photographs from the cache under `weights/inat-birds/` or the "
                "iNaturalist open-data bucket — every cached file is re-hashed and every fetched file refused on any byte-size or "
                "SHA-256 mismatch — and `read_corpus` decodes them into `{{id, image, label}}` records with their observation "
                "page, observer and species names. `build_sample_dataset` draws a seeded stratified split per species (36 / 8 / "
                "16 → 216 / 48 / 96). `validate_dataset` then checks every record against the contract, `check_split_disjoint` "
                "asserts no photograph (by decoded-pixel digest) is shared, `observer_overlap` reports how many observers "
                "contributed to more than one split (an observation about the draw, not an assertion), and the training split's "
                "labels table is written to `outputs/{stem}_train.csv` in the shape BYOD expects. The labels are for the probe "
                "in Sections 6 and 8; the continuation in Section 7 never reads them.\n\n"
                "Look for: 360 photographs, the six species with 36 / 8 / 16 each, three digests, and four refusal probes — a "
                "duplicate id, an image over the side ceiling, a dataset with one label and one too small to split — each "
                "rejected before the model does anything.\n\n"
                "*Evaluation practice.* **Bring your own data (optional):** set `USE_BYOD = True` and either `BYOD_PATH` (a zip or "
                "a folder holding `labels.csv` and the images, as a path in this runtime — this works on Colab, Kaggle and "
                "Jupyter) or leave `BYOD_PATH` empty to upload exactly one zip through the Colab dialog; then choose **Run after** "
                "from this cell. This cell first puts the model back to the pinned base, so Section 6 reads the frozen model and "
                "Section 7's epoch 0 is the frozen model. The effective minimum is 12 photographs for two labels.\n\n"
                "**Predict before running:** the reconstruction objective never reads the labels. Why split by species at all?"
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "# A re-run after Section 7 (BYOD, or a new split): Section 6 and Section 7's epoch 0 must read the pinned base.\n"
                "had_adapter = pipe.adapter is not None\n"
                "restored_tensors = pipe.restore_base()\n"
                "if had_adapter or restored_tensors:\n"
                "    print({{'restored_pinned_base': len(restored_tensors), 'note': 'the continued decoder and blocks were put back to the checkpoint and the probe was dropped; Sections 5-7 start from it again'}})\n"
                "if USE_BYOD:\n"
                "    if BYOD_PATH.strip():\n"
                "        byod_zip = Path(BYOD_PATH.strip()).expanduser()\n"
                "        if not byod_zip.exists():\n"
                "            raise FileNotFoundError(f'BYOD_PATH {{BYOD_PATH!r}} does not exist (relative paths start at {{Path.cwd()}}): give a .zip or a folder holding labels.csv and the images.')\n"
                "        file_name = byod_zip.name\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise RuntimeError('USE_BYOD is True but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: on Kaggle or Jupyter put the zip (or folder) in the runtime and set BYOD_PATH to its path.') from None\n"
                "        uploaded = files.upload() or {{}}\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'Upload exactly one .zip file (received {{len(uploaded)}}; a cancelled dialog sends none): run this cell again.')\n"
                "        file_name, payload = next(iter(uploaded.items()))\n"
                "        if not file_name.lower().endswith('.zip'):\n"
                "            raise ValueError(f'{{file_name}}: upload one .zip holding labels.csv and the images.')\n"
                "        byod_zip = Path('work') / 'byod.zip'\n"
                "        byod_zip.parent.mkdir(parents=True, exist_ok=True)\n"
                "        byod_zip.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_zip)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    display_names = {{}}\n"
                "    n_labels = len({{r['label'] for r in records}})\n"
                "    raw_rows = {{'byod': len(records), 'labels': n_labels, 'duplicate_images_dropped': len(records) - sum(len(part) for part in splits.values()), 'effective_minimum': min_byod_records(max(n_labels, 2))['total']}}\n"
                "    if len(splits['test']) < 5 * n_labels:\n"
                "        print({{'caution': f\"only {{len(splits['test'])}} held-out test photographs for {{n_labels}} labels (fewer than 5 per label): probe accuracy moves in large steps; add photographs before reading it\"}})\n"
                "else:\n"
                "    corpus_files = fetch_corpus(cache_dir='weights/inat-birds')\n"
                "    corpus = read_corpus(corpus_files)\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}}: {{CORPUS_RELEASE}} ({{CORPUS_LICENSE}})'\n"
                "    display_names = {{key: common for key, (_scientific, common) in SPECIES.items()}}\n"
                "    raw_rows = {{'photographs': len(corpus), 'bytes': sum(len(v) for v in corpus_files.values()), 'observers': len({{r['observer'] for r in corpus}})}}\n"
                "# The training split must hold MIN_RECORDS; validation and test only need one photograph each (split_dataset checks the labels).\n"
                "dataset_manifests = {{name: validate_dataset(part, min_records=MIN_RECORDS if name == 'train' else 1) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "classes = class_names(train_records)\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint, 'observer_overlap': observer_overlap(splits), 'classes': classes}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'label_counts': manifest['label_counts'], 'image_side': manifest['image_side'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = train_records[0]\n"
                "print({{'example': {{'id': example['id'], 'label': example['label'], 'display_name': display_names.get(example['label'], example['label']), 'size': list(example['image'].size), 'observation': example.get('inat_observation_url')}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'image over the side ceiling': [{{**train_records[0], 'image': Image.new('RGB', (MAX_IMAGE_SIDE + 1, 8))}}, *train_records[1:8]],\n"
                "    'one label only': [{{**r, 'label': 'bird'}} for r in train_records[:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "**What to notice:** 360 photographs, 216 / 48 / 96, the three digests, `observer_overlap`, and the four refusals.\n\n<details><summary>Check your reasoning</summary>Because the probe in Sections 6 and 8 does read them: stratifying keeps every species in every split, so the probe is scored on all six. The reconstruction numbers would be valid under any image-disjoint split. A review probe counted 53 of 222 observers in more than one split, so a probe can partly learn a photographer's style.</details>"
            ),
        },
        {
            "md": (
                "## 5. Mask and reconstruct through the inference contract\n\n"
                "*Core concept.* A masked autoencoder hides most of an image and predicts the hidden pixels from the visible "
                "ones; its loss is computed on the hidden patches only. **Predict before running:** on a flat-coloured drawn "
                "square, will the model's masked MSE be lower or higher than on a photograph?\n\n"
                "The inference contract is exercised on three 32×32 synthetic shapes — a red square, a green circle and a blue "
                "triangle — rendered in code as ASCII PPM files exactly as the repository's `examples/sample-data/generate_samples.py` "
                "renders them and digest-asserted against `SHA256SUMS`, a different image family from the photographs, and images "
                "the model will be asked to reconstruct again after adaptation. `validate_inputs` applies exactly the checks the "
                "public operations apply and returns an input manifest; a remote URL is validated too and its rejection recorded as "
                "a finding. `reconstruct` hides a seeded random 75 % of the 196 patches (147 of them), runs the encoder on the "
                "visible 49 and the decoder on all 196, and returns per image the **masked MSE** (over the hidden patches, in the "
                "processor's normalised pixel space), the **visible MSE** (the decoder also predicts the patches the encoder saw), "
                "the mask itself and the reconstruction composited over the visible patches. `embed` returns the mean of the 196 "
                "patch tokens with nothing hidden, L2-normalised. The per-call `evaluation_report` on three drawn shapes is "
                "`sample-sanity` — plumbing evidence, not a measurement; whether the model reconstructs *well* is what Section 6 "
                "measures on 96 photographs.\n\n"
                "Look for: the same seed hiding the same 147 patches twice, a different seed hiding different ones, and the "
                "pipeline's mean masked MSE equal to the model's own `loss` — the pipeline computes what the checkpoint was trained on. "
                "On these flat drawings the hidden patches are trivial to fill (masked MSE about 0.0005–0.02 in the build record) "
                "while the visible ones are not (about 0.55–0.64): the decoder was never trained on the patches the encoder saw, "
                "which is why the visible-patch error is reported but never read as quality."
            ),
            "code": (
                "SAMPLE_DIGESTS = {{  # examples/sample-data/SHA256SUMS\n"
                "    'red_square.ppm': 'b38ff0c9131677ed6cf09832eff40a22841e1a4725d426fad3b1bf6a1dbdb096',\n"
                "    'green_circle.ppm': '2e3e657686a0f6a6df3f3621d21a410d20d9a47b109f06f9405faa4f747a663f',\n"
                "    'blue_triangle.ppm': 'f0f4c38c7af92b3a6b1d55a25272156edd87b41e029e116cfa059003dae029a3',\n"
                "}}\n"
                "WIDTH, HEIGHT, BACKGROUND = 32, 32, (245, 245, 245)\n\n\n"
                "def shape_mask(shape, x, y):\n"
                "    if shape == 'square':\n"
                "        return 8 <= x < 24 and 8 <= y < 24\n"
                "    if shape == 'circle':\n"
                "        return (x - 16) ** 2 + (y - 16) ** 2 <= 9**2\n"
                "    if not 7 <= y < 26:\n"
                "        return False\n"
                "    half = (y - 7) // 2\n"
                "    return 16 - half <= x <= 16 + half\n\n\n"
                "def render_ppm(foreground, shape):\n"
                "    # The repository's generate_samples.py rendering: ASCII P3, 24 values per line.\n"
                "    lines = ['P3', f'{{WIDTH}} {{HEIGHT}}', '255']\n"
                "    for y in range(HEIGHT):\n"
                "        row = []\n"
                "        for x in range(WIDTH):\n"
                "            pixel = foreground if shape_mask(shape, x, y) else BACKGROUND\n"
                "            row.extend(str(v) for v in pixel)\n"
                "        for start in range(0, len(row), 24):\n"
                "            lines.append(' '.join(row[start : start + 24]))\n"
                "    return '\\n'.join(lines) + '\\n'\n\n\n"
                "Path('outputs/sample-data').mkdir(parents=True, exist_ok=True)\n"
                "specs = {{'red_square.ppm': ((220, 40, 40), 'square'), 'green_circle.ppm': ((40, 170, 75), 'circle'), 'blue_triangle.ppm': ((40, 90, 220), 'triangle')}}\n"
                "shape_images = []\n"
                "for name, (foreground, shape) in specs.items():\n"
                "    path = Path('outputs/sample-data') / name\n"
                "    path.write_bytes(render_ppm(foreground, shape).encode('ascii'))\n"
                "    digest = hashlib.sha256(path.read_bytes()).hexdigest()\n"
                "    if digest != SAMPLE_DIGESTS[name]:\n"
                "        raise ValueError(f'Synthetic sample digest mismatch for {{name}}: {{digest}} != {{SAMPLE_DIGESTS[name]}}')\n"
                "    shape_images.append(path)\n"
                "shape_sha256 = {{p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in shape_images}}\n"
                "print({{'ceilings': {{'DEFAULT_MASK_RATIO': DEFAULT_MASK_RATIO, 'NUM_PATCHES': NUM_PATCHES, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_BATCH': MAX_BATCH, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS, 'MIN_CLASSES': MIN_CLASSES, 'image_contract': '224x224 RGB after processor resize, 196 patches of 16x16', 'device': str(pipe.device)}}}})\n"
                "input_manifest = validate_inputs(shape_images, names=[p.name for p in shape_images])\n"
                "try:\n"
                "    validate_inputs('https://example.invalid/not-allowed.png')\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'remote-url-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'shapes': [p.name for p in shape_images], 'sha256': {{k: v[:16] + '...' for k, v in shape_sha256.items()}}, 'manifest_verdict': input_manifest['verdict'], 'hidden_patches': input_manifest['hidden_patches'], 'findings': len(input_manifest['findings'])}})\n"
                "SCENE_SEED = 7\n"
                "t0 = time.perf_counter()\n"
                "scene = pipe.reconstruct(shape_images, seed=SCENE_SEED)\n"
                "again = pipe.reconstruct(shape_images, seed=SCENE_SEED, return_images=False)\n"
                "other = pipe.reconstruct(shape_images, seed=SCENE_SEED + 1, return_images=False)\n"
                "shape_embeddings = pipe.embed(shape_images)['embeddings']\n"
                "scene_rows = []\n"
                "for path, entry in zip(shape_images, scene['results'], strict=True):\n"
                "    entry['reconstruction'].save(f'outputs/{stem}_frozen_{{path.stem}}.png')\n"
                "    scene_rows.append({{'image': path.name, 'hidden_patches': entry['hidden_patches'], 'masked_mse': entry['masked_mse'], 'visible_mse': entry['visible_mse']}})\n"
                "    print(path.name, '->', {{'hidden_patches': entry['hidden_patches'], 'masked_mse': round(entry['masked_mse'], 4), 'visible_mse': round(entry['visible_mse'], 4)}})\n"
                "checks = {{\n"
                "    'one_result_per_image': len(scene['results']) == len(shape_images),\n"
                "    'hidden_patches_match_ratio': all(e['hidden_patches'] == int(NUM_PATCHES * DEFAULT_MASK_RATIO) for e in scene['results']),\n"
                "    'errors_non_negative': all(e['masked_mse'] >= 0.0 and e['visible_mse'] >= 0.0 for e in scene['results']),\n"
                "    'same_seed_same_mask': all(a['mask'] == b['mask'] for a, b in zip(scene['results'], again['results'], strict=True)),\n"
                "    'other_seed_other_mask': any(a['mask'] != b['mask'] for a, b in zip(scene['results'], other['results'], strict=True)),\n"
                "    'pipeline_mse_is_model_loss': abs(scene['model_loss'] - float(np.mean([e['masked_mse'] for e in scene['results']]))) < 1e-4,\n"
                "    'embeddings_unit_norm': bool(np.allclose(np.linalg.norm(shape_embeddings, axis=1), 1.0, atol=1e-4)),\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'inference output failed a sanity check: {{checks}}')\n"
                "frozen_scene = evaluation_report(scene, sample_kind='synthetic')\n"
                "print({{'checks': checks, 'seconds': round(time.perf_counter() - t0, 2), 'frozen_scene': {{m['id']: round(m['value'], 4) for m in frozen_scene['metrics']}}, 'verdict': frozen_scene['verdict'], 'embedding_shape': list(shape_embeddings.shape)}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the masked and visible MSE per shape, and `hidden_patches` (147 of 196 for each).\n\n<details><summary>Check your reasoning</summary>Much lower. In the CPU build record the three drawn shapes scored masked MSE 0.0005, 0.0113 and 0.0228 against 0.2281 for the test photographs: a hidden patch of a flat-coloured drawing on a white ground is predictable from its visible neighbours, while a photograph's hidden feathers and background are not. A low error on drawings says the task is easy, not that the model is good.</details>"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model on the test photographs, two ways\n\n"
                "**Reconstruction.** `pipe.evaluate_reconstruction` hides one seeded 75 % mask per photograph (the seed is derived "
                "from the record id, so the same photograph gets the same mask in every evaluation of this notebook) and reports "
                "the mean and median **masked MSE**, the **visible MSE** and the **PSNR of the hidden patches** over the 96 test "
                "photographs. Two non-neural fills are scored on exactly the same masks: the **mean-patch fill** (every hidden "
                "patch is the mean colour of the visible patches — what a model that learned nothing about images would do) and "
                "the **blur fill** (every hidden patch is the mean colour of its visible neighbours — the cheapest use of locality). "
                "**What to look for:** the frozen model against both fills, on the mean *and* the median.\n\n"
                "**Linear probe.** `pipe.fit_probe` embeds the 216 training photographs with nothing hidden, standardises the "
                "features with the training set's mean and standard deviation (the paper's affine-free BatchNorm) and fits a "
                "multinomial logistic-regression head by full-batch Adam; `pipe.evaluate` reports its **accuracy** and **macro F1** "
                "on the test photographs with a per-species breakdown, and beside it a **5-nearest-neighbour** vote on the same "
                "features. The **majority floor** answers every photograph with the most frequent training label (accuracy 1/6 on "
                "a balanced split), and the **colour nearest neighbour** answers with the label of the training photograph whose "
                "3×3 mean-colour grid is closest. MAE features are known to probe poorly before fine-tuning (the paper reports "
                "67.8 % linear-probe top-1 on ImageNet against 83.6 % after fine-tuning), and six bird species from 216 "
                "photographs is a hard probe. Read the per-species recall: it is where the probe's number comes from.\n\n"
                "*Evaluation practice.* The cell records **verdicts** — whether the frozen model beats both fills, and whether the "
                "probe beats the majority floor — instead of asserting them: on a small BYOD test split a miss is a finding, and "
                "the notebook continues to the continuation, export and reload.\n\n"
                "**Predict before running:** MAE features are known to probe poorly before fine-tuning. Will the linear probe beat "
                "a k-NN on the same features, and by how much will it beat the 1/6 floor?"
            ),
            "code": (
                "RK = ('masked_mse', 'masked_mse_median', 'masked_mse_visible', 'psnr_masked')\n"
                "CK = ('accuracy', 'macro_f1')\n"
                "t0 = time.perf_counter()\n"
                "frozen_rec = pipe.evaluate_reconstruction(test_records, seed=0)\n"
                "frozen_rec_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'frozen_reconstruction_test': {{k: round(frozen_rec[k], 4) for k in RK}}, 'n': frozen_rec['n'], 'mask_ratio': frozen_rec['mask_ratio'], 'verdict': frozen_rec['verdict'], 'seconds': frozen_rec_seconds}})\n"
                "for name, fill in frozen_rec['baselines'].items():\n"
                "    print({{name: {{k: round(fill[k], 4) for k in RK}}, 'note': fill['baseline']}})\n"
                "print({{'definitions': frozen_rec['definitions']}})\n"
                "# Reported verdicts, not assertions: on your images a fill may come close, and that is a finding.\n"
                "frozen_rec_verdict = 'frozen below both fills' if frozen_rec['masked_mse'] < min(frozen_rec['baselines']['blur_fill']['masked_mse'], frozen_rec['baselines']['mean_patch_fill']['masked_mse']) else 'a fill matches or beats the frozen model'\n"
                "print({{'frozen_vs_fills': frozen_rec_verdict}})\n\n"
                "baseline_majority = majority_baseline(train_records, test_records, classes)\n"
                "baseline_neighbour = colour_neighbour_baseline(train_records, test_records, classes)\n"
                "print({{'majority_baseline': {{k: round(baseline_majority[k], 3) for k in CK}}, 'n': baseline_majority['n'], 'note': baseline_majority['baseline']}})\n"
                "print({{'colour_neighbour_baseline': {{k: round(baseline_neighbour[k], 3) for k in CK}}, 'note': baseline_neighbour['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_probe_fit = pipe.fit_probe(train_records)\n"
                "frozen_probe = pipe.evaluate(test_records)\n"
                "probe_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'frozen_probe_test': {{k: round(frozen_probe[k], 3) for k in CK}}, 'n': frozen_probe['n'], 'verdict': frozen_probe['verdict'], 'train_loss': [round(v, 4) for v in frozen_probe_fit['train_loss']], 'standardisation': frozen_probe_fit['standardisation'], 'seconds': probe_seconds}})\n"
                "print({{'knn_on_the_same_features': {{k: round(frozen_probe['knn'][k], 3) for k in CK}}, 'note': frozen_probe['knn']['baseline']}})\n"
                "frozen_fields = {{c: {{'n': v['support'], 'recall': round(v['recall'], 2), 'f1': round(v['f1'], 2)}} for c, v in frozen_probe['per_class'].items()}}\n"
                "print({{'by_species_frozen': frozen_fields}})\n"
                "frozen_probe_verdict = 'above floor' if frozen_probe['accuracy'] > baseline_majority['accuracy'] else 'at or below floor'\n"
                "print({{'frozen_probe_vs_floor': frozen_probe_verdict}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the masked MSE mean and median against both fills, the probe against the three classifiers, and the per-species recall.\n\n<details><summary>Check your reasoning</summary>In the Kaggle T4 release run (21 September 2026) the frozen model's masked MSE was 0.2281 (median 0.1801) against 0.5452 for the blur fill and 0.7652 for the mean fill, PSNR 19.3 dB on the hidden patches. The probe scored accuracy 0.365 and macro F1 0.365 against the majority floor's 0.167, the colour neighbour's 0.260 and the k-NN's 0.219 — above every baseline, but with a 95 % interval of roughly ±0.10 on 96 photographs, and far from a classifier you would ship.</details>"
            ),
        },
        {
            "md": (
                "## 7. Bounded continuation of the masked-autoencoding objective\n\n"
                "`pipe.adapt` continues pre-training: it trains the whole decoder, the last `TRAINABLE_BLOCKS` encoder blocks "
                "and the encoder's final LayerNorm — two blocks by default, 40,286,464 of 111,907,840 parameters; the patch "
                "embedding, the position embeddings and the first ten blocks stay frozen — on the checkpoint's own loss, the "
                "masked-patch MSE under a fresh seeded random 75 % mask every step. AdamW at a fixed learning rate with weight "
                "decay 0.05, gradient clipping at 1.0, seeded shuffling and masks, no scheduler, and **no labels**. Epoch 0 "
                "records the frozen model's validation reconstruction; every epoch is scored on the 48 validation photographs "
                "with their fixed seeded masks, and the epoch with the lowest validation masked MSE is kept — so the selection "
                "can return the frozen model itself (epoch 0) when nothing beats it, and never returns one worse *on the "
                "validation split*; the test split can still move either way. Every call starts from the **pinned base**: "
                "tensors an earlier call (or an artifact) changed are put back first, so epoch 0 is always the frozen model and "
                "re-running Sections 7–8 with a changed field repeats the comparison validly. The fitted "
                "probe is discarded, because the features it was fitted on no longer exist.\n\n"
                "Watch the numbers move very little: the pre-trained model has seen 1,600 epochs of ImageNet, and 216 "
                "photographs of birds are not a new domain to it. The build record's learning-rate sweep found `1e-4` makes "
                "the validation masked MSE **worse** from the first epoch (batches of eight with random masks are a noisy "
                "gradient), and `1e-5` moves it from 0.2335 to 0.2332 at epoch 3 of 5. That is the honest "
                "default: the configuration that did not hurt, chosen on the validation split. To compare a change side by side "
                "without replacing the default exports, use Section 10.\n\n"
                "**Predict before running:** will any epoch beat the frozen model's validation masked MSE, and by how much?"
            ),
            "code": (
                "EPOCHS = 5  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 8  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_BLOCKS = 2  # @param {{type:\"integer\"}}\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_' + k: round(entry['val'][k], 4) for k in ('masked_mse', 'masked_mse_visible', 'psnr_masked')}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "settings = {{'epochs': EPOCHS, 'lr': LEARNING_RATE, 'batch_size': BATCH_SIZE, 'trainable_blocks': TRAINABLE_BLOCKS}}\n"
                "if settings != {{'epochs': 5, 'lr': 1e-5, 'batch_size': 8, 'trainable_blocks': 2}}:\n"
                "    print({{'note': 'changed settings: this run starts again from the pinned base and replaces the default results of Sections 8-9; Section 10 compares a change side by side instead', 'settings': settings}})\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_blocks=TRAINABLE_BLOCKS, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'objective': adapt_result['objective'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'seconds': adapt_seconds}})\n"
                "val_history = {{h['epoch']: h['val']['masked_mse'] for h in adapt_result['history']}}\n"
                "if val_history[adapt_result['best_epoch']] > val_history[0]:  # a contract check on the selector, not a quality claim\n"
                "    raise RuntimeError(f\"the selector returned epoch {{adapt_result['best_epoch']}}, worse on validation than the frozen model\")\n"
                "print({{'started_from': adapt_result['started_from'], 'validation_masked_mse_by_epoch': {{k: round(v, 4) for k, v in val_history.items()}}}})"
            ),
        },
        {
            "md": (
                '**What to notice:** the validation masked MSE per epoch and `best_epoch`.\n\n<details><summary>Check your reasoning</summary>Barely. In the release run validation masked MSE went 0.2335 → 0.2338 → 0.2338 → 0.2332 → 0.2334 → 0.2337 and epoch 3 was kept — a gain of 0.0003 on 48 photographs, with no dispersion estimate. A model trained for 1,600 ImageNet epochs is already converged on natural photographs.</details>'
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation, two ways\n\n"
                "The test photographs were never used for training or epoch selection, and no photograph appears in two splits. "
                "The adapted model is scored exactly as the frozen model was in Section 6: the same seeded masks per photograph "
                "for the reconstruction (so a delta is the model's, not the mask's), and a probe re-fitted on the adapted features "
                "of the same 216 training photographs with the same seed. The systems are put side by side — three on "
                "reconstruction (mean fill, blur fill, frozen, adapted), five on the probe (majority, colour neighbour, k-NN, "
                "frozen probe, adapted probe) — the per-species breakdown is repeated and the evaluation report is written as "
                "JSON. Read it in this order: the **validation masked MSE** that selected the epoch, then the **test masked MSE** "
                "— its **mean and its median together**, and `paired_per_image`, how many test photographs got better or worse "
                "under the same masks — then the probe and the per-species recall. The selector's guarantee holds on the "
                "validation split only; held-out error can move either way. The cell records verdicts instead of asserting a "
                "gain. Ninety-six photographs from one seeded split "
                "give **no dispersion estimate**; accuracy moves in steps of one photograph; and a masked MSE that moved in the "
                "fourth decimal is the honest reading of continued pre-training on a converged model — sample-sanity evidence that "
                "the adaptation contract works, not a benchmark, and not a claim about your images until you measure them.\n\n"
                "**Predict before running:** if the mean test masked MSE barely moves, will the median move the same way?"
            ),
            "code": (
                "adapted_rec = pipe.evaluate_reconstruction(test_records, seed=0)\n"
                "adapted_val_rec = pipe.evaluate_reconstruction(val_records, seed=0)\n"
                "adapted_probe_fit = pipe.fit_probe(train_records)\n"
                "adapted_probe = pipe.evaluate(test_records)\n"
                "adapted_fields = {{c: {{'n': v['support'], 'recall': round(v['recall'], 2), 'f1': round(v['f1'], 2)}} for c, v in adapted_probe['per_class'].items()}}\n"
                "comparison = {{\n"
                "    'reconstruction': {{k: {{'mean_patch_fill': round(frozen_rec['baselines']['mean_patch_fill'][k], 4), 'blur_fill': round(frozen_rec['baselines']['blur_fill'][k], 4), 'frozen': round(frozen_rec[k], 4), 'adapted': round(adapted_rec[k], 4)}} for k in RK}},\n"
                "    'reconstruction_delta_vs_frozen': {{k: round(adapted_rec[k] - frozen_rec[k], 5) for k in RK}},\n"
                "    'validation_masked_mse': {{'frozen': round(val_history[0], 4), 'selected_epoch': adapt_result['best_epoch'], 'selected': round(val_history[adapt_result['best_epoch']], 4), 'rescored': round(adapted_val_rec['masked_mse'], 4)}},\n"
                "    'probe': {{k: {{'majority': round(baseline_majority[k], 3), 'colour_neighbour': round(baseline_neighbour[k], 3), 'knn_frozen': round(frozen_probe['knn'][k], 3), 'knn_adapted': round(adapted_probe['knn'][k], 3), 'frozen': round(frozen_probe[k], 3), 'adapted': round(adapted_probe[k], 3)}} for k in CK}},\n"
                "    'probe_delta_vs_frozen': {{k: round(adapted_probe[k] - frozen_probe[k], 3) for k in CK}},\n"
                "    'by_species': {{c: {{'n': frozen_fields[c]['n'], 'frozen_recall': frozen_fields[c]['recall'], 'adapted_recall': adapted_fields[c]['recall'], 'frozen_f1': frozen_fields[c]['f1'], 'adapted_f1': adapted_fields[c]['f1']}} for c in classes}},\n"
                "}}\n"
                "frozen_by_id = {{row['id']: row['masked_mse'] for row in frozen_rec['per_image']}}\n"
                "paired_deltas = [row['masked_mse'] - frozen_by_id[row['id']] for row in adapted_rec['per_image']]\n"
                "comparison['paired_per_image'] = {{'better': sum(d < 0 for d in paired_deltas), 'worse': sum(d > 0 for d in paired_deltas), 'same': sum(d == 0 for d in paired_deltas), 'of': len(paired_deltas)}}\n"
                "def direction(new, old, lower_is_better):\n"
                "    if new == old:\n"
                "        return 'no change'\n"
                "    return 'improved' if (new < old) == lower_is_better else 'worse'\n"
                "# Reported verdicts, not assertions: a continuation that does not help is a result to record, and export and reload still run.\n"
                "comparison['verdicts'] = {{\n"
                "    'frozen_vs_fills': frozen_rec_verdict,\n"
                "    'frozen_probe_vs_floor': frozen_probe_verdict,\n"
                "    'adapted_below_blur_fill': bool(adapted_rec['masked_mse'] < adapted_rec['baselines']['blur_fill']['masked_mse']),\n"
                "    'test_masked_mse_mean': direction(adapted_rec['masked_mse'], frozen_rec['masked_mse'], True),\n"
                "    'test_masked_mse_median': direction(adapted_rec['masked_mse_median'], frozen_rec['masked_mse_median'], True),\n"
                "    'probe_accuracy': direction(adapted_probe['accuracy'], frozen_probe['accuracy'], False),\n"
                "}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': DEFAULT_MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'classes': classes,\n"
                "    'display_names': display_names,\n"
                "    'reconstruction': {{'frozen_test': {{k: v for k, v in frozen_rec.items() if k != 'per_image'}}, 'adapted_test': {{k: v for k, v in adapted_rec.items() if k != 'per_image'}}, 'adapted_validation': {{k: v for k, v in adapted_val_rec.items() if k != 'per_image'}}}},\n"
                "    'probe': {{'baselines': {{'majority': baseline_majority, 'colour_neighbour': baseline_neighbour}}, 'frozen_test': {{k: v for k, v in frozen_probe.items() if k != 'predictions'}}, 'frozen_fit': frozen_probe_fit, 'adapted_test': {{k: v for k, v in adapted_probe.items() if k != 'predictions'}}, 'adapted_fit': adapted_probe_fit}},\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "if abs(adapted_val_rec['masked_mse'] - val_history[adapt_result['best_epoch']]) >= 1e-4:  # a contract check: the kept epoch re-scores to the number that selected it\n"
                "    raise RuntimeError('the kept epoch does not re-score to the validation masked MSE that selected it')\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the mean and median rows of `reconstruction_delta_vs_frozen`, `paired_per_image`, the probe and the `verdicts`.\n\n<details><summary>Check your reasoning</summary>No. In the release run the mean test masked MSE moved from 0.2281 to 0.2280 (−0.00005) while the median rose from 0.1801 to 0.1827 (+0.0026) — about fifty times larger and in the opposite direction: the typical photograph got slightly worse while the mean stayed flat. The probe stayed at 0.365 accuracy and 0.365 macro F1. The selector's guarantee is about the 48 validation photographs, not the test split; a single metric does not characterise the change.</details>"
            ),
        },
        {
            "md": (
                "## 9. Re-reconstruct the drawn shapes, export the adapter and reload it\n\n"
                "The three shapes from Section 5 are reconstructed again by the adapted model under the same seeded masks — "
                "drawings, a different image family from the photographs it was tuned on, so this is a small look at what the "
                "continuation did *outside* its corpus (the build record's numbers are in `docs/release-verification.md`; a "
                "changed error here is a finding to record, not a failure) — and reported with the per-call `evaluation_report` "
                "(`sample-sanity`). Both readings are written as JSON with the reconstructions as PNG.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the decoder, the last two encoder blocks and the encoder "
                "LayerNorm, about 161 MB — plus the probe head and its standardisation statistics as `adapter.safetensors`, "
                "with a `manifest.json` recording the artifact format, the base model id and revision, the digest of the base "
                "`model.safetensors`, the objective and mask ratio, the probe record (classes, steps, learning rate), the tensor "
                "names, the file size and SHA-256, the training configuration and the epoch history (OUT8). "
                "`ViTMAEPipeline.from_artifact` re-verifies the base snapshot, checks the artifact manifest, its digest and its "
                "exact tensor set **before** deserialising, refuses any tensor outside the declared blocks, and overlays the "
                "tensors onto a freshly loaded base — a new object from files, not the in-memory model (VER2). The cell asserts "
                "identical masked MSE on eight test photographs under the same masks and identical probe decisions (VER4) — a "
                "contract check, so it stays a hard check.\n\n"
                "**Predict before running:** the continuation saw only photographs. Will the drawn shapes reconstruct better or "
                "worse afterwards?"
            ),
            "code": (
                "import shutil\n\n"
                "adapted_scene = pipe.reconstruct(shape_images, seed=SCENE_SEED)\n"
                "adapted_scene_rows = []\n"
                "for path, before, after in zip(shape_images, scene['results'], adapted_scene['results'], strict=True):\n"
                "    after['reconstruction'].save(f'outputs/{stem}_adapted_{{path.stem}}.png')\n"
                "    adapted_scene_rows.append({{'image': path.name, 'hidden_patches': after['hidden_patches'], 'masked_mse': after['masked_mse'], 'visible_mse': after['visible_mse']}})\n"
                "    print({{'image': path.name, 'frozen_masked_mse': round(before['masked_mse'], 4), 'adapted_masked_mse': round(after['masked_mse'], 4), 'same_mask': before['mask'] == after['mask']}})\n"
                "adapted_scene_report = evaluation_report(adapted_scene, sample_kind='synthetic')\n"
                "print({{'scene_after_adaptation': {{m['id']: round(m['value'], 4) for m in adapted_scene_report['metrics']}}, 'verdict': adapted_scene_report['verdict']}})\n"
                "with open('outputs/{stem}_shapes.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump({{'seed': SCENE_SEED, 'frozen': scene_rows, 'adapted': adapted_scene_rows}}, handle, indent=2)\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'probe_classes': artifact_manifest['probe']['classes'], 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = ViTMAEPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "parity_images = [r['image'] for r in test_records[:8]]\n"
                "before = pipe.reconstruct(parity_images, seed=11, return_images=False)['results']\n"
                "after = reloaded.reconstruct(parity_images, seed=11, return_images=False)['results']\n"
                "before_cls, after_cls = pipe.classify(parity_images)['results'], reloaded.classify(parity_images)['results']\n"
                "parity = {{\n"
                "    'max_abs_masked_mse_difference': float(max(abs(a['masked_mse'] - b['masked_mse']) for a, b in zip(before, after, strict=True))),\n"
                "    'identical_reconstructions': int(sum(abs(a['masked_mse'] - b['masked_mse']) < 1e-6 for a, b in zip(before, after, strict=True))),\n"
                "    'identical_probe_decisions': int(sum(a['top1'] == b['top1'] for a, b in zip(before_cls, after_cls, strict=True))),\n"
                "    'max_abs_probe_score_difference': float(max(abs(a['scores'][c] - b['scores'][c]) for a, b in zip(before_cls, after_cls, strict=True) for c in a['scores'])),\n"
                "    'of': len(parity_images),\n"
                "}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch'], 'reloaded_probe_classes': reloaded.probe['classes']}})\n"
                "assert parity['identical_reconstructions'] == parity['of'] and parity['identical_probe_decisions'] == parity['of'] and parity['max_abs_probe_score_difference'] < 1e-4\n\n"
                "write_provenance('outputs/provenance.json', pipeline=pipe)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'fetched_this_run': fetched, 'weight_file': MODEL_FILENAME, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': MODEL_SHA256}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'license': CORPUS_LICENSE, 'base_url': CORPUS_BASE_URL, 'bytes': CORPUS_BYTES, 'pinned_photographs': len(SAMPLE_RECORDS), 'species': {{k: list(v) for k, v in SPECIES.items()}}}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'shapes': {{'names': [p.name for p in shape_images], 'sha256': shape_sha256, 'seed': SCENE_SEED}}, 'frozen_report': frozen_scene, 'adapted_report': adapted_scene_report, 'frozen_rows': scene_rows, 'adapted_rows': adapted_scene_rows}},\n"
                "    'comparison': comparison,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'numpy': numpy.__version__, 'device': str(pipe.device), 'dtype': 'float32', 'checkpoint_source': pipe.checkpoint_source}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
        {
            "md": (
                '**What to notice:** `frozen_masked_mse` against `adapted_masked_mse` for each shape (`same_mask` should be true), and the reload parity line.\n\n<details><summary>Check your reasoning</summary>In the CPU build record the three shapes moved from 0.0005 / 0.0113 / 0.0228 to 0.0006 / 0.0117 / 0.0219 under the same masks — two slightly worse, one slightly better — and three drawings cannot tell you which is typical — record the change as a finding about the continuation outside its corpus, not as a measurement. Reload parity held in the release run: identical reconstructions and probe decisions on eight of eight test photographs.</details>'
            ),
        },
        {
            "md": (
                "## 10. Change one thing: a ten-times-higher learning rate, or another mask ratio (optional)\n\n"
                "*Evaluation practice.* A **Predict → Change one thing → Run → Observe → Explain** activity, off by default so "
                "Run all is unaffected. Set `RUN_EXPERIMENT = True`, change **one** field — by default the learning rate goes from "
                "1e-5 to 1e-4; `EXPERIMENT_MASK_RATIO` changes the mask ratio for both the continuation and its own frozen control "
                "— and run this cell after Sections 4–9. The experiment loads its **own** pipeline from the verified snapshot, so it "
                "starts from the checkpoint and never touches the default `pipe`; it scores its own frozen control at the same mask "
                "ratio, writes only to `outputs/{stem}_experiment/`, prints the default and the changed run side by side "
                "(including epoch 0, which must equal the default's at the default ratio), and checks that the default exports "
                "are byte-identical afterwards.\n\n"
                "**Predict:** at 1e-4, which epoch will the selector keep?"
            ),
            "code": (
                "RUN_EXPERIMENT = False  # @param {{type:\"boolean\"}}\n"
                "EXPERIMENT_LEARNING_RATE = 1e-4  # @param {{type:\"number\"}}\n"
                "EXPERIMENT_MASK_RATIO = 0.75  # @param {{type:\"number\"}}\n"
                "EXPERIMENT_TRAINABLE_BLOCKS = 2  # @param {{type:\"integer\"}}\n\n"
                "if not RUN_EXPERIMENT:\n"
                "    print({{'experiment': 'skipped (RUN_EXPERIMENT = False); the default path above is complete'}})\n"
                "else:\n"
                "    canonical_files = {{'adapter': artifact_dir / 'adapter.safetensors', 'evaluation_report': Path('outputs/{stem}_evaluation_report.json'), 'result': Path('outputs/{stem}_result.json')}}\n"
                "    canonical = {{name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in canonical_files.items()}}\n"
                "    experiment_dir = Path('outputs/{stem}_experiment')\n"
                "    shutil.rmtree(experiment_dir, ignore_errors=True)\n"
                "    experiment_dir.mkdir(parents=True)\n"
                "    # Its own pipeline from the verified snapshot: the experiment starts from the checkpoint and the default pipe is untouched.\n"
                "    experiment_pipe = ViTMAEPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "    experiment_control = experiment_pipe.evaluate_reconstruction(test_records, mask_ratio=EXPERIMENT_MASK_RATIO, seed=0)\n"
                "    experiment_result = experiment_pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=EXPERIMENT_LEARNING_RATE, batch_size=BATCH_SIZE, trainable_blocks=EXPERIMENT_TRAINABLE_BLOCKS, mask_ratio=EXPERIMENT_MASK_RATIO, progress=report)\n"
                "    experiment_rec = experiment_pipe.evaluate_reconstruction(test_records, mask_ratio=EXPERIMENT_MASK_RATIO, seed=0)\n"
                "    side_by_side = {{\n"
                "        'settings': {{'default': {{'lr': adapt_result['lr'], 'mask_ratio': adapt_result['mask_ratio'], 'trainable_blocks': adapt_result['trainable_blocks']}}, 'experiment': {{'lr': EXPERIMENT_LEARNING_RATE, 'mask_ratio': EXPERIMENT_MASK_RATIO, 'trainable_blocks': EXPERIMENT_TRAINABLE_BLOCKS}}}},\n"
                "        'validation_masked_mse_by_epoch': {{'default': [round(h['val']['masked_mse'], 4) for h in adapt_result['history']], 'experiment': [round(h['val']['masked_mse'], 4) for h in experiment_result['history']]}},\n"
                "        'best_epoch': {{'default': adapt_result['best_epoch'], 'experiment': experiment_result['best_epoch']}},\n"
                "        'test_masked_mse': {{'default_frozen': round(frozen_rec['masked_mse'], 4), 'default_adapted': round(adapted_rec['masked_mse'], 4), 'experiment_frozen_control': round(experiment_control['masked_mse'], 4), 'experiment_adapted': round(experiment_rec['masked_mse'], 4)}},\n"
                "        'test_masked_mse_median': {{'default_frozen': round(frozen_rec['masked_mse_median'], 4), 'default_adapted': round(adapted_rec['masked_mse_median'], 4), 'experiment_frozen_control': round(experiment_control['masked_mse_median'], 4), 'experiment_adapted': round(experiment_rec['masked_mse_median'], 4)}},\n"
                "    }}\n"
                "    for key, row in side_by_side.items():\n"
                "        print({{key: row}})\n"
                "    with open(experiment_dir / 'experiment_report.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump({{'side_by_side': side_by_side, 'history': experiment_result['history']}}, handle, indent=2, ensure_ascii=False, default=str)\n"
                "    unchanged = {{name: hashlib.sha256(path.read_bytes()).hexdigest() == canonical[name] for name, path in canonical_files.items()}}\n"
                "    if not all(unchanged.values()):\n"
                "        raise RuntimeError(f'the experiment changed a default export: {{unchanged}}')\n"
                "    print({{'default_exports_unchanged': unchanged, 'experiment_outputs': str(experiment_dir)}})\n"
                "    del experiment_pipe"
            ),
        },
        {
            "md": (
                "**Observe → Explain.** Compare the two validation curves (epoch 0 must be equal at the default mask ratio) and the "
                "test rows against each run's own frozen control.\n\n"
                "<details><summary>Check your reasoning</summary>The build record's sweep found that `1e-4` makes the validation "
                "masked MSE worse from the first epoch — batches of eight with random masks are a noisy gradient — so the "
                "selector keeps epoch 0, the frozen model, and the experiment's adapted row equals its control. That is the "
                "selector doing its job. With a lower mask ratio the encoder sees more of each image, so both the control and "
                "the adapted masked MSE fall; compare each run only with its own control. No experiment run is recorded on the "
                "release runtime.</details>"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen model reconstructs hidden patches of photographs it was never trained on far better than the two "
        "non-neural fills (masked MSE 0.2281 against 0.5452 and 0.7652 in the build record), and its "
        "mean-pooled features carry enough to lift a linear probe over six bird species above the majority floor, the colour "
        "neighbour and a k-NN on the same features (0.365 against 0.167, 0.260 and 0.219) — while "
        "still being, as the paper says of MAE features, a poor linear-probe backbone until fine-tuned. A bounded continuation "
        "of the masked-autoencoding objective on 216 photographs, chosen on the validation split, moved the held-out mean "
        "masked MSE from 0.2281 to 0.2280 while the median rose from 0.1801 to 0.1827 (Kaggle T4 release run, 21 September "
        "2026), left the probe at 0.365, and exported a 161 MB adapter that reloads to identical reconstructions and probe "
        "decisions. That is the claim: the adaptation contract works end to end on a real photograph set, the selector is "
        "honest enough to keep the frozen model when nothing beats it on validation — a guarantee about the validation split, "
        "not the test split —  and the numbers it produces are read on the objective and on a probe, per species, against non-neural baselines "
        "and the frozen model rather than in isolation.\n\n"
        "What the numbers say is that continued pre-training on a few hundred in-domain photographs does not change a model "
        "that has seen 1,600 epochs of ImageNet: the build record's learning-rate sweep (`docs/release-verification.md`) "
        "found `1e-4` hurts from the first epoch and `1e-5` moves the validation masked MSE in the fourth decimal. Where "
        "continuation earns its place is a domain the checkpoint has not seen — medical, satellite, microscopy, line art — "
        "and that is what BYOD is for; the check in Section 7 (the kept epoch is never worse than the frozen model on "
        "validation) and "
        "the baselines in Sections 6 and 8 are what make such a run readable. The test split is 96 photographs from one "
        "seeded draw of one sample, the validation split that picks the epoch is 48, a masked MSE is the model's own "
        "objective and not a perceptual judgement, and the probe's accuracy moves in steps of one photograph.\n\n"
        "Three things to carry to real data. **Baselines first:** the two fills and the three non-neural classifiers on *your* "
        "images are the numbers to read before any adapted one. **Leakage:** keep every photograph in one split (the contract "
        "de-duplicates by decoded pixels) and split by photographer or session when your images come from few sources — the "
        "sample's observer overlap is printed for exactly that reason. **The readout is not the product:** a linear probe "
        "measures the features; a classifier you would ship is a full fine-tuning of the encoder, which this repository does "
        "not expose.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this standalone notebook, can "
        "acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real photograph set, validate the "
        "demonstrated dataset contract without leakage, execute the inference contract and a bounded continuation of the "
        "pre-training objective, evaluate against non-neural baselines and the frozen model on an image-disjoint split two "
        "ways, and emit the shown machine-readable artifacts — without the repository being reachable. It does **not** "
        "establish benchmark superiority, representation quality on any other population or camera, or production fitness.\n\n"
        "**Optional experiments (off by default; each names its field and what to run):** Section 10 runs a changed "
        "continuation in its own pipeline beside the default — `EXPERIMENT_LEARNING_RATE` (1e-4 by default: watch the "
        "selector keep epoch 0), `EXPERIMENT_MASK_RATIO` (for example 0.5, scored against its own frozen control) or "
        "`EXPERIMENT_TRAINABLE_BLOCKS` (0 trains the decoder alone; compare the artifact size); change one and run that cell "
        "again. Changing `EPOCHS`, `LEARNING_RATE` or `TRAINABLE_BLOCKS` and choosing **Run after** from Section 7 also starts "
        "from the pinned base — every `adapt` restores it first — but replaces the default results and exports. BYOD: "
        "`USE_BYOD` and `BYOD_PATH` in Section 4, then **Run after** from Section 4; bring images from an unfamiliar domain "
        "and read the fills before the adapted number.\n\n"
        "## Troubleshooting\n\n"
'- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — you are on Windows, macOS or an ARM machine. Use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused, an incomplete one is finished. If it repeats, the network is blocking or altering `files.pythonhosted.org` or `pypi.org`.\n- **"The isolated environment\'s Python process exited"** — usually out of memory. Restart the session and choose **Run all**; leave the optional experiment off on a small runtime.\n- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable, so the cells after it keep working. After a session restart, run from the top.\n- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message names the file. Delete it from the snapshot folder Section 3 prints and run Section 3 again; the snapshot comes from `huggingface.co`.\n- **Section 4 cannot fetch a photograph, or one fails its digest** — `fetch_corpus` names it; the default path needs `inaturalist-open-data.s3.amazonaws.com`. Run Section 4 again (cached photographs are re-hashed); delete `weights/inat-birds/` if a cached file is corrupt.\n- **Out of memory** — lower `BATCH_SIZE` in Section 7, or restart the session and choose **Run all**; leave Section 10 off on a small runtime.\n- **BYOD: "BYOD_PATH … does not exist"** — the path is relative to the working directory printed in the message.\n- **BYOD: "the upload dialog exists only in Google Colab"** — on Kaggle or Jupyter, put the zip in the runtime (or attach it as a dataset) and set `BYOD_PATH`.\n- **BYOD: "Upload exactly one .zip file"** — the dialog was cancelled or several files were chosen; run the cell again.\n- **BYOD: "labels.csv line N (file …): that image file is not in the dataset"** — fix the `file` column of that row, or add the image to the zip.\n- **BYOD: "… Pillow cannot decode the image"** — the file on that line is corrupt or not an image.\n- **BYOD: "split leaves no train/validation photograph for labels …" or "… training records"** — add photographs; the message names the minimum for your label count.\n'
        "## Glossary\n\n"
        "- **Mask / mask ratio** — the random set of hidden patches; 75 % of 196 by default, seeded per photograph here.\n"
        "- **Masked MSE / visible MSE** — the mean squared error over the hidden patches (the model's objective) and over "
        "the visible ones, in the processor's normalised pixel space.\n"
        "- **PSNR** — peak signal-to-noise ratio of the hidden patches, in dB; higher is better.\n"
        "- **Mean-patch fill / blur fill** — two non-neural reconstructions: the mean visible colour, and the mean of each "
        "hidden patch's visible neighbours.\n"
        "- **Mean pooling / standardisation** — averaging the patch tokens into one vector; scaling each feature by the "
        "training set's mean and standard deviation.\n"
        "- **Linear probe / k-NN** — a logistic-regression head on frozen features; a nearest-neighbour vote on the same "
        "features.\n"
        "- **Accuracy / macro F1** — the fraction right; the mean of per-class F1.\n"
        "- **Epoch selection** — keeping the epoch with the lowest validation masked MSE; a guarantee on validation only.\n"
        "- **Paired per-image count** — how many test photographs got better or worse under the same masks.\n"
        "- **Adapter / reload parity** — the trained tensors and the probe head (safetensors) overlaid on the pinned base; "
        "the reloaded pipeline gives identical outputs.\n"
        "- **BYOD** — bring your own data: your photographs through the same cells.\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional — fill in from **your** run, not the recorded one:\n\n"
        "- The data was ___ photographs over ___ labels; test split ___.\n"
        "- Frozen masked MSE ___ (median ___) against the blur fill ___ and the mean fill ___.\n"
        "- Probe accuracy ___ against the floor ___, the colour neighbour ___ and the k-NN ___.\n"
        "- After continuation (epoch ___ kept): masked MSE ___ (median ___), ___ photographs better and ___ worse; probe ___.\n"
        "- What I would need before claiming continuation helps: ___ (for example an unfamiliar domain, more test images, several seeds).\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/vit-mae-pretraining-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/vit-mae-pretraining-pipeline/blob/main/MODEL_CARD.md\n"
        "- Sample dataset card (synthetic shapes): https://github.com/kurtvalcorza/vit-mae-pretraining-pipeline/blob/main/examples/sample-data/DATASET_CARD.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/facebookresearch/mae — served through https://github.com/huggingface/transformers (`ViTMAEForPreTraining`)\n"
        "- Masked Autoencoders Are Scalable Vision Learners (He, Chen, Xie, Li, Dollár and Girshick, CVPR 2022): https://arxiv.org/abs/2111.06377\n"
        "- The SigLIP zero-shot row in this fleet, sharing the corpus and the code shape: https://github.com/kurtvalcorza/siglip-v1-zero-shot-pipeline\n"
        "- iNaturalist open data (CC0 photographs, each observer's own licence): https://www.inaturalist.org/pages/developers — bucket https://inaturalist-open-data.s3.amazonaws.com/\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
