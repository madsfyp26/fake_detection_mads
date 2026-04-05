---
title: Fake Audio Detection
emoji: 🏆
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.44.1
app_file: unified_deepfake_app.py
pinned: false
license: apache-2.0
short_description: Lightweight ML method to detect forged / synthetic audio
---

## What this repo contains

This repo has **two** detection pipelines and one unified Streamlit demo:

- **NOMA (audio-only)**: classical ML (hand-crafted features + SVM) via Mozilla’s `fake-audio-detection`. Input: **audio** files (e.g. WAV/MP3/OGG).
- **AVH-Align (audio-visual)**: AV-HuBERT feature extraction + fusion model score (lip–speech mismatch). Input: **video** (face + audio). The model consumes **visual frames** (mouth/ROI image sequence) and **waveforms** — not a standalone still-image upload in the default video flow; advanced users can also score from a **`.npz`** with pre-extracted `visual` + `audio` tensors.

The Streamlit app includes an **interface language** selector (English / Español / हिन्दी) on Home and Research pages. Detection models are **not** language-specific in the same way as NLP: performance still depends on training data and acoustic/visual conditions. The **Research chat** accepts questions in **many languages** (text in → Gemini + APIs out).
- **Unified demo**: `unified_deepfake_app.py` exposes:
  - `NOMA (Audio-Only)`
  - `AVH-Align (Audio-Visual)`
  - `Combined (AVH → NOMA)` + **reliability-weighted fusion** of calibrated scores + optional **Grad-CAM evidence** + optional **robustness delta** + **evidence bundle export** + multimodal corroboration / optional CMID embeddings.

## Quickstart (recommended)

### 1) Streamlit (venv) for the unified demo

```bash
cd /Users/soham/fake-audio-detection
python -m venv venv
source venv/bin/activate
pip install -r requirements-venv.txt
streamlit run unified_deepfake_app.py
```

### 2) AVH pipeline (conda env `avh`)

AVH needs a separate environment (fairseq/omegaconf compatibility). Create it from:
- `environment-avh.yml`

Then point Streamlit sidebar **Python for AVH video** to your conda python:
- Example: `/opt/homebrew/Caskroom/miniforge/base/envs/avh/bin/python`

## Spatial pre-crop (reels / on-screen UI)

Before the mouth ROI step, AVH can **spatially crop** the frame to reduce caption bars and UI chrome on vertical video (`--smart_crop` on `AVH/test_video.py`, default `auto`). See `AVH/smart_spatial_crop.py`. Audio is always taken from the **original** upload.

In **Streamlit**, you can optionally **draw a rectangle** on the first frame (requires `streamlit-drawable-canvas` + `Pillow`) to lock the region before AVH; that disables automatic spatial pre-crop for that run. See `ui/video_manual_crop.py`.

## Forensics evidence outputs (panel-ready)

In **Combined (AVH → NOMA)**:
- **Grad-CAM**: mouth ROI overlays explain visual sensitivity (evidence, not proof).
- **Robustness delta**: baseline AVH score vs adversarially-perturbed score (feature-space hard misalignment).
- **Evidence bundle export**: download a `.zip` containing:
  - `manifest.json` with sha256 hashes + scores
  - extracted `audio.wav` + `mouth_roi.mp4`
  - Grad-CAM overlays + `index.json`
  - NOMA predictions CSV

## Offline / research tools

- [tools/calibration_fit.py](tools/calibration_fit.py) — calibration fitting (CLI).
- [tools/leakage_audit.py](tools/leakage_audit.py) — split/leakage checks (CLI).

### FakeAVCeleb-style evaluation (manifest + metrics)

Download **FakeAVCeleb** via the [official release](https://sites.google.com/view/fakeavcelebdash-lab/download) (no scraping in-repo). Then:

1. Build a manifest CSV from the extracted tree (path heuristics for `RealVideo-RealAudio` vs fake combos):

```bash
PYTHONPATH=. python tools/prepare_fakeavceleb_manifest.py \
  --dataset_root /path/to/FakeAVCeleb_extracted \
  --out_csv fakeavceleb_manifest.csv
```

If your folder names differ, use `--from_list paths.txt` with lines `absolute_or_relative_path,label` (`0` = real, `1` = fake).

2. Run batched Combined inference (no Grad-CAM by default; cap e.g. 600 clips on 8GB RAM):

```bash
PYTHONPATH=. python tools/evaluate_fakeavceleb.py \
  --manifest_csv fakeavceleb_manifest.csv \
  --python_exe /path/to/conda/envs/avh/bin/python \
  --max_videos 600 \
  --out_dir eval_runs/fakeavceleb_run
```

Outputs: `results.csv` (per-video scores + **reliability fused** `p_fused`) and `metrics.json` (AUROC / AP / Brier / ECE when both classes exist).

3. **Optional:** refit `calibration_artifacts.json` from the same run:

```bash
PYTHONPATH=. python tools/evaluate_fakeavceleb.py ... --write_calibration_csvs
PYTHONPATH=. python tools/calibration_fit.py \
  --avh_csv eval_runs/fakeavceleb_run/avh_for_calib.csv \
  --noma_csv eval_runs/fakeavceleb_run/noma_for_calib.csv
```

## Research assistant (optional): Serp + Google Lens + News + Gemini

Set API keys via environment (see [.env.example](.env.example)):

| Variable | Purpose |
|----------|---------|
| `SERPAPI_API_KEY` | SerpAPI: Google organic web + Google Lens (same key) |
| `NEWS_API_KEY` | [NewsAPI.org](https://newsapi.org) |
| `GEMINI_API_KEY` | Google AI Gemini synthesis |
| `GEMINI_MODEL` | Optional (default: `gemini-2.5-flash`) |
| `TELEGRAM_BOT_TOKEN` | For the Telegram bot only |

- **Streamlit:** open **Research chat** in the sidebar. Optionally include the last **Combined** run summary as extra context for Gemini.
- **Telegram:** run `PYTHONPATH=. python -m telegram_bot.run_bot` with `TELEGRAM_BOT_TOKEN` set. Same orchestration as Streamlit; chat history is kept per Telegram user for the process lifetime.

Outputs are **not** legal advice; external APIs and LLMs can be wrong or incomplete.

## Notes
- Shared Streamlit CSS lives in [ui/streamlit_css.py](ui/streamlit_css.py).
- `normalization.py` defines `CustomNormalizer` required by the bundled NOMA joblib pipeline (do not remove).
- You must download/copy required AVH checkpoints into the expected `AVH/` paths (see sidebar “AVH setup status” in the app).
- Extra AVH Python interpreters: set `AVH_PYTHON_ALLOWLIST_EXTRA` to a comma-separated list of absolute paths (appended to the built-in allowlist in [config.py](config.py)).
