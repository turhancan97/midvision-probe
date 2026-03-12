# SpaRRTA (Hugging Face Space App)

This folder contains a Gradio app for serving **SpaRRTA** style spatial probing.

## Standalone design

The Space app is intentionally isolated from the parent repository. It does not import code from `evals/` or other project modules.

- `app.py`: Gradio UI and runtime orchestration.
- `src/geometry.py`: local Unreal metadata parsing + relative-direction logic.
- `src/probes.py`: `EfficientProbing` head class (checkpoint-compatible).
- `src/attention.py`: attention selection and overlay rendering.
- `src/gradio_compat.py`: Gradio boolean-schema compatibility patch.
- `src/ui_theme.py`: custom desktop-first visual theme + favicon head injection.
- `src/ui_components.py`: reusable HTML cards/banners/legend/footer renderers.
- `src/ui_state.py`: UI settings serialization + browser localStorage persistence script.

## What this app does

- Supports both `camera` and `human` perspectives.
- Uses timm DINOv3 backbones only.
- Extracts **last-layer patch tokens** only.
- Runs **EfficientProbing** heads.
- Shows:
  - top-2 predictions + probabilities,
  - raw logits,
  - selected attention visualization (`mean`, `max`, `min`, `std`, `q1`, `q2`, `q3`, `q4`),
  - split/tabs result modes and optional compare mode,
  - loading stage indicators (`Loading backbone`, `Loading probe`, `Running inference`),
  - copy-settings + reset/recommended preset actions,
  - downloadable selected attention heatmap.

## Artifact packaging

All packaged artifacts live in `space/artifacts/`:

- `manifest.yaml`: model/triplet/checkpoint mapping.
- `samples/`: demo sample images + JSON metadata.
- `checkpoints/`: efficient probing checkpoints.

## Important note about packaged checkpoints

The non-experimental checkpoints included here are **placeholder checkpoints** to keep the app self-contained.

For real probe behavior, replace these checkpoint files with your trained heads while keeping the same paths defined in `manifest.yaml`.

## Experimental backbones

- `vit_7b_patch16_dinov3.lvd1689m`

They are hidden by default and only enabled when:

- runtime has GPU, and
- all required packaged checkpoints exist for that backbone.

## Run locally

```bash
cd space
pip install -r requirements.txt
python app.py
```

## Smoke test

```bash
cd space
python smoke_test.py
python smoke_test.py --run-infer
```
