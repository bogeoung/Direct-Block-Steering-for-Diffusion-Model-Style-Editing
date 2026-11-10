# DBsteer core implementation

This directory contains only the original implementation of Direct-Block
Steering (DBsteer): estimate a raw style--neutral difference-of-means vector
at a transformer-block output, then inject that fixed vector during SDXL
denoising.

It deliberately does **not** include third-party source code, pretrained
weights, SAE checkpoints, generated images, experimental features, metrics,
or comparison methods. In particular, it excludes SDXL-Unbox, StyleAligned,
all SAE steering recipes, prompt-only baselines, comparison/evaluation scripts,
and paper figure scripts.

## Contents

- `scripts/extract_direction.py`: produces raw block-update statistics from
  paired neutral and styled generations.
- `scripts/generate_dbsteer.py`: estimates the DBsteer direction and injects
  it into a selected SDXL transformer block.
- `src/dbsteer/runtime.py`: block hooks, SDXL loading, and generation helpers
  written for this project.

## External dependencies

The code calls the public hook-enabled SDXL pipeline supplied by SDXL-Unbox,
but does not distribute its source or SAE checkpoints. Clone/install that
project separately and set `SDXL_UNBOX_PATH` to its checkout before running.
The required third-party projects and models are listed in
`THIRD_PARTY_NOTICES.md`.

Example:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export SDXL_UNBOX_PATH=/path/to/sdxl-unbox

python scripts/extract_direction.py \
  --conditions photo watercolor \
  --output-dir outputs/directions

python scripts/generate_dbsteer.py \
  --style watercolor \
  --direction-dir outputs/directions \
  --output-dir outputs/generation
```

## Provenance and scope

This export is intended to identify the project-authored DBsteer code for
copyright-registration review. It does not assert ownership of any external
dependency, model, checkpoint, metric, or baseline method. Confirm the final
submission scope and authorship with the institution or copyright professional
handling the registration.
