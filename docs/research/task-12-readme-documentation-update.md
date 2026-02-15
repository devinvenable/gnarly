# Task 12 Research: README Documentation Update

## Scope
Update `README.md` with:
- all new CLI flags with examples
- GPU feature table
- examples for each feature combination
- requirements documentation refresh

## Key Findings

1. `README.md` is missing many current headless CLI flags.
Current README headless options only mention `--frames`, `--rule`, `--grid-scale`, `--blend-alpha`, `--fps`, `--max-dimension` (`README.md:30`), but `gnarly/cli.py` defines additional flags for Deep Dream, zoom, detection, morphing, and faces (`gnarly/cli.py:114`, `gnarly/cli.py:145`, `gnarly/cli.py:173`, `gnarly/cli.py:202`, `gnarly/cli.py:224`).

2. GPU requirements are explicit in runtime validation.
`run_headless()` exits if GPU-only features are enabled without CUDA (`gnarly/runners/headless.py:144`). GPU-gated checks are:
- Deep Dream (`gnarly/runners/headless.py:43`)
- detection path (`gnarly/runners/headless.py:48`)
- morphing (`gnarly/runners/headless.py:57`)

3. Feature composition order is deterministic and should be documented.
Effects are chained as:
- always `CAEffect` first (`gnarly/runners/headless.py:195`)
- optional `ZoomEffect` (`gnarly/runners/headless.py:205`)
- optional `DeepDreamEffect` (`gnarly/runners/headless.py:214`)
- optional `MorphEffect` (`gnarly/runners/headless.py:221`)

4. Input type changes behavior for `--frames` and `--fps`.
For video input, output frames/fps come from source video; for image input, output frames/fps come from CLI config (`gnarly/runners/headless.py:187`, `gnarly/runners/headless.py:191`).

5. There is a packaging/import coupling issue affecting "minimal" usage.
`python -m gnarly --help` currently fails if `torch` is not installed because:
- `gnarly/cli.py` imports `STYLE_NAMES` from morph effect (`gnarly/cli.py:9`)
- `gnarly/effects/morph_effect.py` imports `..detection.base` (`gnarly/effects/morph_effect.py:13`)
- package import executes `gnarly/detection/__init__.py`, which imports YOLO (`gnarly/detection/__init__.py:5`)
- YOLO imports torch (`gnarly/detection/yolo.py:6`)
This conflicts with README messaging that headless mode is minimal using `requirements-core.txt` (`README.md:20`, `README.md:55`).

## Full Headless CLI Flags (Source of Truth)
From `gnarly/cli.py`:

- Positional/required:
  - `input` (`gnarly/cli.py:50`)
  - `-o, --output` required (`gnarly/cli.py:56`)
- General:
  - `-V, --version` (`gnarly/cli.py:44`)
  - `--frames` (`gnarly/cli.py:63`)
  - `--rule` (`gnarly/cli.py:70`)
  - `--grid-scale` (`gnarly/cli.py:79`)
  - `--divisor` (`gnarly/cli.py:86`)
  - `--blend-alpha` (`gnarly/cli.py:93`)
  - `--fps` (`gnarly/cli.py:100`)
  - `--max-dimension` (`gnarly/cli.py:107`)
- Deep Dream:
  - `--deep-dream` (`gnarly/cli.py:114`)
  - `--dream-iterations` (`gnarly/cli.py:120`)
  - `--dream-lr` (`gnarly/cli.py:127`)
  - `--dream-layers` (`gnarly/cli.py:134`)
- Zoom:
  - `--zoom` (`gnarly/cli.py:145`)
  - `--zoom-speed` (`gnarly/cli.py:151`)
  - `--zoom-min` (`gnarly/cli.py:158`)
  - `--zoom-max` (`gnarly/cli.py:165`)
- Detection:
  - `--detect` (`gnarly/cli.py:173`)
  - `--detect-model {yolo,efficientdet}` (`gnarly/cli.py:179`)
  - `--detect-confidence` (`gnarly/cli.py:187`)
  - `--detect-interval` (`gnarly/cli.py:194`)
- Morph:
  - `--morph` (`gnarly/cli.py:202`)
  - `--morph-style` (`gnarly/cli.py:208`)
  - `--creativity` (`gnarly/cli.py:216`)
- Face:
  - `--faces` (`gnarly/cli.py:224`)

Validation behavior:
- input must exist (`gnarly/cli.py:233`)
- creativity must be 0.0 to 1.0 (`gnarly/cli.py:235`)

## GPU Feature Table Data
Use these rows in README:

- CA core: GPU required = No (no CUDA checks; always present in pipeline) (`gnarly/runners/headless.py:195`)
- Zoom: GPU required = No (`gnarly/runners/headless.py:205`)
- Face detection (`--faces`): GPU required = No (uses dlib, no CUDA guard) (`gnarly/runners/headless.py:77`, `gnarly/detection/face.py:76`)
- Deep Dream (`--deep-dream`): GPU required = Yes (`gnarly/runners/headless.py:43`, `gnarly/effects/deep_dream_effect.py:138`)
- Object detection (`--detect`): GPU required = Yes by current guard logic (`gnarly/runners/headless.py:49`, `gnarly/runners/headless.py:52`)
- Morph (`--morph`): GPU required = Yes (`gnarly/runners/headless.py:57`)

## Feature Combination Examples (Recommended to add)
These align to implemented feature toggles and execution path:

1. CA only (CPU-safe):
```bash
python -m gnarly input.jpg -o out.mp4
```

2. CA + zoom (CPU-safe):
```bash
python -m gnarly input.jpg -o out.mp4 --zoom --zoom-speed 1.01 --zoom-max 1.6
```

3. CA + faces (CPU-safe):
```bash
python -m gnarly input.mp4 -o out.mp4 --faces
```

4. CA + Deep Dream (GPU):
```bash
python -m gnarly input.jpg -o out.mp4 --deep-dream --dream-iterations 20 --dream-lr 0.015
```

5. CA + detection (GPU):
```bash
python -m gnarly input.mp4 -o out.mp4 --detect --detect-model yolo --detect-confidence 0.55 --detect-interval 20
```

6. CA + morph (GPU; detection auto-created):
```bash
python -m gnarly input.mp4 -o out.mp4 --morph --morph-style cyberpunk --creativity 0.8
```

7. CA + detection + morph + Deep Dream + zoom (GPU full pipeline):
```bash
python -m gnarly input.mp4 -o out.mp4 \
  --zoom --deep-dream --detect --detect-model efficientdet --morph \
  --morph-style surrealistic --creativity 0.7
```

8. Image-specific frame count example:
```bash
python -m gnarly input.jpg -o out.mp4 --frames 480 --fps 24
```

## Requirements Documentation Data

- Minimal dependencies (`requirements-core.txt`):
  - `numpy`, `opencv-python`, `Pillow`, `tqdm` (`requirements-core.txt:1`)
- Full dependencies (`requirements.txt`) add:
  - `torch`, `torchvision`, `diffusers`, `transformers`, `effdet`, `dlib`, `pandas`, `scipy`, `pygame`, `huggingface-hub`, `accelerate` (`requirements.txt:2`)

Important accuracy note for README:
- Current code path imports torch-related modules even for CLI startup (see "packaging/import coupling issue" above). If README continues claiming `requirements-core.txt` is sufficient for `python -m gnarly`, that is inaccurate until import coupling is fixed.

## Bottrace Evidence

Executed runtime trace for CA core flow (minimal executable path in this environment):

Command:
```bash
PYTHONPATH=. bottrace run --calls --call-counts 30 --max-depth 6 --backend settrace --include /home/devin/src/2026/gnarly/worktrees/docs/gnarly --timeout 5 /tmp/gnarly_bottrace_ca.py
```

Observed calls:
- `CAEngine.__init__` at `gnarly/core/ca.py:75`
- `initialize_grid` at `gnarly/core/ca.py:8`
- `step` at `gnarly/core/ca.py:92` (3 calls)
- `update_grid` at `gnarly/core/ca.py:26` (3 calls)

This confirms CA engine execution and the expected core update loop.
