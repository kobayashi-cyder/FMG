# FMG — Fly Media Generator

Current image core: **FMG Image Generator Connectome V2.4**

## Production profile

- Resolution: **1024 × 1024 fixed**
- Default inference steps: **50**
- Default CFG / guidance: **9.0**
- Default model: `segmind/SSD-1B`
- Controller: fly-connectome-inspired sparse routing
- Qwen: not used
- Cloud API: not required after local model preparation

The fixed 1024 image plane is intentional. Width/height variation was removed
from the production control path so the connectome does not waste sensory state
on a parameter that is constant for this FMG profile.

Legacy API callers may still send `width` and `height`; V2.1 ignores them and
normalizes every request to 1024 × 1024.

```text
Prompt / generation effort
  -> PN-like sensory channels (64)
  -> KC-like sparse expansion (1024, fan-in=6, ~5% active)
  -> MBON-like backend competition
  -> selected image organ only (lazy activation)
  -> artifact verification
  -> DAN-like reward modulation
  -> KC->MBON local plasticity
```

## What was removed in V2.1

The connectome no longer receives aspect-ratio or pixel-count routing signals.
Variable image-size normalization was also removed. The retained variable
controls are prompt, steps, guidance, seed, and optional debug backend override.

## Backends

- A1111 / Forge loopback API
- Diffusers
- SSD-1B distilled student model

## Run

Windows:

```bat
RUN_FMG_IMAGE_GENERATOR.cmd
```

Linux:

```bash
./RUN_FMG_IMAGE_GENERATOR.sh
```

UI/API:

`http://127.0.0.1:18765/`

## API request

```json
{
  "prompt": "cinematic photograph of a beagle running beside a lake at sunset",
  "steps": 50,
  "guidance": 9,
  "seed": -1,
  "backend": "connectome"
}
```

The output resolution is always 1024 × 1024.

## Tests

```bash
python -m unittest discover -s tests -v
```

Model weights, runtime artifacts, caches, and generated images are excluded from Git.


## V2.2 quality reflex

FMG now runs a compact local quality loop after 1024×1024 generation:

```text
generate
  -> MBON::evaluate
  -> good enough -> MBON::accept
  -> weak local region -> MBON::repair
  -> A1111 / Forge masked img2img
  -> re-evaluate
  -> keep repair only when technical score improves
```

The evaluator stays lightweight: entropy, edge/detail energy, contrast,
clipping, and exact 1024×1024 compliance. It does not load a second large
vision model and does not claim semantic judgment of faces, hands, text, or
prompt alignment.

Repair is conservative: up to three weak local regions, blurred masks,
denoising strength 0.28, and automatic rejection when the repaired image does
not improve the technical quality score.


## V2.3 — FCA-Vision-Lite

FMG now uses a reduced FCA as its post-generation visual decision controller.

Kept from FCA:
- 128 sensory channels
- 256 KC-like units
- top-16 competition
- fan-in 6
- temporal trace
- MBON action competition
- reward-prediction-error learning
- lazy specialist organs
- fail-closed evidence handling

Removed for FMG:
- conversation
- general world model
- repository tooling
- general autonomy loop
- unrelated memory/retrieval and skill machinery

FCA-Vision-Lite selects one of three actions:

```text
accept
local_repair
global_repair
```

Optional lazy specialists are face (OpenCV), hand (MediaPipe), and text
(Tesseract). If a specialist is unavailable, it does not create negative
evidence; the controller falls back to the technical quality evaluator.

The default launch scripts now start `fmg_image_generator_v23.py`.


## V2.4 — 2,000,000 FCA-Vision-Lite rehearsal events

A deterministic bootstrap policy is bundled in `data/fca_vision_bootstrap_v24.json`.

These are **2,000,000 sequential FCA routing/reward updates**, not two million diffusion image generations.

- distinct training patterns: 1,607
- distinct holdout patterns: 353
- holdout routing accuracy before: 65.16%
- holdout routing accuracy after: 71.67%
- online rehearsal accuracy: 77.02%
- seed: 2401

The bootstrap is loaded only when no local runtime state exists. Real generated images continue to update the runtime state after bootstrap.

## V2.5 — strict prompt/objective evaluation loop

FMG now includes a fail-closed evaluation loop for improving prompt adherence rather than accepting images on technical quality alone.

The bundled suite is `data/fmg_prompt_suite_v25.jsonl`:

- 316 total prompt cases
- 253 training cases
- 63 holdout cases
- categories include attribute binding, counting, spatial relations, exact text, portraits, hands, materials, lighting, camera geometry, motion, food, architecture, and macro detail

Each case carries contrastive negative descriptions. The strict evaluator compares the generated image against the requested prompt and competing wrong descriptions. PASS requires all available gates to clear:

```text
1024x1024 + technical quality
  -> prompt/image contrastive alignment
  -> positive-vs-wrong-description margin
  -> required specialist evidence (face / hand / exact text)
  -> PASS / FAIL / UNKNOWN
```

Missing semantic evidence is `UNKNOWN`, never an automatic PASS.

Training cases may be regenerated with a refined prompt derived from the previous failure reasons. Holdout cases are evaluated once and are not prompt-refined, so they remain useful as unseen checks.

Continuous Windows run:

```bat
RUN_FMG_EVAL_LOOP.cmd
```

Continuous Linux run:

```bash
./RUN_FMG_EVAL_LOOP.sh
```

The launchers run the training split continuously, publish generated image+JSON pairs under `generated/fmg_eval/images/`, and push batches of 32 artifacts to the currently checked-out Git branch.

Local verbose evaluation logs stay under `runtime/fmg_eval/logs/`. A small tracked file, `generated/fmg_eval/LOG_USAGE.json`, mirrors cumulative evaluation-log bytes so external monitoring can detect successive 50 GB thresholds without committing the full logs.

For a one-shot untouched evaluation:

```bash
python run_fmg_eval_loop.py --split holdout --limit 63
```

The holdout result should be reported separately from training results. Do not mix regenerated training attempts into the holdout score.

### Long-running watchdog and tracked metrics

For unattended operation that restarts the evaluator after a process/backend/push failure:

Windows:

```bat
WATCH_FMG_EVAL_LOOP.cmd
```

Linux:

```bash
./WATCH_FMG_EVAL_LOOP.sh
```

The evaluator also mirrors compact aggregate statistics to
`generated/fmg_eval/METRICS.json`. This tracks per-category PASS/FAIL/UNKNOWN
counts, repeated failure reasons, attempts, and best observed score per prompt
case. Full verbose logs remain local under `runtime/`; only compact usage and
metrics metadata are tracked with generated image/JSON pairs.

