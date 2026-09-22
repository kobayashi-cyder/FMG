# FMG — Fly Media Generator

Current image core: **FMG Image Generator Connectome V2.1**

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
