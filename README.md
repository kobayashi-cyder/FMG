# FMG — Fly Media Generator

FMG image-generation core using a fly-connectome-inspired sparse controller.

## Current version
FMG Image Generator Connectome V2

```text
Prompt / parameters
  -> PN-like sensory channels (64)
  -> KC-like sparse expansion (1024, fan-in=6, ~5% active)
  -> MBON-like action competition
  -> selected image organ only (lazy activation)
  -> artifact verification
  -> DAN-like reward modulation
  -> KC->MBON local plasticity
```

The controller is a connectome-inspired engineering abstraction, not a literal neuron-by-neuron biological reconstruction.

## Image backends
- A1111 / Forge loopback API
- Diffusers
- Default distilled student model: `segmind/SSD-1B`
- Qwen: not used
- Cloud API: not required after local model preparation

## Run
Windows:
```bat
RUN_FMG_IMAGE_GENERATOR.cmd
```

Linux:
```bash
./RUN_FMG_IMAGE_GENERATOR.sh
```

UI/API default:
`http://127.0.0.1:18765/`

## Tests
```bash
python -m unittest discover -s tests -v
```

## Repository policy
Model weights, runtime artifacts, caches, and generated images are excluded from Git.
