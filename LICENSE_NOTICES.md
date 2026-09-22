# Model / License Notes

FMG source code in this repository is the FMG connectome image-generator implementation.

Default external model:
- Repository: segmind/SSD-1B
- Upstream model card reports Apache-2.0
- A1111 checkpoint: SSD-1B-A1111.safetensors
- Expected SHA-256: 1895a00bfc769a00b0c0c43a95e433e79e9db8a85402b45a33e8448785bde94d

Model weights are not stored in this Git repository.

The connectome controller is a newly implemented engineering abstraction inspired by fly mushroom-body routing. It does not redistribute FlyWire/FAFB connectome datasets and is not presented as a neuron-by-neuron biological reconstruction.
