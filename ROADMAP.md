# Roadmap

What we intend to build, in order. Dates are deliberately absent; each item ships when it has
been validated on real hardware, not before. Issues and pull requests that move any of these
forward are welcome.

1. **llama.cpp engine and GGUF checkpoints, with CPU and offload.**
   - Registration of GGUF files and a llama.cpp server engine.
   - Analysis on CPU-only machines.
   - Analysis of models split between GPU and CPU: layers kept on the CPU, mixture-of-experts
     experts offloaded to system memory, and KV cache offload. The diagnosis says when the link
     between CPU and GPU memory is what holds a run back, and how much more to keep on the GPU.
2. **Apple Silicon.** Analysis on Macs, with llama.cpp (Metal) and MLX as a new engine.
3. **Edge devices.** On-device inference on phones and tablets, starting with Apple's (MLX and
   llama.cpp on iOS and iPadOS).
   - Measurement of a model running on the device, where memory, heat and power limit what fits
     and how fast it runs.
   - Advice on what to change for that device: model size, quantization, context length and
     offload.
4. **SGLang engine.** A second implementation of the `Engine` interface
   (`src/tensward/engines/protocol.py`), with its own flags, metrics and
   launch command, so the same project can be analysed on vLLM and on SGLang.
5. **Runtime analysis.** Analysis of a server while it handles its real traffic, alongside the
   runs Tensward launches itself.
   - Reading a running engine's metrics over a period, without restarting it.
   - Diagnoses that need real traffic, for example a prefix cache that is on but rarely reused
     because the shared part of the prompts changes (a timestamp or request id at the top of the
     system prompt, or tool definitions in a different order).
6. **Multi-GPU on one host.** Instances with several GPUs (for example 4 or 8 on one machine).
   - Tensor and pipeline parallelism: registration, ceilings and profiling across the GPUs.
   - A diagnosis for communication between GPUs (PCIe or NVLink).
   - Advice on how many GPUs to use and how to split the model across them.
7. **Production workload fidelity.** What real serving traffic carries:
   - per-request structured output (grammars, JSON schemas);
   - LoRA adapters as the request's model;
   - token-id prompts;
   - per-request generation limits;
   - a batch "drain N requests" workload that reports time to drain and a unit-of-work rate.
8. **Suggestions with memory.** Remember what was already tried and measured, and propose combinations of changes that each helped.
9. **Two models on one GPU.** A fit check for models that must stay resident together.
10. **More NVIDIA GPUs validated.** Today the full analysis has been run on an L4 and an A10G.
   The ceiling table already lists more GPUs from datasheets; each one still needs a real run.
11. **Audio and video inputs.** For models that take them.
12. **Multi-node clusters.** Last, because it depends on everything above.

Engines, checkpoint formats and hardware platforms are now separate parts of the code
(`engines/`, `formats/`, `platforms/`), so each of the first four items is an addition: a new
engine, a new format, a new platform.
