# Roadmap

What we intend to build, in order. Dates are deliberately absent; each item ships when it has
been validated on real hardware, not before. Issues and pull requests that move any of these
forward are welcome.

1. **llama.cpp engine and GGUF checkpoints.** Registration of GGUF files and a llama.cpp server
   engine.
2. **Apple Silicon.** Analysis on Macs, with llama.cpp (Metal) and MLX as a new engine.
3. **SGLang engine.** A second implementation of the `Engine` interface
   (`src/tensward/engines/protocol.py`), with its own flags, metrics and
   launch command, so the same project can be analysed on vLLM and on SGLang.
4. **Production workload fidelity.** What real serving traffic carries:
   - per-request structured output (grammars, JSON schemas);
   - LoRA adapters as the request's model;
   - token-id prompts;
   - per-request generation limits;
   - a batch "drain N requests" workload that reports time to drain and a unit-of-work rate.
5. **Suggestions with memory.** Remember what was already tried and measured, and propose combinations of changes that each helped.
6. **Two models on one GPU.** A fit check for models that must stay resident together.
7. **More NVIDIA GPUs validated.** Today the full analysis has been run on an L4 and an A10G.
   The ceiling table already lists more GPUs from datasheets; each one still needs a real run.
8. **Multi-GPU on one host.** Tensor and pipeline parallelism: registration, ceilings and
   profiling across several GPUs of one machine.
9. **Audio and video inputs.** For models that take them.
10. **Multi-node clusters.** Last, because it depends on everything above.

Engines, checkpoint formats and hardware platforms are now separate parts of the code
(`engines/`, `formats/`, `platforms/`), so each of the first three items is an addition: a new
engine, a new format, a new platform.
