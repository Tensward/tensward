# Roadmap

What we intend to build, in order. Dates are deliberately absent; each item ships when it has
been validated on real hardware, not before. Issues and pull requests that move any of these
forward are welcome.

1. **Output quality in every before/after.** The same prompts with the same seed on both
   settings: how many answers stayed the same and how close the rest are, agreement with the
   reference answers a workload declares, and format health (tool calls, valid JSON, answers
   cut short). A faster setting is then shown together with what it did to the answers.
2. **SGLang engine.** A second implementation of the `Engine` interface
   (`src/tensward/engines/protocol.py`), with its own flags, metrics and
   launch command, so the same project can be analysed on vLLM and on SGLang.
3. **llama.cpp engine and GGUF checkpoints.** Registration of GGUF files and a llama.cpp server
   engine. This is also the path to Apple Silicon.
4. **More NVIDIA GPUs validated.** Today the full analysis has been run on an L4 and an A10G.
   The ceiling table already lists more GPUs from datasheets; each one still needs a real run.
5. **Apple Silicon.** Analysis on Macs, through the llama.cpp engine.
6. **Multi-GPU on one host.** Tensor and pipeline parallelism: registration, ceilings and
   profiling across several GPUs of one machine.
7. **Audio and video inputs.** For models that take them.
8. **Multi-node clusters.** Last, because it depends on everything above.
