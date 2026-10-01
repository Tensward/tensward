# Roadmap

What we intend to build, in order. Dates are deliberately absent; each item ships when it has
been validated on real hardware, not before. Issues and pull requests that move any of these
forward are welcome.

1. **Image workloads.** Prompts with images (OpenAI chat image parts pointing at local files),
   image tokens in the measurements and fit, and separate figures for requests with and
   without images. Image+text checkpoints already register and run text workloads.
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
7. **Audio and video inputs.** For models that take them, after image workloads.
8. **Multi-node clusters.** Last, because it depends on everything above.
