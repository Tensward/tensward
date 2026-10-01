# Roadmap

What we intend to build, in order. Dates are deliberately absent; each item ships when it has
been validated on real hardware, not before. Issues and pull requests that move any of these
forward are welcome.

1. **SGLang engine.** A second implementation of the `Engine` interface
   (`src/tensward/engines/protocol.py`), with its own flags, metrics and
   launch command, so the same project can be analysed on vLLM and on SGLang.
2. **llama.cpp engine and GGUF checkpoints.** Registration of GGUF files and a llama.cpp server
   engine. This is also the path to Apple Silicon.
3. **More NVIDIA GPUs validated.** Today the full analysis has been run on an L4 and an A10G.
   The ceiling table already lists more GPUs from datasheets; each one still needs a real run.
4. **Apple Silicon.** Analysis on Macs, through the llama.cpp engine.
5. **Multi-GPU on one host.** Tensor and pipeline parallelism: registration, ceilings and
   profiling across several GPUs of one machine.
6. **Multi-node clusters.** Last, because it depends on everything above.
