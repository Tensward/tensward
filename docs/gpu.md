# GPU and engines

Tensward does not install a serving engine as a Python dependency. The engine runs as a separate
program, pinned by its container image (`--runtime docker`) or by whatever server command the
machine provides (`--runtime local`). CI and `make check` need no GPU.

The optional `gpu` extra only adds `nvidia-ml-py`, which reads local GPU information via NVML
(nothing is sent anywhere):

```sh
uv sync --locked --extra gpu
```

## Engines

Tensward talks to a serving engine through the small `Engine` interface in
`tensward/engines/protocol.py`. vLLM (`tensward/engines/vllm.py`) is the first implementation and
the only one so far. It owns everything vLLM-specific: the flags that engine-neutral `Settings`
map to, the default image (`vllm/vllm-openai:v0.30.0`, pinned), the `VLLM_API_KEY` variable, the health,
models, metrics and tokenize endpoints, and the Prometheus metric names read into engine-neutral
signals (KV usage, running, waiting, preemptions, prefix-cache hits).

The API key is passed to the server in its environment, never on an argument vector. Docker images
are never pulled implicitly.
