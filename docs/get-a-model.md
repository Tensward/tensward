# Get a model

Tensward never downloads models. It tunes a model you already have and serve. If you have none,
this page downloads one with the Hugging Face CLI. The commands are the same on Linux and macOS.

## Install the CLI

```sh
uv tool install huggingface_hub
hf --help
```

`pipx install huggingface_hub` works too. Open a new shell if `hf` is not found.

Qwen3-8B is not gated, so no login is needed. For a gated model, run `hf auth login` first.

## For vLLM: safetensors (Linux with an NVIDIA GPU)

```sh
hf download Qwen/Qwen3-8B --local-dir ~/models/Qwen3-8B
```

This is about 16 GB (bf16). Serve it with vLLM, pointing at the folder:

```sh
vllm serve ~/models/Qwen3-8B
```

vLLM does not run on a Mac GPU. On a Mac, use the GGUF form below.

## For llama.cpp: GGUF (Linux or macOS)

Download one quantized file, not the whole repository:

```sh
hf download Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf --local-dir ~/models/Qwen3-8B-GGUF
```

This is about 5 GB. Serve it with:

```sh
llama-server -m ~/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf
```

## Where files go

`--local-dir` puts the files in a folder you choose, which is the path you give to the engine and
to `tensward init`. Without it, `hf` stores files in `~/.cache/huggingface/hub`.

If a download stops, run the same command again; it resumes.
