<!-- Provenance. This is the genuine, unedited excerpt of `answers.md` of the run, apart from this header and one scrub: the absolute home path `/home/ubuntu/` was replaced by `~/`. The text was also checked for AWS instance ids, IP addresses, account ids, hostnames, tokens and customer names; none appeared, so nothing else was changed. -->

# Case 4, prefix caching: answers excerpt

How this was produced (2026-10-01):

- run id: `20261001T210738Z-cfc9 (baseline) vs 20261001T211247Z-60a9 (prefix caching)`
- GPU: NVIDIA A10G (24 GB), AWS g5.xlarge (driver 595.91.07)
- engine image: `vllm/vllm-openai:v0.30.0`, run by Tensward with `--runtime docker`
- Tensward version: 0.1.6 (release validation run)
- model: Qwen2.5-7B-Instruct-AWQ
- inputs: [`examples/config.json`](../../config.json) and [`examples/prompts.jsonl`](../../prompts.jsonl), unchanged (160 requests over 10 prompts, 32 concurrent clients, temperature 0, seed 0)
- `--current` command: `vllm serve /m/qwen --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes`
- change: `--engine-arg enable-prefix-caching`

This excerpt keeps the heading and the first prompt (chat-01). The other nine prompts were also reported identical (agreement 1.00 for each, per `compare.json`).

The text below the line is unedited apart from this header and the path scrub. Numbers from one run on one machine are an example, not a promise.

---

# Answers: 20261001T210738Z-cfc9 (your current setup) and run 20261001T211247Z-60a9

Answers unchanged within noise

## chat-01: agreement 1.00, the baseline's own 1.00

```
system: You are a helpful assistant. Answer clearly and concisely.
user: Explain the difference between a process and a thread in two short paragraphs.
```

### Your current setup

```
A process is a running instance of a program, which includes the program code, data, and the execution context. Each process has its own memory space and resources, ensuring that processes are isolated from each other to prevent interference. Processes are managed by the operating system and can communicate with each other through inter-process communication mechanisms.

A thread, on the other hand, is a lightweight subunit of a process that can execute concurrently with other threads within the same process. Threads share the same memory space and resources as the process they belong to, which allows for efficient communication and data sharing between threads. While processes are separate entities with their own resources,
```

### This run

```
A process is a running instance of a program, which includes the program code, data, and the execution context. Each process has its own memory space and resources, ensuring that processes are isolated from each other to prevent interference. Processes are managed by the operating system and can communicate with each other through inter-process communication mechanisms.

A thread, on the other hand, is a lightweight subunit of a process that can execute concurrently with other threads within the same process. Threads share the same memory space and resources as the process they belong to, which allows for efficient communication and data sharing between threads. While processes are separate entities with their own resources,
```

