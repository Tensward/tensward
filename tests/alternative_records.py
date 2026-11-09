"""Alternative records and setups for the Beyond flags tests."""

RECORDS = r'''
[[alternative]]
name = "qwen38-27b-l4-agent-iq4-xs"
kind = "file"
label = "the IQ4_XS file"
engine = "llamacpp"
quant = "IQ4_XS"
repo = "unsloth/Qwen3.8-27B-GGUF"
revision = "0123456789abcdef0123456789abcdef01234567"
file = "Qwen3.8-27B-UD-IQ4_XS.gguf"
weight_gib = 14.1
overhead_gib = 0.6
[alternative.match]
models = ["Qwen3.8-27B"]
family = "qwen35"
params_b = [26.0, 29.0]
gpus = ["L4"]
workload = "agent"
concurrency = [2, 4]
engine = "llamacpp"
files = ["Q4_K_M"]
[alternative.measured]
model = "Qwen3.8-27B"
gpu = "NVIDIA L4"
runs = 3
same_sign = true
tpot_p50 = -9.0
tpot_p95 = -8.0
ttft_p50 = -5.0
req_s = 6.0
[alternative.measured.noise]
tpot_p50 = 2.0
tpot_p95 = 2.0
ttft_p50 = 1.0
req_s = 2.0
[alternative.answers]
identical = 40
of = 64
floor = 58
broken = 1
base_broken = 1

[[alternative]]
name = "qwen38-27b-l4-agent-vllm-int4"
kind = "engine"
label = "vLLM with the int4 checkpoint"
engine = "vllm"
quant = "int4"
repo = "example/Qwen3.8-27B-int4"
revision = "0123456789abcdef0123456789abcdef01234567"
folder = "Qwen3.8-27B-int4"
command = """vllm serve --model {checkpoint} --max-num-seqs {concurrency} --max-model-len 4096 \
--gpu-memory-utilization 0.95 --max-num-batched-tokens 512 --language-model-only \
--kv-cache-memory-bytes 2027414271 --enable-auto-tool-choice --tool-call-parser qwen3_xml"""
context = 4096
tool_parser = "qwen3_xml"
weight_gib = 16.5
overhead_gib = 3.0
[alternative.match]
models = ["Qwen3.8-27B"]
family = "qwen35"
params_b = [26.0, 29.0]
gpus = ["L4"]
workload = "agent"
concurrency = [2, 4]
engine = "llamacpp"
files = ["Q4_K_M"]
[alternative.measured]
model = "Qwen3.8-27B"
gpu = "NVIDIA L4"
runs = 3
same_sign = true
tpot_p50 = -17.0
ttft_p50 = -4.6
[alternative.measured.noise]
tpot_p50 = 3.0
ttft_p50 = 0.4
[alternative.answers]
identical = 12
of = 192
floor = 144
broken = 0
base_broken = 0

[[alternative]]
name = "lfm2-8b-a1b-l4-chat-vllm"
kind = "engine"
label = "vLLM with the original checkpoint"
engine = "vllm"
quant = "BF16"
repo = "LiquidAI/LFM2-8B-A1B"
revision = "0123456789abcdef0123456789abcdef01234567"
folder = "LFM2-8B-A1B"
command = "vllm serve {checkpoint} --max-model-len 4096 --max-num-seqs {concurrency}"
context = 4096
weight_gib = 15.5
overhead_gib = 3.0
[alternative.match]
models = ["LFM2-8B-A1B"]
family = "lfm2_moe"
params_b = [8.0, 9.0]
gpus = ["L4"]
workload = "chat"
concurrency = [1, 1]
engine = "llamacpp"
files = ["Q4_K_M"]
[alternative.measured]
model = "LFM2-8B-A1B"
gpu = "NVIDIA L4"
runs = 2
same_sign = true
tpot_p50 = -30.0
[alternative.measured.noise]
tpot_p50 = 4.0
[alternative.answers]
identical = 10
of = 64
floor = 60
broken = 0
base_broken = 0

[[alternative]]
name = "under-three-percent"
kind = "file"
label = "the Q4_0 file"
engine = "llamacpp"
quant = "Q4_0"
repo = "unsloth/Qwen3.8-27B-GGUF"
revision = "0123456789abcdef0123456789abcdef01234567"
file = "Qwen3.8-27B-Q4_0.gguf"
weight_gib = 14.1
overhead_gib = 0.6
[alternative.match]
models = ["Qwen3.8-27B"]
family = "qwen35"
params_b = [26.0, 29.0]
gpus = ["L4"]
workload = "agent"
concurrency = [2, 4]
engine = "llamacpp"
files = ["Q4_K_M"]
[alternative.measured]
model = "Qwen3.8-27B"
gpu = "NVIDIA L4"
runs = 3
same_sign = true
tpot_p50 = -2.9
[alternative.measured.noise]
tpot_p50 = 0.5
[alternative.answers]
identical = 40
of = 64
floor = 58
broken = 1
base_broken = 1

[[alternative]]
name = "within-noise"
kind = "file"
label = "the Q4_1 file"
engine = "llamacpp"
quant = "Q4_1"
repo = "unsloth/Qwen3.8-27B-GGUF"
revision = "0123456789abcdef0123456789abcdef01234567"
file = "Qwen3.8-27B-Q4_1.gguf"
weight_gib = 14.1
overhead_gib = 0.6
[alternative.match]
models = ["Qwen3.8-27B"]
family = "qwen35"
params_b = [26.0, 29.0]
gpus = ["L4"]
workload = "agent"
concurrency = [2, 4]
engine = "llamacpp"
files = ["Q4_K_M"]
[alternative.measured]
model = "Qwen3.8-27B"
gpu = "NVIDIA L4"
runs = 3
same_sign = true
tpot_p50 = -5.0
[alternative.measured.noise]
tpot_p50 = 3.0
[alternative.answers]
identical = 40
of = 64
floor = 58
broken = 1
base_broken = 1

[[alternative]]
name = "one-run"
kind = "file"
label = "the Q3_K_M file"
engine = "llamacpp"
quant = "Q3_K_M"
repo = "unsloth/Qwen3.8-27B-GGUF"
revision = "0123456789abcdef0123456789abcdef01234567"
file = "Qwen3.8-27B-Q3_K_M.gguf"
weight_gib = 14.1
overhead_gib = 0.6
[alternative.match]
models = ["Qwen3.8-27B"]
family = "qwen35"
params_b = [26.0, 29.0]
gpus = ["L4"]
workload = "agent"
concurrency = [2, 4]
engine = "llamacpp"
files = ["Q4_K_M"]
[alternative.measured]
model = "Qwen3.8-27B"
gpu = "NVIDIA L4"
runs = 1
same_sign = true
tpot_p50 = -20.0
[alternative.measured.noise]
tpot_p50 = 2.0
[alternative.answers]
identical = 40
of = 64
floor = 58
broken = 1
base_broken = 1

[[alternative]]
name = "mixed-sign"
kind = "file"
label = "the Q3_K_S file"
engine = "llamacpp"
quant = "Q3_K_S"
repo = "unsloth/Qwen3.8-27B-GGUF"
revision = "0123456789abcdef0123456789abcdef01234567"
file = "Qwen3.8-27B-Q3_K_S.gguf"
weight_gib = 14.1
overhead_gib = 0.6
[alternative.match]
models = ["Qwen3.8-27B"]
family = "qwen35"
params_b = [26.0, 29.0]
gpus = ["L4"]
workload = "agent"
concurrency = [2, 4]
engine = "llamacpp"
files = ["Q4_K_M"]
[alternative.measured]
model = "Qwen3.8-27B"
gpu = "NVIDIA L4"
runs = 3
same_sign = false
tpot_p50 = -20.0
[alternative.measured.noise]
tpot_p50 = 2.0
[alternative.answers]
identical = 40
of = 64
floor = 58
broken = 1
base_broken = 1

[[alternative]]
name = "more-broken"
kind = "file"
label = "the Q3_K_L file"
engine = "llamacpp"
quant = "Q3_K_L"
repo = "unsloth/Qwen3.8-27B-GGUF"
revision = "0123456789abcdef0123456789abcdef01234567"
file = "Qwen3.8-27B-Q3_K_L.gguf"
weight_gib = 14.1
overhead_gib = 0.6
[alternative.match]
models = ["Qwen3.8-27B"]
family = "qwen35"
params_b = [26.0, 29.0]
gpus = ["L4"]
workload = "agent"
concurrency = [2, 4]
engine = "llamacpp"
files = ["Q4_K_M"]
[alternative.measured]
model = "Qwen3.8-27B"
gpu = "NVIDIA L4"
runs = 3
same_sign = true
tpot_p50 = -20.0
[alternative.measured.noise]
tpot_p50 = 2.0
[alternative.answers]
identical = 40
of = 64
floor = 58
broken = 5
base_broken = 1

[[alternative]]
name = "wrong-direction"
kind = "file"
label = "the Q6_K file"
engine = "llamacpp"
quant = "Q6_K"
repo = "unsloth/Qwen3.8-27B-GGUF"
revision = "0123456789abcdef0123456789abcdef01234567"
file = "Qwen3.8-27B-Q6_K.gguf"
weight_gib = 14.1
overhead_gib = 0.6
[alternative.match]
models = ["Qwen3.8-27B"]
family = "qwen35"
params_b = [26.0, 29.0]
gpus = ["L4"]
workload = "agent"
concurrency = [2, 4]
engine = "llamacpp"
files = ["Q4_K_M"]
[alternative.measured]
model = "Qwen3.8-27B"
gpu = "NVIDIA L4"
runs = 3
same_sign = true
tpot_p50 = 20.0
[alternative.measured.noise]
tpot_p50 = 2.0
[alternative.answers]
identical = 40
of = 64
floor = 58
broken = 1
base_broken = 1

[[alternative]]
name = "placeholder"
kind = "file"
label = "the Q5_K_M file"
engine = "llamacpp"
quant = "Q5_K_M"
repo = "unsloth/Qwen3.8-27B-GGUF"
revision = "0123456789abcdef0123456789abcdef01234567"
file = "Qwen3.8-27B-Q5_K_M.gguf"
weight_gib = 14.1
overhead_gib = 0.6
[alternative.match]
models = ["Qwen3.8-27B"]
family = "qwen35"
params_b = [26.0, 29.0]
gpus = ["L4"]
workload = "agent"
concurrency = [2, 4]
engine = "llamacpp"
files = ["Q4_K_M"]
[alternative.measured]
model = "Qwen3.8-27B"
gpu = "NVIDIA L4"
runs = 3
same_sign = true
tpot_p50 = -40.0
[alternative.measured.noise]
tpot_p50 = "WAVE-ALT"
[alternative.answers]
identical = 40
of = 64
floor = 58
broken = 1
base_broken = 1
'''
# The 0.4.2 tester's run: Qwen3.8-27B UD-Q4_K_M, llama.cpp, one L4, agent workload, 3 clients.
TESTER = {
    "model.identity": "qwen3827b",
    "model.family": "qwen35",
    "model.params_b": 27.320697856,
    "model.ftype": "Q4_K_M",
    "gpu.model": "L4",
    "gpu.count": 1,
    "gpu.memory_gib": 22.494140625,
    "platform.name": "nvidia",
    "engine.name": "llamacpp",
    "workload.tools": True,
    "workload.clients": 3,
    "workload.prompt_tokens_max": 821,
    "workload.output_tokens_limit": 1280,
    "anatomy.kv_bytes_per_token": 65536,
}
LFM2 = TESTER | {
    "model.identity": "lfm28ba1b",
    "model.family": "lfm2moe",
    "model.params_b": 8.3,
    "workload.tools": False,
    "workload.clients": 1,
    "anatomy.kv_bytes_per_token": 12288,
}
