# vLLM, Tensward and Nsight Compute in one environment: what `analyse --counters` needs.
# Verified on an L4 and an A10G. Tensward is built from this checkout, so the build context is the repo root:
#   docker build -f examples/counters/Dockerfile -t tensward-counters .
FROM vllm/vllm-openai:v0.30.0

# Nsight Compute CLI 2025.3.1 (the first release that supports CUDA 13) from NVIDIA's CUDA apt
# repository for Ubuntu 24.04. The SHA-256 is from that repository's Packages index; the build
# stops if the download differs.
ARG NCU_SHA256=e2d393e1a0ba7ae1c983b1062f6494a8e85a98c987503f71bb73cc615aafe512
RUN curl -fsSLo /tmp/ncu.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/nsight-compute-2025.3.1_2025.3.1.4-1_amd64.deb \
    && echo "${NCU_SHA256}  /tmp/ncu.deb" | sha256sum -c - \
    && dpkg -i /tmp/ncu.deb \
    && ln -s /opt/nvidia/nsight-compute/2025.3.1/ncu /usr/local/bin/ncu \
    && rm /tmp/ncu.deb

COPY pyproject.toml README.md LICENSE NOTICE /tmp/tensward/
COPY src /tmp/tensward/src
RUN pip install /tmp/tensward && rm -rf /tmp/tensward

ENTRYPOINT []
