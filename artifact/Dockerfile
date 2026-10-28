# Lightweight build/run image for the MCCL artifact (SimAI ns-3 backend, CPU only).
#
#   docker build -f artifact/Dockerfile -t mccl-simai .
#   docker run --rm -it -v "$PWD/artifact/results:/workspace/SimAI/artifact/results" mccl-simai
#   # inside the container:
#   artifact/run.sh smoke && python3 artifact/summarize.py artifact/results/smoke-*
#
# Build context must be a checkout with submodules initialized
# (git submodule update --init --recursive) or the extracted Zenodo archive.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake libxml2-dev python3 python3-matplotlib git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/SimAI
COPY . .

# Default build profile, same as the paper's runs. JOBS controls build parallelism.
ARG JOBS=""
RUN bash scripts/build.sh -c ns3 ${JOBS:+-j $JOBS} && test -x bin/SimAI_simulator

CMD ["/bin/bash"]
