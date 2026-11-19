#!/usr/bin/env bash
# Build llama.cpp with CUDA for a Colab T4 (Turing, sm_75).
#
# Only llama-server and llama-bench are built, and only for sm_75, which
# keeps the build to roughly 10-15 minutes on Colab's 2 CPU cores.
#
# Usage:  bash setup_llamacpp.sh [install_dir]        (default /content/llama.cpp)
#         LLAMA_CPP_REF=<commit or tag> bash setup_llamacpp.sh   to pin a version
set -euo pipefail

DIR="${1:-/content/llama.cpp}"
REF="${LLAMA_CPP_REF:-master}"
ARCH="${CUDA_ARCH:-75}"

if [ ! -d "$DIR/.git" ]; then
  git clone --filter=blob:none https://github.com/ggml-org/llama.cpp "$DIR"
fi
cd "$DIR"
git fetch --quiet origin "$REF"
git checkout --quiet FETCH_HEAD
echo "llama.cpp at $(git rev-parse --short HEAD): $(git log -1 --format=%s)"

# DSpark support landed in llama.cpp PR #27383 ("spec : support DSpark for LFM2 models").
if ! grep -rqi "dspark" common/ tools/ 2>/dev/null; then
  echo "WARNING: this llama.cpp checkout has no DSpark code. Speculative runs will fail." >&2
fi

cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF

cmake --build build --config Release -j "$(nproc)" --target llama-server llama-bench

ls -la build/bin/llama-server build/bin/llama-bench
if build/bin/llama-server --help 2>&1 | grep -q "draft-dspark"; then
  echo "OK: llama-server knows --spec-type draft-dspark"
else
  echo "WARNING: llama-server --help does not list draft-dspark" >&2
fi
