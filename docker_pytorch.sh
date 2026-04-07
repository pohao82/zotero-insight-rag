#!/bin/bash

# --- Path Configuration ---
# Use absolute paths from your Home directory for Zotero and Database
ZOTERO_PATH="${HOME}/Zotero"
DATABASE_PATH="${HOME}/db_zotero_rag"

# Define a GLOBAL Hugging Face cache on your host machine
# This prevents downloading the same 1.4GB+ models for every new project
GLOBAL_HF_CACHE="${HOME}/.cache/huggingface"

echo "Using Zotero Path: $ZOTERO_PATH"
echo "Using Database Path: $DATABASE_PATH"
echo "Using Global HF Cache: $GLOBAL_HF_CACHE"

# Ensure the global cache directory exists on the host
mkdir -p "$GLOBAL_HF_CACHE"

# --- 2. Docker Run ---
docker run --gpus all -it --rm \
  --name zotero_rag_engine \
  --ipc=host \
  --network host \
  --cap-add=SYS_ADMIN \
  --user "$(id -u):$(id -g)" \
  -v "${ZOTERO_PATH}":/workspace/data/Zotero \
  -v "${DATABASE_PATH}":/workspace/data/database \
  -v "${PWD}":/workspace \
  -v "${GLOBAL_HF_CACHE}":/workspace/data/hf_cache \
  -e HF_HOME=/workspace/data/hf_cache \
  -e XDG_CACHE_HOME=/workspace/data/hf_cache \
  -w /workspace \
  zotero-rag:v1
