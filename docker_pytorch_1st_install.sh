#!/bin/bash

# --- 1. Path Configuration ---
ZOTERO_PATH="${HOME}/Zotero"
DATABASE_PATH="${HOME}/db_zotero_rag"
GLOBAL_HF_CACHE="${HOME}/.cache/huggingface"

# --- 2. Pre-Flight Permission Fix ---
# Ensure directories exist and are owned by you BEFORE Docker touches them
mkdir -p "$DATABASE_PATH"
mkdir -p "$GLOBAL_HF_CACHE"
sudo chown -R $(id -u):$(id -g) "$DATABASE_PATH" "$GLOBAL_HF_CACHE"

# --- 3. Docker Run ---
docker run --gpus all -it --rm \
  --ipc=host \
  --network host \
  --cap-add=SYS_ADMIN \
  --user "$(id -u):$(id -g)" \
  -v "${ZOTERO_PATH}":/workspace/data/Zotero \
  -v "${DATABASE_PATH}":/workspace/data/database \
  -v "${GLOBAL_HF_CACHE}":/workspace/data/hf_cache \
  -v "${PWD}":/workspace \
  -e HF_HOME=/workspace/data/hf_cache \
  -e XDG_CACHE_HOME=/workspace/data/hf_cache \
  \
  -w /workspace \
  nvcr.io/nvidia/pytorch:25.11-py3
