# Docker & DGX Setup

For ingestion (marker-pdf parsing) on NVIDIA DGX or CUDA-enabled servers, use the PyTorch-optimized container:
1.  Run the initial setup: `sh docker_pytorch_1st_install.sh` (modified from the official playbook: [Fine-tune with Pytorch](https://build.nvidia.com/spark/pytorch-fine-tune/instructions)).
2.  Install libraries
    ```pip install marker-pdf duckdb langchain-ollama langchain-community pyyaml requests streamlit pypdf```
3.  To avoid having to reinstall libraries, commit the container by running the following outside of the docker environment:
    `docker commit <container_id> zotero-rag:v1`
    `<container_id>` can be found using `docker ps`
4.  After the initial setup, run `docker_pytorch.sh` for future use.

_Note:
The default image name is set to zotero-rag:v1. If you choose to rename it, ensure the new name is updated consistently within the docker_pytorch.sh script._

```
make sure the following empty folders exist insider data/ to avoid permssion issues
├── app/
├── data/ (project root)
     ├── hf_cache/
     ├── database/
     └── Zotero/
```

or Change ownership for all the folders need to be modifed from docker
```sudo chown -R $(id -u):$(id -g) ~/.cache/huggingface ~/db_zotero_rag $(pwd)```
