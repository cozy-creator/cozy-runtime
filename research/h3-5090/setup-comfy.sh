#!/usr/bin/env bash
# Run on the owned benchmark rental after extracting ComfyUI v0.39.0 here.
set -euo pipefail
cd /root/h3-bench
if [ ! -f comfy/main.py ]; then
  echo 'Extract the ComfyUI source into /root/h3-bench/comfy before setup.' >&2
  exit 1
fi
if [ ! -d comfy-env ]; then
  uv venv --python /usr/local/bin/python comfy-env
fi
UV_CACHE_DIR=/opt/cozy/dependency-seed/uv-cache uv pip install \
  --python comfy-env/bin/python -r comfy/requirements.txt \
  'torch==2.14.0' 'torchvision==0.29.0' websocket-client 'huggingface_hub==1.33.0' \
  --extra-index-url https://download.pytorch.org/whl/cu130
uv pip freeze --python comfy-env/bin/python > comfy-environment.txt

comfy-env/bin/python -u - <<'PY'
import hashlib
from pathlib import Path
from huggingface_hub import hf_hub_download

models = [
    "diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
    "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors",
    "vae/minimax_h3_video_vae_fp16.safetensors",
    "vae/minimax_h3_audio_vae_fp32.safetensors",
]
with open("comfy-models.sha256", "w") as receipt:
    for model in models:
        print("Downloading", model, flush=True)
        path = Path(hf_hub_download(
            "Comfy-Org/MiniMax-H3", model,
            revision="e5eb578a89295337b8ff433a035929ce0279e0b6",
            local_dir="comfy/models",
        ))
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        print(digest, path, file=receipt, flush=True)
PY
