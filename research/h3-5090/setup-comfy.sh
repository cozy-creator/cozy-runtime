#!/usr/bin/env bash
# Run on the owned benchmark rental after extracting ComfyUI v0.39.0 here.
set -euo pipefail
cd /root/h3-bench
uv venv --python /usr/local/bin/python comfy-env
UV_CACHE_DIR=/opt/cozy/dependency-seed/uv-cache uv pip install \
  --python comfy-env/bin/python -r comfy/requirements.txt \
  'torch==2.14.0' 'torchvision==0.29.0' websocket-client \
  --extra-index-url https://download.pytorch.org/whl/cu130
uv pip freeze --python comfy-env/bin/python > comfy-environment.txt

revision=e5eb578a89295337b8ff433a035929ce0279e0b6
for model in \
  diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors \
  text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
  vae/minimax_h3_video_vae_fp16.safetensors \
  vae/minimax_h3_audio_vae_fp32.safetensors; do
  destination="comfy/models/$model"
  mkdir -p "$(dirname "$destination")"
  if [ ! -f "$destination" ]; then
    curl --fail --location --retry 3 --continue-at - \
      "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/$revision/$model" \
      --output "$destination.partial"
    mv "$destination.partial" "$destination"
  fi
  sha256sum "$destination" >> comfy-models.sha256
done
