"""Submit one full H3 benchmark to an already running, otherwise idle ComfyUI."""

import argparse
import json
import time
import urllib.request
import uuid
from pathlib import Path

import websocket


def node(kind, **inputs):
    return {"class_type": kind, "inputs": inputs}


def graph(payload, sigmas, label, sparse, world_size=1):
    if world_size > 1 and sparse:
        raise ValueError("The pinned multi-GPU Comfy node requires --dense")
    model_node = "7" if sparse else ("6" if world_size == 1 else "5")
    nodes = {
        "1": node("UNETLoader", unet_name="minimax_h3_fl2va_pruned_fp8_scaled.safetensors", weight_dtype="default"),
        "2": node("CLIPLoader", clip_name="qwen3vl_32b_minimax_h3_bf16.safetensors", type="minimax", device="default"),
        "3": node("VAELoader", vae_name="minimax_h3_video_vae_fp16.safetensors"),
        "4": node("VAELoader", vae_name="minimax_h3_audio_vae_fp32.safetensors"),
        "5": node("MiniMaxH3SigmaShift", model=["1", 0], shift_video=12.0, shift_audio=3.0),
        "8": node("MiniMaxH3ImageToVideo", clip=["2", 0], vae=["3", 0], prompt=payload["prompt"], width=1344, height=768, length=362),
        "9": node("BasicGuider", model=[model_node, 0], conditioning=["8", 0]),
        "10": node("RandomNoise", noise_seed=payload["seed"]),
        "11": node("KSamplerSelect", sampler_name="euler"),
        "12": node("ManualSigmas", sigmas=", ".join(format(s, ".10f") for s in sigmas)),
        "13": node("SamplerCustomAdvanced", noise=["10", 0], guider=["9", 0], sampler=["11", 0], sigmas=["12", 0], latent_image=["8", 1]),
        "14": node("VAEDecode", samples=["13", 0], vae=["3", 0]),
        "15": node("VAEDecodeAudio", samples=["13", 0], vae=["4", 0]),
        "16": node("CreateVideo", images=["14", 0], audio=["15", 0], fps=24.0, bit_depth=8),
        "17": node("SaveVideo", video=["16", 0], filename_prefix="h3-bench/" + label, format="mp4", **{"format.codec": "h264"}),
    }
    if world_size == 1:
        nodes["6"] = node("ModelAttentionBackend", model=["5", 0], attention="comfy kitchen attention")
    else:
        nodes["1"] = node("MiniMaxH3SPUNETLoader", unet_name="minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
                          weight_dtype="default", world_size=world_size, devices="auto")
    if sparse:
        # Put the threshold between evaluations 9 and 10: exactly ten dense evaluations.
        threshold = (sigmas[9] + sigmas[10]) / 2
        start = 1 - threshold / (12 - 11 * threshold)
        nodes["7"] = node("BlockSparseAttention", model=["6", 0], selection="sol-attn", **{
            "selection.tau": 1.0, "start_percent": start, "end_percent": 1.0,
            "dense_blocks": "0,1", "min_tokens": 0, "extra_tokens": 0,
            "sink_conditioning": "exact_kv_and_rows", "verbose": True,
        })
    return nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("sigmas", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--world-size", type=int, choices=(1, 2, 4, 7, 8), default=1)
    parser.add_argument("--port", type=int, default=8188)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    assert payload["duration_s"] == 15 and payload["steps"] == 30
    sigmas = json.loads(args.sigmas.read_text())
    assert len(sigmas) == 31
    prompt = graph(payload, sigmas, args.output.name, not args.dense, args.world_size)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "prompt.json").write_text(json.dumps(prompt, indent=2))
    client = str(uuid.uuid4())
    address = f"127.0.0.1:{args.port}"
    ws = websocket.create_connection(f"ws://{address}/ws?clientId={client}", timeout=60)
    started = time.monotonic()
    request = urllib.request.Request(
        f"http://{address}/prompt", json.dumps({"prompt": prompt, "client_id": client}).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        accepted = json.load(response)
    (args.output / "accepted.json").write_text(json.dumps(accepted, indent=2))
    prompt_id = accepted["prompt_id"]
    with (args.output / "events.jsonl").open("w") as log:
        while True:
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue  # The observer never cancels a long-running render.
            if not isinstance(raw, str):
                continue
            event = json.loads(raw)
            row = {"unix_s": time.time(), "elapsed_s": time.monotonic() - started, "event": event}
            print(json.dumps(row), file=log, flush=True)
            data = event.get("data", {})
            if data.get("prompt_id") != prompt_id:
                continue
            if event["type"] in ("execution_error", "execution_interrupted"):
                raise RuntimeError(data)
            if event["type"] == "executing" and data.get("node") is None:
                break
    with urllib.request.urlopen(f"http://{address}/history/{prompt_id}") as response:
        history = json.load(response)
    elapsed = time.monotonic() - started
    (args.output / "history.json").write_text(json.dumps(history, indent=2))
    (args.output / "timing.json").write_text(json.dumps({"wall_s": elapsed, "prompt_id": prompt_id}, indent=2))
    ws.close()
    print(json.dumps({"wall_s": elapsed, "prompt_id": prompt_id}), flush=True)


if __name__ == "__main__":
    main()
