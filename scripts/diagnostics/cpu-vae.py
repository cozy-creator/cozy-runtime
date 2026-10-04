"""One full CPU VAE decode through existing Runtime CPU weight bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import resource
import time
from types import SimpleNamespace

MIB = 1024 * 1024
GIB = 1024 * MIB
MANIFEST = "sha256:288440e7dc660d047b23dc72efee3d9ff4d4222a35e45b4bd640848e50bee642"
LATENT_SHA = "87cad3ed70cfae838271022697730e7d3c627d75ef6979b4d8e8c2e4aab72899"


def available_host() -> int:
    info = pathlib.Path("/proc/meminfo").read_text()
    return (
        int(next(line.split()[1] for line in info.splitlines() if line.startswith("MemAvailable:")))
        * 1024
    )


def main() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU proof must hide CUDA"
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent", type=pathlib.Path, required=True)
    parser.add_argument("--store", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    raw = args.latent.read_bytes()
    assert len(raw) == 131072 and hashlib.sha256(raw).hexdigest() == LATENT_SHA
    allowance = 16 * GIB
    before = available_host()
    assert before >= allowance + 2 * GIB, "CPU host allowance refused before construction"

    import importlib.metadata
    import inspect

    import msgspec
    import tensorfs
    import torch
    from diffusers import AutoencoderKL

    from cozy_runtime.author._codec import encode_webp
    from cozy_runtime.author._loader import census
    from cozy_runtime.internal.derive import Observations, serving_substrate
    from cozy_runtime.internal.fill import Checkpoint
    from cozy_runtime.internal.weights import PlaneBackend, WeightResidency, Weights

    torch.set_num_threads(2)
    assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    book = Checkpoint(args.store, MANIFEST)
    config = json.loads(book.header["configs"]["vae"])
    rows = book.rows("vae")
    assert len(rows) == 248
    weights = Weights(torch, torch.device("cpu"), "cpu")
    weights.set_budget(-1, pinned=GIB)
    backend = PlaneBackend.for_script(
        {"vae": book},
        rows,
        weights=weights,
        construction="native-cpu-vae",
        release=importlib.metadata.version("cozy-runtime"),
        encoded_leaves="accept",
    )
    result = {
        "proof_level": "CPU_COMPONENT_DECODE_AND_NATIVE_WEIGHT_BINDINGS",
        "manifest": MANIFEST,
        "header_sha256": hashlib.sha256(book.header_bytes).hexdigest(),
        "latent_sha256": LATENT_SHA,
        "latent_bytes": len(raw),
        "original_seed": 1114002,
        "source_weight_rows": len(rows),
        "source_weight_bytes": sum(r.nbytes for r in rows),
        "original_vae_config": config,
        "host_available_before": before,
        "declared_host_allowance_bytes": allowance,
        "host_admission_limit": "Snapshot admission; not a machine host-budget reservation",
        "runtime": importlib.metadata.version("cozy-runtime"),
        "tensorfs": importlib.metadata.version("tensorfs"),
        "tensorfs_store_feature": tensorfs.STORE,
        "plane_limit": "Existing public Store-enabled0.3.93 CPU plane; not f089 STORE=False image proof",
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "diffusers": importlib.metadata.version("diffusers"),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "dtype": "bfloat16",
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "source_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        "weights_source_sha256": hashlib.sha256(
            pathlib.Path(inspect.getfile(Weights)).read_bytes()
        ).hexdigest(),
        "decoder_inputs": [],
        "vae_decode_calls": 0,
        "decoder_forward_calls": 0,
        "module_copy": False,
        "gpu_module_or_weight_copy": False,
        "whole_forward_retry": False,
    }

    def save() -> None:
        with (args.out / "result.json").open("w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

    try:
        # Construct ONE original module on meta, then use the existing CPU plane.
        with serving_substrate("cpu", Observations()):
            vae = AutoencoderKL.from_config(config).to(torch.float16)
        module_id = id(vae)
        holder = SimpleNamespace(components={"vae": vae})
        walked = census(holder)
        verdict = backend.fit(walked, encoded_leaves="accept")
        assert verdict is None or verdict.get("ok"), verdict
        backend.expect({"vae": [d.key for d in walked.destinations]})
        live = backend.materialize(holder, walked)
        for destination in walked.destinations:
            backend.fill(destination.key, destination.spec, live[destination.key])
        backend.commit()
        residency = WeightResidency(weights, backend.components, {"decode": ("vae",)})
        assert weights.host_only and not residency.hooks
        assert all(p.device.type == "cpu" for p in vae.parameters())
        # The original GPU diagnostic upcasts f16 weights/latents to BF16 before scaling.
        # This converts dtype in the same module; no GPU module is copied or moved.
        vae.to(dtype=torch.bfloat16)
        assert id(vae) == module_id
        assert all(p.device.type == "cpu" and p.dtype == torch.bfloat16 for p in vae.parameters())
        vae.disable_tiling()
        vae.disable_slicing()
        latent = torch.frombuffer(bytearray(raw), dtype=torch.float16).reshape(1, 4, 128, 128)
        latent = latent.to(dtype=torch.bfloat16)
        result["backend_before_decode"] = backend.stats
        result["host_available_before_forward"] = available_host()
        assert result["host_available_before_forward"] >= allowance
        result["admission_before_forward"] = True
        result["cpu_weight_binding_ready"] = backend.state == "ready"
        result["tiling"] = vae.use_tiling
        result["slicing"] = vae.use_slicing
        result["cuda_initialized_before_forward"] = torch.cuda.is_initialized()
        save()

        def observe_decoder(_module: object, values: tuple[object, ...]) -> None:
            result["decoder_forward_calls"] += 1
            value = values[0]
            result["decoder_inputs"].append(
                {
                    "shape": list(value.shape),
                    "device": str(value.device),
                    "dtype": str(value.dtype),
                }
            )

        observer = vae.decoder.register_forward_pre_hook(observe_decoder)
        residency.admit("decode", ("vae",))
        try:
            start = time.perf_counter_ns()
            result["vae_decode_calls"] += 1
            with torch.inference_mode():
                decoded = vae.decode(latent / float(config["scaling_factor"])).sample
            result["decode_ns"] = time.perf_counter_ns() - start
        finally:
            observer.remove()
            residency.release("decode", ("vae",))
        assert result["vae_decode_calls"] == 1 and result["decoder_forward_calls"] == 1
        assert decoded.shape == (1, 3, 1024, 1024) and decoded.dtype == torch.bfloat16
        assert bool(torch.isfinite(decoded).all().item())
        pixels = ((decoded / 2 + 0.5).clamp(0, 1)[0] * 255).to(torch.uint8)
        rgb = pixels.permute(1, 2, 0).contiguous().numpy().tobytes()
        image = encode_webp(1024, 1024, rgb)
        (args.out / "cpu-native-bf16.webp").write_bytes(image)
        result["image_sha256"] = hashlib.sha256(image).hexdigest()
        result["image_bytes"] = len(image)
        result["decoded_shape"] = list(decoded.shape)
        result["decoded_dtype"] = str(decoded.dtype)
        result["decoded_device"] = str(decoded.device)
        result["rss_high_water_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        result["weight_plane_facts"] = msgspec.to_builtins(weights.facts())
        result["cuda_initialized_after_decode"] = torch.cuda.is_initialized()
        assert not result["cuda_initialized_after_decode"]
        result["outcome"] = "SUCCEEDED"
    except Exception as error:
        result["outcome"] = "FAILED"
        result["error"] = repr(error)
        raise
    finally:
        backend.close()
        result["backend_closed"] = backend.state != "ready" or not backend.components
        save()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
