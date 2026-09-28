"""CUDA proof against the actual H3 package constructors and native TensorFS.

Run with H3's locked dependencies and this Runtime installed. This shrinks widths,
not layer count or inference code; it does not certify full-video activation memory.
The all-resident reference is test data, discarded before the paging path begins.
"""

import argparse
import copy
import gc
import importlib
import io
import json
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from tensorfs.derived import Config as NativeConfig
from tensorfs.derived import Derivation, Part, Target, Tensor

from cozy_runtime.internal.weights_sink import weights_transaction_id
from cozy_runtime.internal.worker.ledger import Ledger
from cozy_runtime.internal.worker.plan import (
    DeclaredBinding,
    PlanChooser,
    PreparedModel,
    PreparedRequest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("component", choices=["text_encoder", "fl2va_dit", "video_vae"])
    parser.add_argument("--terminal-arm", choices=["failure", "alias", "opaque"], default="failure")
    parser.add_argument("--terminal-cache", choices=["enabled", "disabled"], default="enabled")
    arguments = parser.parse_args()
    arguments.work_dir.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [
        str(arguments.packages_root / "minimax-h3"),
        str(arguments.packages_root / "minimax-h3-tools/src"),
    ]
    import tensorfs
    import torch

    build_text_conditioner = importlib.import_module("conditioner").build_text_conditioner
    config_module = importlib.import_module("h3_tables.model_config")
    dual_full_config = config_module.dual_full_config
    parse_production_config = config_module.parse_production_config

    from cozy_runtime.author._loader import census
    from cozy_runtime.internal.derive import Observations, serving_substrate
    from cozy_runtime.internal.encoding import SPEC_PLAIN
    from cozy_runtime.internal.fill import Checkpoint, StreamingFillBackend, _Holder
    from cozy_runtime.internal.paging import partition
    from cozy_runtime.internal.residency import ComponentResidency, ResidencyRefusal

    sections = parse_production_config(
        (
            arguments.packages_root / "minimax-h3-tools/src/h3_tables/assets/model-config.json"
        ).read_bytes()
    )
    component = arguments.component
    config = json.loads(dual_full_config(sections))
    torch.manual_seed(4321)
    if component == "text_encoder":
        config = config[component]
        config["text_config"].update(
            hidden_size=512,
            intermediate_size=1024,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=128,
            vocab_size=128,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        config["vision_config"].update(
            hidden_size=128,
            intermediate_size=256,
            out_hidden_size=512,
            num_heads=4,
            depth=2,
            deepstack_visual_indexes=[0, 1],
        )

        def factory() -> Any:
            return build_text_conditioner(copy.deepcopy(config))

        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device="cuda"),
            "attention_mask": torch.ones((1, 8), device="cuda", dtype=torch.long),
            "output_hidden_states": True,
            "use_cache": False,
            "return_dict": True,
        }

        def outputs(value: Any) -> tuple[Any, ...]:
            return (value.hidden_states[-1].cpu(),)
    elif component == "video_vae":
        from diffusers import AutoencoderKLMiniMaxH3
        from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3VideoDecodeStep
        from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline

        config = config[component]
        config.update(
            block_out_channels=[32, 32, 64, 64, 64, 128],
            decoder_num_attention_heads=4,
            decoder_attention_head_dim=32,
        )

        def factory() -> Any:
            return AutoencoderKLMiniMaxH3.from_config(copy.deepcopy(config)).eval()

        inputs = {"latents": torch.randn((1, 24, 7, 20, 20), device="cuda") * 0.1}

        def outputs(value: Any) -> tuple[Any, ...]:
            return (value.cpu(),)

        def invoke(model: Any) -> Any:
            pipe = MiniMaxH3ModularPipeline(blocks=MiniMaxH3VideoDecodeStep())
            pipe.update_components(vae=model)
            return pipe(latents=inputs["latents"], output_type="pt", output="videos")
    else:
        official = importlib.import_module("official")
        _build_dit, _dit_spec = official._build_dit, official._dit_spec

        config, structure, plan = _dit_spec(config, "fl2va")
        config.update(
            hidden_size=512,
            ffn_dim=1024,
            num_attention_heads=4,
            attention_head_dim=128,
            text_dim=512,
            time_embed_dim=256,
            time_embed_hidden_dim=512,
            freq_dim=128,
        )

        def factory() -> Any:
            return _build_dit(copy.deepcopy(config), structure, plan)

        inputs = {
            "hidden_states": torch.randn((1, 4, 96), device="cuda"),
            "audio_hidden_states": torch.randn((1, 2, 32), device="cuda"),
            "encoder_hidden_states": torch.randn((1, 3, 512), device="cuda"),
            "timestep": torch.tensor([0.3, 0.8], device="cuda"),
            "timestep_indices": torch.tensor([0, 0, 1, 1, 0, 0, 0, 1, 1], device="cuda"),
            "token_tags": torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2], device="cuda"),
            "position_ids": torch.arange(27, device="cuda").reshape(9, 3),
            "video_indices": torch.arange(4, device="cuda"),
            "text_indices": torch.arange(4, 7, device="cuda"),
            "audio_indices": torch.arange(7, 9, device="cuda"),
        }

        def outputs(value: Any) -> tuple[Any, ...]:
            return (value.sample.cpu(), value.audio_sample.cpu())

    if component != "video_vae":

        def invoke(model: Any) -> Any:
            return model(**inputs)

    reference = factory()
    with serving_substrate("sm{}{}".format(*torch.cuda.get_device_capability()), Observations()):
        model = factory()
    layout = partition(model)
    assert layout is not None
    print(json.dumps({"component": component, "layout": layout.document()}), flush=True)
    with tempfile.TemporaryDirectory(
        prefix="h3-block-paging-", dir=arguments.work_dir
    ) as directory:
        store_root = Path(directory) / "store"
        store = tensorfs.Store.ensure(str(store_root))
        targets = {}
        dtype = {"torch.bfloat16": "bf16", "torch.float32": "f32", "torch.int64": "i64"}
        state = reference.state_dict()
        for key, tensor in state.items():
            kind = dtype[str(tensor.dtype)]
            targets[key] = Tensor(
                logical_dtype=kind,
                shape=tuple(tensor.shape),
                encoding=SPEC_PLAIN,
                parts={"value": Part(dtype=kind, shape=tuple(tensor.shape))},
            )
        total = sum(t.numel() * t.element_size() for t in state.values())
        transaction = store.begin_derived(
            weights_transaction_id("paging-proof", "paging-proof", "sha256:" + "1" * 64, "model"),
            1,
            *Derivation(
                sources={},
                targets={component: Target(add=targets)},
                configs={"model": NativeConfig("add")},
                order=tuple((component, key) for key in state),
            ).native_arguments(total + 4096),
            work_fingerprint="sha256:" + "b" * 64,
        )
        for key, tensor in state.items():
            transaction.add_part(
                component,
                key,
                "value",
                io.BytesIO(tensor.contiguous().view(torch.uint8).numpy().tobytes()),
            )
        transaction.add_config("model", io.BytesIO(b'{"proof":"paging"}'))
        receipt = transaction.commit()
        manifest = "sha256:" + receipt["manifest"]["sha256"]
        print(json.dumps({"phase": "native_commit", "bytes": total}), flush=True)
        del state, transaction
        reference.to("cuda")
        with torch.no_grad():
            expected = outputs(invoke(reference))
        print(json.dumps({"phase": "resident_reference"}), flush=True)
        del reference
        gc.collect()
        torch.cuda.empty_cache()
        # The text-encoder proof has 241 MB total weights but an approximately 9 MB
        # working set. Enforce a 256 MiB allocator quota without occupying the rest of
        # the user's GPU. The planner's capacity budget below is this real quota,
        # not a claim that physical driver-free VRAM was reduced to 256 MiB.
        allocation_limit = 256 << 20 if component == "text_encoder" else 0
        if allocation_limit:
            torch.cuda.set_per_process_memory_fraction(
                allocation_limit / torch.cuda.get_device_properties(0).total_memory
            )
        checkpoint = Checkpoint(store_root, manifest)
        rows = checkpoint.rows(component)
        backend = StreamingFillBackend.for_script(
            checkpoint,
            rows,
            release="paging-proof/1",
            store="fixture",
            snapshot=manifest,
            placement="component_staged",
            device="cuda",
            window_bytes=1 << 20,
            slots=4,
            readers=4,
            inflight=2,
        )
        backend.resident_budget = layout.working_bytes + 4096
        walked = census(_Holder({component: model}))
        backend.expect({component: [d.key for d in walked.destinations]})
        fit = backend.fit(walked, encoded_leaves="refuse")
        assert fit is not None and fit["ok"]
        live = backend.materialize(_Holder({component: model}), walked)
        for d in walked.destinations:
            backend.fill(d.key, d.spec, live[d.key])
        backend.commit()
        del live, walked
        residency = ComponentResidency(backend=backend, torch=torch, placement="component_staged")
        ledger = Ledger(worker_pid=1)
        ledger.begin_generation(1)
        capacity = allocation_limit or torch.cuda.mem_get_info()[0]
        ledger.observe_construction(
            {
                "filled_bytes": total,
                "allocator_bytes": 0,
                "reserved_bytes": 0,
                "resident": {},
                "parked": [component],
                "evicted": {component: total},
                "paging": {component: layout.document()},
                "declared_scopes": {"forward": [component]},
                "device_free_bytes": capacity,
                "device_total_bytes": capacity,
            }
        )
        plan = PlanChooser(ledger).choose(
            DeclaredBinding(
                entrypoint_binding_digest="sha256:paging-proof",
                entrypoint="forward",
                model_class="PagingProof",
                model_binding_path="fixture:model",
                model_parameter_name="model",
                release="fixture/1",
                logical_weight_bytes=total,
            ),
            PreparedModel(delivery_rung="verbatim"),
            PreparedRequest.unresolved("forward", {}),
        )
        assert plan.placement == "component_staged"
        assert plan.headroom_bytes == 0 and dict(plan.scope_headroom_bytes) == {"forward": 0}
        print(
            json.dumps(
                {
                    "phase": "advisory_activation_reserve",
                    "allocator_limit_bytes": allocation_limit,
                    "planner_capacity_bytes": capacity,
                    "activation_estimate_bytes": plan.headroom_bytes,
                    "enforced_headroom_bytes": plan.headroom_bytes,
                }
            ),
            flush=True,
        )
        visited: list[Any] = []
        expected_cache = [True]

        def inspect_enter(unit: Any, module: Any, args: Any) -> None:
            assert not torch.is_autocast_cache_enabled()
            if component == "video_vae":
                assert not torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
                assert torch.is_autocast_enabled("cuda")
                assert torch.get_autocast_dtype("cuda") == torch.float16
            visited.append(unit)

        def inspect_exit(module: Any, args: Any, output: Any) -> None:
            assert torch.is_autocast_cache_enabled() == expected_cache[0]

        from functools import partial

        for unit in backend.paging[component].blocks:
            unit.owners[0][1].register_forward_pre_hook(partial(inspect_enter, unit))
            unit.owners[0][1].register_forward_hook(inspect_exit, always_call=True)
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        try:
            for iteration in range(2):
                residency.open_attempt(
                    plan.placement, plan.headroom_bytes, dict(plan.scope_headroom_bytes)
                )
                with torch.no_grad():
                    residency.admit("forward", (component,))
                    expected_cache[0] = iteration == 0
                    with torch.autocast(
                        "cuda",
                        enabled=component == "video_vae",
                        dtype=torch.float16,
                        cache_enabled=expected_cache[0],
                    ):
                        actual = outputs(invoke(model))
                        assert torch.is_autocast_cache_enabled() == expected_cache[0]
                    residency.release("forward", (component,))
                assert all(torch.equal(a, e) for a, e in zip(actual, expected, strict=True)), [
                    (a - e).abs().max().item() for a, e in zip(actual, expected, strict=True)
                ]
                assert not backend.page_resident
                assert all(
                    t.is_meta
                    for unit in layout.blocks
                    for _, m in unit.owners
                    for t in m.parameters(recurse=False)
                )
                print(
                    json.dumps(
                        {
                            "iteration": iteration,
                            "equal": True,
                            "stages": residency.attempt_stages,
                            "evictions": residency.attempt_evictions,
                            "peak_device_bytes": residency.attempt_absolute_peak_bytes,
                            "activation_peaks": residency.attempt_activation_peaks,
                            "weight_budget_bytes": backend.resident_budget,
                            "nominal_weight_bytes": total,
                            "host_ring_bytes": backend.pinned.get("registered_bytes"),
                            "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                            * 1024,
                        }
                    ),
                    flush=True,
                )
            residency.vacate()
            original_budget = backend.resident_budget
            backend.resident_budget = layout.working_bytes - 1
            allocated = torch.cuda.memory_allocated()
            stages = residency.stages
            try:
                residency.admit("forward", (component,))
            except ResidencyRefusal as exc:
                assert exc.code == "device_shortfall" and not residency.poisoned
                assert exc.shortfall == {
                    "resource": "vram",
                    "scope": "forward",
                    "needed_bytes": layout.working_bytes,
                    "available_bytes": layout.working_bytes - 1,
                }
            else:
                raise AssertionError("undersized working-set ceiling was accepted")
            assert torch.cuda.memory_allocated() == allocated and residency.stages == stages
            backend.resident_budget = original_budget
            with torch.no_grad():
                residency.admit("forward", (component,))
            stages = residency.stages
            expected_cache[0] = torch.is_autocast_cache_enabled()
            try:
                visited[0].owners[0][1]()  # reject before the block can inspect arguments
            except ResidencyRefusal as exc:
                assert exc.code == "paging_autograd"
            else:
                raise AssertionError("autograd paging was accepted")
            assert residency.stages == stages and not residency.poisoned
            residency.release("forward", (component,))
            print(
                json.dumps({"red": "working_set_and_autograd", "no_weight_mutation": True}),
                flush=True,
            )
            executed = visited[0]
            expected_cache[0] = arguments.terminal_cache == "enabled"
            with torch.autocast(
                "cuda",
                enabled=component == "video_vae",
                dtype=torch.float16,
                cache_enabled=expected_cache[0],
            ):
                if arguments.terminal_arm == "failure":

                    def fail_forward(*args: Any) -> None:
                        raise RuntimeError("controlled block failure")

                    failure_hook = executed.owners[0][1].register_forward_pre_hook(fail_forward)
                    with torch.no_grad():
                        residency.admit("forward", (component,))
                        try:
                            invoke(model)
                        except RuntimeError as exc:
                            assert str(exc) == "controlled block failure"
                        else:
                            raise AssertionError("controlled failure not reached")
                    failure_hook.remove()
                else:
                    # A real returned view, with nested containers and nonzero offset. The
                    # module's usual inference is irrelevant to this storage-lifetime red arm.
                    block = executed.owners[0][1]
                    retained = []

                    def return_view(*args: Any, **kwargs: Any) -> Any:
                        view = next(block.parameters()).flatten()[1:3]
                        retained.append(view)
                        if arguments.terminal_arm == "opaque":
                            from types import SimpleNamespace

                            return SimpleNamespace(hidden_weight=view)
                        return {"nested": (None, [view])}

                    block.forward = return_view
                    with torch.no_grad():
                        residency.admit("forward", (component,))
                        try:
                            invoke(model)
                        except ResidencyRefusal as exc:
                            assert exc.code == (
                                "paging_output_schema"
                                if arguments.terminal_arm == "opaque"
                                else "paging_output_alias"
                            )
                        else:
                            raise AssertionError("returned weight view was freed")
                    assert retained and retained[0].untyped_storage().nbytes() > 0
                    assert torch.equal(retained[0], next(block.parameters()).flatten()[1:3])
                assert bool(residency.poisoned) and backend.page_resident.get(component) is executed
                assert torch.is_autocast_cache_enabled() == expected_cache[0]
            print(
                json.dumps(
                    {
                        "red": arguments.terminal_arm,
                        "generation_poisoned": True,
                        "autocast_cache_restored": True,
                        "failed_block_retained_until_process_exit": True,
                    }
                ),
                flush=True,
            )
            print(
                json.dumps(
                    {
                        "seconds": time.monotonic() - start,
                        "store_bytes": sum(
                            p.stat().st_size for p in store_root.rglob("*") if p.is_file()
                        ),
                    }
                ),
                flush=True,
            )
        finally:
            if not residency.poisoned:
                residency.vacate()
            backend.close()


if __name__ == "__main__":
    main()
