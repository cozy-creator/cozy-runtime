"""Explicit model selectors stay inside their captured callable and model slot."""

from collections.abc import Mapping

import pytest

from cozy_runtime.internal import package_interface as interface
from cozy_runtime.internal.worker import machine_model_choices
from cozy_runtime.internal.worker.workspace_executions import ExecutionWorkspaceRefusal
from cozy_runtime.protocol import worker_pb2 as pb


def declaration(name: str, *parameters: str) -> interface.CallableDoc:
    return interface.CallableDoc(
        name=name,
        request={},
        result={},
        models=tuple(
            interface.ModelSlot(
                path=f"{name}.models.{parameter}", class_name="Model", component_use={}
            )
            for parameter in parameters
        ),
    )


def captured() -> tuple[pb.MachineExecutionCapture, Mapping[str, interface.PackageInterface]]:
    root = interface.PackageInterface(
        format=interface.SCHEMA,
        application="h3:app",
        jobs=(declaration("long_form"),),
        entrypoints=(
            declaration("ref2va", "model"),
            declaration("motion_segment_turbo", "base_model"),
            declaration("unused", "model"),
        ),
    )
    qwen = interface.PackageInterface(
        format=interface.SCHEMA,
        application="qwen:app",
        jobs=(),
        entrypoints=(declaration("generate_image", "base_model"),),
    )
    capture = pb.MachineExecutionCapture(
        root_installation_id="root",
        installed_packages=[pb.InstalledPackage(installation_id="root", package="paul/h3")],
        deferred_installations=[pb.DeferredInstallation(key="qwen@1", package="paul/qwen")],
        bindings=[
            pb.MachineCallableBinding(
                caller_installation_id="root",
                callee_installation_id="root",
                entrypoint="motion_segment_turbo",
            ),
            pb.MachineCallableBinding(
                caller_installation_id="root",
                callee_deferred_key="qwen@1",
                entrypoint="generate_image",
            ),
        ],
    )
    return capture, {"root": root, "qwen@1": qwen}


def test_workflow_overrides_keep_exact_child_scope_order_and_defaults() -> None:
    capture, interfaces = captured()
    original = capture.model_choices.add(parameter="motion_segment_turbo.models.base_model")
    original.adapters.add(model="paul/character", component="ref2va_dit")
    original.adapters.add(model="paul/style", component="ref2va_dit", scale="-0.5")
    capture.model_choices.add(
        parameter="paul/qwen/generate_image.models.base_model", repository="paul/qwen-checkpoint"
    )
    before = capture.SerializeToString(deterministic=True)
    resolved = machine_model_choices.resolve(capture, "long_form", interfaces)
    assert set(resolved) == {
        ("root", "motion_segment_turbo", "base_model"),
        ("qwen@1", "generate_image", "base_model"),
    }
    choice = resolved["root", "motion_segment_turbo", "base_model"]
    assert choice.parameter == "base_model" and not choice.repository
    assert [(a.model, a.source_component, a.scale) for a in choice.adapters] == [
        ("paul/character", "adapter", "1"),
        ("paul/style", "adapter", "-0.5"),
    ]
    assert capture.SerializeToString(deterministic=True) == before
    choice.adapters[0].model = "changed"
    assert original.adapters[0].model == "paul/character"


@pytest.mark.parametrize(
    "selector",
    [
        "base_model",
        "unused.models.model",
        "generate_image.models.base_model",
        "other/h3/ref2va.models.model",
        "motion_segment_turbo.models.missing",
    ],
)
def test_unknown_or_uncaptured_choices_never_broadcast(selector: str) -> None:
    capture, interfaces = captured()
    capture.model_choices.add(parameter=selector, repository="paul/base")
    with pytest.raises(ExecutionWorkspaceRefusal, match="unknown or ambiguous"):
        machine_model_choices.resolve(capture, "long_form", interfaces)


def test_root_alias_collision_and_ambiguous_package_instances_refuse() -> None:
    capture, interfaces = captured()
    capture.model_choices.add(parameter="model", repository="paul/base")
    capture.model_choices.add(parameter="ref2va.models.model", repository="paul/other")
    with pytest.raises(ExecutionWorkspaceRefusal, match="repeats"):
        machine_model_choices.resolve(capture, "ref2va", interfaces)
    del capture.model_choices[:]
    capture.installed_packages.add(installation_id="qwen2", package="paul/qwen")
    capture.bindings.add(
        caller_installation_id="root", callee_installation_id="qwen2", entrypoint="generate_image"
    )
    interfaces = {**interfaces, "qwen2": interfaces["qwen@1"]}
    capture.model_choices.add(
        parameter="paul/qwen/generate_image.models.base_model", repository="paul/base"
    )
    with pytest.raises(ExecutionWorkspaceRefusal, match="ambiguous"):
        machine_model_choices.resolve(capture, "long_form", interfaces)


@pytest.mark.parametrize("scale", ["NaN", "Inf", "1e999", "1x", " 1 "])
def test_invalid_adapter_strength_refuses_before_resolution(scale: str) -> None:
    capture, interfaces = captured()
    choice = capture.model_choices.add(parameter="motion_segment_turbo.models.base_model")
    choice.adapters.add(model="paul/adapter", component="ref2va_dit", scale=scale)
    with pytest.raises(ExecutionWorkspaceRefusal, match="finite"):
        machine_model_choices.resolve(capture, "long_form", interfaces)


def test_provider_choices_keep_profiles_but_reject_catalog_mixing() -> None:
    capture, interfaces = captured()
    choice = capture.model_choices.add(parameter="motion_segment_turbo.models.base_model")
    adapter = choice.adapters.add(
        source="hf://org/model@" + "a" * 40, profiles=["lora/1"], scale="0"
    )
    resolved = machine_model_choices.resolve(capture, "long_form", interfaces)
    actual = resolved["root", "motion_segment_turbo", "base_model"].adapters[0]
    assert actual.source == adapter.source and actual.profiles == ["lora/1"]
    assert actual.scale == "0"
    adapter.model = "paul/mixed"
    with pytest.raises(ExecutionWorkspaceRefusal, match="provider source or a catalog"):
        machine_model_choices.resolve(capture, "long_form", interfaces)
