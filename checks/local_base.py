"""Deterministic checks for the two admitted observed-local base profiles."""

from __future__ import annotations

from cozy_runtime.internal.local_base import LocalBaseRefusal, _cuda_profile, _profile


def main() -> None:
    profile, target, minimum_cuda, minimum_driver = _profile([])
    assert profile["torch_release"] == "none"
    assert target["accelerator_backend"] == "none"
    assert (minimum_cuda, minimum_driver) == ("none", "none")

    profile, target, minimum_cuda, minimum_driver = _cuda_profile("2.13.0+cu130", "13.0")
    assert profile == {
        "accelerator_build": "cu130",
        "os_cpu": "linux-x86",
        "python_abi": "cp312",
        "torch_release": "2.13.0",
    }
    assert target["accelerator_backend"] == "cuda"
    assert (minimum_cuda, minimum_driver) == ("13.0", "580.126.20")

    for version, cuda in (("2.12.0+cu130", "13.0"), ("2.13.0+cu128", "12.8")):
        try:
            _cuda_profile(version, cuda)
        except LocalBaseRefusal:
            pass
        else:
            raise AssertionError(f"admitted unsupported local base {version} / CUDA {cuda}")


if __name__ == "__main__":
    main()
