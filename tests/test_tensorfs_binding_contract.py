"""The runtime↔tensorfs seam, and the two things its compiler cannot say about itself.

`tensorfs` is a compiled PyO3 extension in a SEPARATE REPOSITORY, resolved by version range.
Until tfs-077 nothing in either tree checked that a call site still matched the binding it
called: tensorfs shipped `py.typed` over an `_ext.pyi` whose operative content was
`__getattr__ -> Any`, so every method on that page type-checked as `Any` under
`strict = true`, and `fill.tensorfs_module() -> Any` erased what little was left.
`Store.prepare_model_source` was called with six arguments against a five-parameter binding
and shipped (cozy-runtime#296); it refused every model-source preparation on a rented pod.

THIS FILE USED TO BE THE COMPILER. It carried a hand-transcribed table of 34 frozen
signatures — the subset someone remembered — and compared them against `inspect.signature`
on every run. That table is GONE, and so is the arm that policed its coverage. tensorfs now
ships a complete `_ext.pyi` with no escape hatch, held against the built extension by
`tensorfs/scripts/stub_conform.py` for all ~110 exports rather than 34, in the repository
where the change is made rather than in the consumer a wheel bump later. Two mechanisms for
one job is a liability; the stub wins and the table goes.

What the stub cannot do is vouch for itself, and that is what is left here:

1. **THAT THE STUB IS REAL.** mypy reports success just as cheerfully over an `Any` as over
   a checked call. If a resolved wheel predates tfs-077, or a `__getattr__` comes back, the
   whole strict-mode run silently stops checking this seam and stays green — the exact
   failure mode that cost two pods. That is invisible to mypy and visible here.

2. **THAT THE CHECK CAN FAIL.** A check that cannot fail is indistinguishable from no
   check, so the six-argument call that shipped is put back through mypy on every run and
   required to be REJECTED.

3. **THAT THE ARGUMENTS MEAN WHAT THEY SAY.** `_ext.pyi` asserts `files` is
   `Sequence[tuple[str, str, int, StrPath]]`; nothing compares that to Rust, because pyo3
   erases types into `PyObject` and `__text_signature__` carries no annotation. The
   integration arms below drive the real runtime function against a real `Store` on disk,
   where a refusal proves the arguments BOUND — the strongest form available.

STILL NOT COVERED, stated so the coverage is not mistaken for more than it is: the `tfs`
CLI's TEXT stdout and vendored artifacts are also crossings and neither is a Python
signature (xs-024 §1B, §1C); and this checks the wheel in THIS environment, while worker
images pin their own (xs-024 §1).
"""

from __future__ import annotations

import ast
import hashlib
import textwrap
from pathlib import Path

import pytest
import tensorfs

from cozy_runtime.internal.worker import model_source_prepare
from cozy_runtime.protocol import worker_pb2 as pb

#: The one binding this seam has broken on, twice. Four parameters after `self`.
SEAM = "prepare_model_source"


def _stub() -> tuple[Path, str]:
    """The `.pyi` that SHIPS BESIDE the installed extension — the one mypy actually reads."""

    path = Path(tensorfs._ext.__file__).with_name("_ext.pyi")
    assert path.exists(), (
        f"tensorfs {tensorfs.__version__} ships `py.typed` and no `_ext.pyi` at {path}: it "
        "promises every consumer a checked seam and delivers an unchecked one"
    )
    return path, path.read_text()


def test_the_installed_tensorfs_ships_a_stub_that_can_actually_refuse() -> None:
    """The guard that keeps mypy from being vacuous, which mypy cannot be.

    `py.typed` is a FACT this package asserts about itself, and for two months it was false.
    A green strict-mode run over `__getattr__ -> Any` looks exactly like a green strict-mode
    run over a checked call, so nothing downstream could tell the difference — and a version
    range that resolves to a pre-tfs-077 wheel puts it back without a line changing here.
    """

    path, source = _stub()
    assert (path.parent / "py.typed").exists(), "tensorfs stopped shipping `py.typed`"

    tree = ast.parse(source)
    hatches = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
    ]
    assert not hatches, (
        f"tensorfs {tensorfs.__version__}'s `_ext.pyi` has a `__getattr__` again. Every "
        "attribute it covers types as `Any`, mypy stops checking this seam, and every run in "
        "this repository goes green over it. That is how cozy-runtime#296 shipped."
    )

    store = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Store"), None
    )
    assert store is not None, "`_ext.pyi` no longer declares Store"
    seam = next((n for n in store.body if isinstance(n, ast.FunctionDef) and n.name == SEAM), None)
    assert seam is not None, (
        f"`_ext.pyi` declares Store without `{SEAM}`. The stub is the seam's compiler, and "
        "an undeclared method is an attribute error mypy will not find."
    )
    names = [a.arg for a in seam.args.posonlyargs + seam.args.args]
    assert names == ["self", "operation_id", "source_selection_digest", "profiles", "files"], (
        f"`Store.{SEAM}` in the installed stub takes {names}. tensorfs CI holds the stub "
        "against the built extension, so this is a real seam change: repair the call sites "
        "in this repository, then follow the new signature here."
    )


def _mypy(tmp_path: Path, name: str, body: str) -> tuple[str, int]:
    """One snippet through THIS repository's own mypy configuration, as CI runs it."""

    from mypy import api

    module = tmp_path / f"{name}.py"
    module.write_text(textwrap.dedent(body).lstrip())
    out, _err, status = api.run(["--no-incremental", str(module)])
    return out, status


def test_mypy_rejects_the_six_argument_call_that_shipped(tmp_path: Path) -> None:
    """The check's own red arm. A check that cannot fail is indistinguishable from no check.

    This is cozy-runtime#296 put back: `prepare_model_source` called with `source_uri` and
    `declared_license` after tensorfs `03c9713` deleted them. It passed mypy for two days and
    then refused every foreign-source preparation on a rented pod. It must not pass now.

    The four-argument arm runs beside it, because a checker that rejects everything proves
    nothing either.
    """

    good, good_status = _mypy(
        tmp_path,
        "binds",
        """
        import tensorfs

        def call(store: tensorfs.Store) -> None:
            store.prepare_model_source("op", "sha256:00", [], [])
        """,
    )
    assert good_status == 0, f"the call the binding DOES take was rejected:\n{good}"

    bad, bad_status = _mypy(
        tmp_path,
        "over_call",
        """
        import tensorfs

        def call(store: tensorfs.Store) -> None:
            store.prepare_model_source("op", "sha256:00", [], [], "uri", "license")
        """,
    )
    assert bad_status != 0, (
        "mypy ACCEPTED a six-argument call to a four-parameter binding. The seam is typing "
        f"as `Any` again — tensorfs {tensorfs.__version__}'s stub, this repository's mypy "
        "configuration, or `fill.TensorfsFacade`. This is exactly the state in which "
        "cozy-runtime#296 shipped."
    )
    assert "prepare_model_source" in bad, bad


def test_mypy_rejects_a_call_through_the_facade_accessor(tmp_path: Path) -> None:
    """The seam is reached through `fill.tensorfs_module()` at nineteen sites, not by import.

    That accessor returned `Any` until tfs-077, so typing the stub alone would have fixed
    nothing at any of them. `fill.TensorfsFacade` is what carries the types across, and mypy
    checks the real module against it at the one `return` — so this arm covers the accessor
    and the protocol together.
    """

    out, status = _mypy(
        tmp_path,
        "through_fill",
        """
        from pathlib import Path

        from cozy_runtime.internal import fill

        def call(root: Path) -> None:
            store = fill.open_store(fill.tensorfs_module(), root)
            store.prepare_model_source("op", "sha256:00", [], [], "uri", "license")
        """,
    )
    assert status != 0, (
        "`fill.tensorfs_module()` is erasing the seam to `Any` again; every call site that "
        f"goes through it is unchecked. mypy said:\n{out}"
    )
    assert "prepare_model_source" in out, out


def _request(tmp_path: Path, *, operation_id: str) -> pb.PrepareModelSourceRequest:
    """One shape-valid request over a real file, so validation passes and the call crosses."""

    blob = tmp_path / "model.safetensors"
    blob.write_bytes(b"a real file that is not a real checkpoint" * 8)
    data = blob.read_bytes()
    return pb.PrepareModelSourceRequest(
        operation_id=operation_id,
        source_selection_digest=hashlib.sha256(b"selection").digest(),
        profiles=[pb.ModelSourceProfile(slot="unet", profile="sdxl/unet")],
        files=[
            pb.LocalModelSourceFile(
                member="model.safetensors",
                object_id="sha256:" + hashlib.sha256(data).hexdigest(),
                length=len(data),
                path=str(blob),
            )
        ],
    )


def test_prepare_model_source_reaches_tensorfs_and_comes_back_typed(tmp_path: Path) -> None:
    """The integration arm. A shape-valid request must REACH the binding.

    The file is deliberately not a checkpoint, so tensorfs answers with a typed refusal
    rather than a manifest — which is the point. A refusal proves the arguments BOUND. With
    the six-argument call this repair deleted, this test raises `TypeError` instead, because
    `prepare_model_source` re-raises anything that is not a tensorfs `Refusal`.
    """

    root = tmp_path / "tensorfs"
    tensorfs.Store.ensure(str(root))

    result = model_source_prepare.prepare_model_source(
        _request(tmp_path, operation_id="op-contract-1"), tensorfs_root=root
    )

    assert result.outcome == pb.MODEL_SOURCE_PREPARE_OUTCOME_REFUSED
    assert result.safe_code.startswith("model_source_tensorfs_"), result.safe_code


def test_release_model_source_reaches_tensorfs_and_comes_back_typed(tmp_path: Path) -> None:
    """The same proof for the lifecycle half, which has its own arity."""

    root = tmp_path / "tensorfs"
    tensorfs.Store.ensure(str(root))

    with pytest.raises(tensorfs.errors.Refusal):
        model_source_prepare.release_model_source(
            tensorfs_root=root, operation_id="op-contract-absent"
        )


def test_prepare_model_source_refuses_a_relative_store_before_touching_tensorfs(
    tmp_path: Path,
) -> None:
    """The refusal paths of this module had no coverage at all; give them a floor."""

    result = model_source_prepare.prepare_model_source(
        _request(tmp_path, operation_id="op-contract-2"), tensorfs_root=Path("relative/store")
    )

    assert result.outcome == pb.MODEL_SOURCE_PREPARE_OUTCOME_REFUSED
    assert result.safe_code == "model_source_store_invalid"


def test_prepare_model_source_refuses_an_unsorted_profile_set(tmp_path: Path) -> None:
    request = _request(tmp_path, operation_id="op-contract-3")
    del request.profiles[:]
    request.profiles.extend(
        [
            pb.ModelSourceProfile(slot="vae", profile="sdxl/vae"),
            pb.ModelSourceProfile(slot="unet", profile="sdxl/unet"),
        ]
    )

    result = model_source_prepare.prepare_model_source(request, tensorfs_root=tmp_path / "s")

    assert result.safe_code == "model_source_profile_order"


def test_two_members_carrying_one_object_share_one_path_and_are_admitted(
    tmp_path: Path,
) -> None:
    """A content-addressed Store makes member->path NON-injective, and that is correct.

    MiniMax-H3's `FL2VA/transformer/model.safetensors.index.json` and its `Ref2VA/`
    counterpart are byte-identical, so its 48 selected members admit 47 objects and two
    members name one file. Run 290 downloaded all 210.3 GB and then refused here, on
    `model_source_file_identity_invalid`, because the duplicate-path check read a shared
    path as a fault. It could not fire before tfs-060: those two members had never been
    fetched, so the pair had never reached this loop.
    """

    request = _request(tmp_path, operation_id="op-contract-shared")
    twin = pb.LocalModelSourceFile()
    twin.CopyFrom(request.files[0])
    twin.member = "z-second-member.safetensors"
    request.files.append(twin)

    result = model_source_prepare.prepare_model_source(request, tensorfs_root=tmp_path / "s")

    # It reaches TensorFS rather than refusing here. The store path is absent, so the
    # answer is a store refusal -- proof the identity gate passed the shared path.
    assert result.safe_code != "model_source_file_identity_invalid"


def test_a_tensorfs_refusal_carries_its_own_words(tmp_path: Path) -> None:
    """TensorFS says WHICH key it refused on; a fixed string threw that away.

    Run 294 refused `model_source_tensorfs_duplicate_key` after moving 210.3 GB and the
    operator's row read only "TensorFS refused source preparation" — so the duplicated
    path had to be found by reading tensorfs source. The code was preserved and the one
    fact that localises it was not.
    """

    request = _request(tmp_path, operation_id="op-contract-detail")
    # An absent store makes tensorfs refuse with a detail naming what it could not find.
    result = model_source_prepare.prepare_model_source(request, tensorfs_root=tmp_path / "absent")

    assert result.outcome == pb.MODEL_SOURCE_PREPARE_OUTCOME_REFUSED
    assert result.safe_code.startswith("model_source_tensorfs_")
    # The detail is no longer the bare fixed string: TensorFS's own words survive.
    assert result.safe_detail != "TensorFS refused source preparation"
    assert len(result.safe_detail) > len("TensorFS refused source preparation")


def test_two_different_objects_claiming_one_path_are_still_refused(tmp_path: Path) -> None:
    """The invariant the duplicate-path check actually protects, kept.

    A shared path is legitimate only when the objects agree. Two DIFFERENT object ids on
    one file is a real fault and must stay a refusal, or the repair above would have
    widened the gate instead of correcting it.
    """

    request = _request(tmp_path, operation_id="op-contract-collide")
    twin = pb.LocalModelSourceFile()
    twin.CopyFrom(request.files[0])
    twin.member = "z-second-member.safetensors"
    twin.object_id = "sha256:" + "0" * 64
    request.files.append(twin)

    result = model_source_prepare.prepare_model_source(request, tensorfs_root=tmp_path / "s")

    assert result.safe_code == "model_source_file_identity_invalid"
    # Which member failed must be in the detail: 48 candidates and one static string is
    # how run 290 refused after moving 210.3 GB while naming nothing.
    assert "z-second-member.safetensors" in result.safe_detail


def test_prepare_model_source_refuses_a_file_whose_length_is_a_lie(tmp_path: Path) -> None:
    request = _request(tmp_path, operation_id="op-contract-4")
    request.files[0].length += 1

    result = model_source_prepare.prepare_model_source(request, tensorfs_root=tmp_path / "s")

    assert result.safe_code == "model_source_file_identity_invalid"
