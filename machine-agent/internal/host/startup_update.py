"""Metadata and read-only probes for the agent's startup update transaction.

The Go agent owns staging, installation, recovery, and admission. This adapter
uses the installed interpreter's standard packaging rules; it never installs.
"""
import importlib.metadata as metadata
import json
import platform
import sqlite3
import sys
import urllib.request
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

PAIR = ("cozy-runtime", "tensorfs")


def stable(value):
    value = Version(value)
    return not (value.is_prerelease or value.is_devrelease or value.local)


def versions():
    return {name: metadata.version(name) for name in PAIR}


def installed_metadata():
    installed, requirements = {}, {}
    for distribution in metadata.distributions():
        name = canonicalize_name(distribution.metadata["Name"])
        if name not in installed:
            installed[name] = distribution.version
            requirements[name] = distribution.requires or []
    return installed, requirements


def safe_journal(store):
    path = Path(store) / ".cozy-workspace/journal.sqlite3"
    if not path.exists():
        return True
    # Unknown views, unreadable records and unfinished/retained work defer updates.
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=2) as db:
        db.execute("PRAGMA query_only=ON")
        return db.execute(
            "SELECT 1 FROM machine_runs_v1 WHERE "
            "state NOT IN ('succeeded','failed','canceled') OR collected=0 LIMIT 1"
        ).fetchone() is None


def read_json(url):
    # A metadata request has a 15-second network observation budget. Failure
    # leaves the verified installed pair in service; it never cancels a run.
    request = urllib.request.Request(url, headers={"User-Agent": "cozy-machine-startup"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def candidates(name, floor):
    minimum = Version(floor)
    index = read_json(f"https://pypi.org/pypi/{name}/json")
    compatible_tags = set(sys_tags())
    for version in sorted(index["releases"], key=Version, reverse=True):
        value = Version(version)
        if not stable(version) or value.major != minimum.major or value < minimum:
            continue
        release = read_json(f"https://pypi.org/pypi/{name}/{version}/json")
        info = release["info"]
        if Version(platform.python_version()) not in SpecifierSet(info.get("requires_python") or ""):
            continue
        for file in release["urls"]:
            if file.get("yanked") or file.get("packagetype") != "bdist_wheel":
                continue
            distribution, wheel_version, _, tags = parse_wheel_filename(file["filename"])
            if distribution != name or wheel_version != value or not compatible_tags.intersection(tags):
                continue
            if not any("manylinux" in tag.platform and tag.abi != "none" for tag in tags):
                continue
            yield {"version": version, "file": file["filename"], "sha256": file["digests"]["sha256"], "url": file["url"], "requires": info.get("requires_dist") or []}
            break


def compatible(pair, installed, requirements):
    environment = default_environment()
    selected = installed | {name: row["version"] for name, row in pair.items()}
    extras = ("",) + (("media",) if "av" in installed else ()) + (("model-execution",) if "torch" in installed else ())
    for name, texts in requirements.items():
        texts = pair[name]["requires"] if name in pair else texts
        for text in texts:
            requirement = Requirement(text)
            applicable = any(not requirement.marker or requirement.marker.evaluate(environment | {"extra": extra}) for extra in (extras if name == "cozy-runtime" else ("",)))
            if not applicable:
                continue
            dependency = canonicalize_name(requirement.name)
            if name not in PAIR and dependency not in PAIR:
                continue
            if dependency not in selected or Version(selected[dependency]) not in requirement.specifier:
                return False
    return True


def plan():
    installed, requirements = installed_metadata()
    tensorfs = list(candidates("tensorfs", installed["tensorfs"]))
    for runtime in candidates("cozy-runtime", installed["cozy-runtime"]):
        for storage in tensorfs:
            pair = {"cozy-runtime": runtime, "tensorfs": storage}
            if compatible(pair, installed, requirements):
                return pair
    raise ValueError("no stable compatible Runtime/TensorFS pair within the installed major lines")


def probe(overlay=None):
    if overlay:
        sys.path.insert(0, overlay)
    from cozy_runtime.internal.worker.session import Worker
    import tensorfs
    from cozy.worker.v1 import wire_version
    from cozy_runtime.cli import runtime_worker
    import contextlib
    import io
    assert callable(getattr(Worker, "request_idle_restart", None)), "Runtime lacks guarded restart"
    assert hasattr(tensorfs.Store, "model_source_operations"), "TensorFS lacks source recovery"
    try:
        import torch
    except ImportError:
        pass
    else:
        if torch.version.cuda:
            from cozy_runtime._kernels import _C
    current = versions()
    installed, requirements = installed_metadata()
    assert compatible({}, installed, requirements), "installed Runtime/TensorFS dependencies disagree"
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        assert runtime_worker.main(["capabilities", "--json"]) == 0
    capabilities = json.loads(output.getvalue())["capabilities"]
    return {"runtime": current["cozy-runtime"], "tensorfs": current["tensorfs"], "wire_minor": wire_version.WIRE_MINOR, "minimum_wire_minor": wire_version.MIN_COMPATIBLE_WIRE_MINOR, "capabilities": capabilities}


def main():
    operation = sys.argv[1]
    if operation == "inspect":
        pair = versions()
        result = {"runtime": pair["cozy-runtime"], "tensorfs": pair["tensorfs"], "eligible": all(stable(v) for v in pair.values()) and safe_journal(sys.argv[2])}
    elif operation == "plan":
        result = plan()
    elif operation == "probe":
        result = probe(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        raise ValueError("unknown startup update operation")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
