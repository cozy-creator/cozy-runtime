"""Metadata and read-only probes for the agent's startup update transaction.

The Go agent owns staging, installation, recovery, and admission. This adapter
uses the installed interpreter's standard packaging rules; it never installs.
"""

import base64
import configparser
import email.parser
import hashlib
import importlib.metadata as metadata
import json
import platform
import sqlite3
import sys
import sysconfig
import urllib.request
import zipfile
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


def software_state(store):
    path = Path(store) / ".cozy-workspace/journal.sqlite3"
    if not path.exists():
        return {"quiescent": True, "accepted_work": False, "workspace_id": ""}
    # The Runtime owns this contract. Missing/unknown views defer updates; the
    # agent never reconstructs its retention or child/native rules.
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=2) as db:
        db.execute("PRAGMA query_only=ON")
        row = db.execute(
            "SELECT quiescent,accepted_work,workspace_id FROM machine_software_update_v1"
        ).fetchone()
        if row is None:
            raise ValueError("Runtime software-update state is unavailable")
        return {"quiescent": bool(row[0]), "accepted_work": bool(row[1]), "workspace_id": row[2]}


def read_json(url):
    # The guardian observes bytes as well as CPU/output using its existing stall
    # policy. Continuous metadata progress has no arbitrary total deadline.
    request = urllib.request.Request(url, headers={"User-Agent": "cozy-machine-startup"})
    with urllib.request.urlopen(request) as response:
        body = bytearray()
        while chunk := response.read1(64 << 10):
            body.extend(chunk)
            print(f"received {len(chunk)} metadata bytes", file=sys.stderr, flush=True)
        return json.loads(body)


def candidates(name, floor):
    minimum = Version(floor)
    index = read_json(f"https://pypi.org/pypi/{name}/json")
    compatible_tags = set(sys_tags())
    for version in sorted(index["releases"], key=Version, reverse=True):
        value = Version(version)
        if (
            not stable(version)
            or value.major != minimum.major
            or (minimum.major == 0 and value.minor != minimum.minor)
            or value < minimum
        ):
            continue
        release = read_json(f"https://pypi.org/pypi/{name}/{version}/json")
        info = release["info"]
        if Version(platform.python_version()) not in SpecifierSet(
            info.get("requires_python") or ""
        ):
            continue
        for file in release["urls"]:
            if file.get("yanked") or file.get("packagetype") != "bdist_wheel":
                continue
            distribution, wheel_version, _, tags = parse_wheel_filename(file["filename"])
            if (
                distribution != name
                or wheel_version != value
                or not compatible_tags.intersection(tags)
            ):
                continue
            if not any("manylinux" in tag.platform and tag.abi != "none" for tag in tags):
                continue
            yield {
                "version": version,
                "file": file["filename"],
                "sha256": file["digests"]["sha256"],
                "url": file["url"],
                "requires": info.get("requires_dist") or [],
            }
            break


def compatible(pair, installed, requirements, allow_missing=False):
    environment = default_environment()
    selected = installed | {name: row["version"] for name, row in pair.items()}
    extras = (
        ("",)
        + (("media",) if "av" in installed else ())
        + (("model-execution",) if "torch" in installed else ())
    )
    for name, texts in requirements.items():
        texts = pair[name]["requires"] if name in pair else texts
        for text in texts:
            requirement = Requirement(text)
            applicable = any(
                not requirement.marker
                or requirement.marker.evaluate(environment | {"extra": extra})
                for extra in (extras if name == "cozy-runtime" else ("",))
            )
            if not applicable:
                continue
            dependency = canonicalize_name(requirement.name)
            if name not in PAIR and dependency not in PAIR:
                continue
            if dependency not in selected and allow_missing:
                continue
            if (
                dependency not in selected
                or Version(selected[dependency]) not in requirement.specifier
            ):
                return False
    return True


def plan():
    installed, requirements = installed_metadata()
    tensorfs = list(candidates("tensorfs", installed["tensorfs"]))
    for runtime in candidates("cozy-runtime", installed["cozy-runtime"]):
        for storage in tensorfs:
            pair = {"cozy-runtime": runtime, "tensorfs": storage}
            if compatible(pair, installed, requirements, allow_missing=True):
                return pair
    raise ValueError("no stable compatible Runtime/TensorFS pair within the installed major lines")


def probe(overlay=None):
    if overlay:
        sys.path.insert(0, overlay)
    import contextlib
    import io

    import tensorfs

    from cozy.worker.v1 import wire_version
    from cozy_runtime.cli import runtime_worker
    from cozy_runtime.internal.worker.session import Worker

    assert callable(getattr(Worker, "request_idle_restart", None)), "Runtime lacks guarded restart"
    assert hasattr(tensorfs.Store, "model_source_operations"), "TensorFS lacks source recovery"
    try:
        import torch
    except ImportError:
        pass
    else:
        if torch.version.cuda:
            from cozy_runtime._kernels import _C

            assert _C is not None
    current = versions()
    installed, requirements = installed_metadata()
    assert compatible({}, installed, requirements), (
        "installed Runtime/TensorFS dependencies disagree"
    )
    resolve_additive(
        {name: {"version": installed[name], "requires": requirements[name]} for name in PAIR},
        installed,
        requirements,
        allow_fetch=False,
    )
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        assert runtime_worker.main(["capabilities", "--json"]) == 0
    capabilities = json.loads(output.getvalue())["capabilities"]
    return {
        "runtime": current["cozy-runtime"],
        "tensorfs": current["tensorfs"],
        "wire_minor": wire_version.WIRE_MINOR,
        "minimum_wire_minor": wire_version.MIN_COMPATIBLE_WIRE_MINOR,
        "capabilities": capabilities,
    }


def wheel_metadata(path):
    with zipfile.ZipFile(path) as wheel:
        names = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError("wheel must contain one distribution metadata record")
        info = email.parser.BytesParser().parsebytes(wheel.read(names[0]))
    name = canonicalize_name(info["Name"])
    distribution, version, _, tags = parse_wheel_filename(Path(path).name)
    if (
        name != distribution
        or Version(info["Version"]) != version
        or not set(sys_tags()).intersection(tags)
    ):
        raise ValueError("wheel metadata, filename or interpreter tags disagree")
    return name, {
        "name": name,
        "version": str(version),
        "requires": info.get_all("Requires-Dist", []),
    }


def dependency_candidates(name, constraints):
    index = read_json(f"https://pypi.org/pypi/{name}/json")
    ranks = {tag: rank for rank, tag in enumerate(sys_tags())}
    for version in sorted(index["releases"], key=Version, reverse=True):
        if not stable(version) or any(Version(version) not in req.specifier for req in constraints):
            continue
        release = read_json(f"https://pypi.org/pypi/{name}/{version}/json")
        info = release["info"]
        if Version(platform.python_version()) not in SpecifierSet(
            info.get("requires_python") or ""
        ):
            continue
        wheels = []
        for row in release["urls"]:
            if row.get("yanked") or row.get("packagetype") != "bdist_wheel":
                continue
            distribution, value, _, tags = parse_wheel_filename(row["filename"])
            compatible_tags = set(ranks).intersection(tags)
            if distribution == name and value == Version(version) and compatible_tags:
                wheels.append((min(ranks[tag] for tag in compatible_tags), row))
        if wheels:
            row = min(wheels, key=lambda item: item[0])[1]
            yield {
                "name": name,
                "version": version,
                "file": row["filename"],
                "sha256": row["digests"]["sha256"],
                "url": row["url"],
                "requires": info.get("requires_dist") or [],
            }


def resolve_additive(pair, installed, requirements, allow_fetch=True):
    environment = default_environment()
    root_extras = (
        {""}
        | ({"media"} if "av" in installed else set())
        | ({"model-execution"} if "torch" in installed else set())
    )

    def search(selected):
        rows = (
            {
                name: {"version": version, "requires": requirements.get(name, [])}
                for name, version in installed.items()
            }
            | pair
            | selected
        )
        extras, expanded, constraints = {"cozy-runtime": root_extras, "tensorfs": {""}}, {}, {}
        queue = list(pair)
        while queue:
            name = queue.pop()
            active = extras.get(name, {""})
            if expanded.get(name, set()) >= active or name not in rows:
                continue
            expanded[name] = set(active)
            for text in rows[name]["requires"]:
                req = Requirement(text)
                if req.marker and not any(
                    req.marker.evaluate(environment | {"extra": extra}) for extra in active
                ):
                    continue
                dependency = canonicalize_name(req.name)
                constraints.setdefault(dependency, []).append(req)
                if dependency in rows and Version(rows[dependency]["version"]) not in req.specifier:
                    raise ValueError(
                        f"installed dependency constraints conflict at {dependency}; "
                        "base upgrades are not implicit"
                    )
                if dependency not in installed and req.url:
                    raise ValueError("missing direct-URL dependencies require a prepared base")
                wanted = extras.setdefault(dependency, {""})
                wanted.update(req.extras)
                if dependency in rows and not expanded.get(dependency, set()) >= wanted:
                    queue.append(dependency)
        # Unrelated installed frameworks remain fixed and may constrain a newly
        # introduced distribution or either replaced member of the pair.
        for name, texts in requirements.items():
            if name in pair:
                continue
            for text in texts:
                req = Requirement(text)
                dependency = canonicalize_name(req.name)
                if dependency not in constraints and dependency not in pair:
                    continue
                if req.marker and not req.marker.evaluate(environment | {"extra": ""}):
                    continue
                constraints.setdefault(dependency, []).append(req)
                if dependency in rows and Version(rows[dependency]["version"]) not in req.specifier:
                    raise ValueError(f"installed reverse dependency prevents changing {dependency}")
        missing = sorted(set(constraints) - rows.keys())
        if not missing:
            return selected
        name = missing[0]
        if not allow_fetch:
            raise ValueError(f"candidate dependency closure is incomplete: {name}")
        for candidate in dependency_candidates(name, constraints[name]):
            try:
                return search(selected | {name: candidate})
            except ValueError:
                continue
        raise ValueError(
            f"no additive wheel-only closure for {name}; "
            "installed versions are fixed, prepare a compatible base"
        )

    return list(search({}).values())


def dependencies(paths):
    pair = dict(wheel_metadata(path) for path in paths)
    if set(pair) != set(PAIR):
        raise ValueError("dependency resolution requires exactly the Runtime/TensorFS pair")
    return resolve_additive(pair, *installed_metadata())


def dependency_files(path):
    name, info = wheel_metadata(path)
    try:
        metadata.distribution(name)
    except metadata.PackageNotFoundError:
        pass
    else:
        raise ValueError(f"additive dependency {name} is already installed")
    scheme = sysconfig.get_paths()
    targets = set()
    with zipfile.ZipFile(path) as wheel:
        for member in wheel.namelist():
            if member.endswith("/"):
                continue
            parts = Path(member).parts
            if ".." in parts or Path(member).is_absolute():
                raise ValueError("dependency wheel contains an unsafe path")
            if parts[0].endswith(".data"):
                if len(parts) < 3 or parts[1] not in ("purelib", "platlib", "scripts"):
                    raise ValueError(
                        "dependency wheel requires unsupported data installation; prepare the base"
                    )
                target = Path(scheme[parts[1]]).joinpath(*parts[2:])
            else:
                target = Path(scheme["purelib"]) / member
            targets.add(target)
            if member.endswith(".dist-info/entry_points.txt"):
                entries = configparser.ConfigParser()
                entries.optionxform = str
                entries.read_string(wheel.read(member).decode())
                for section in ("console_scripts", "gui_scripts"):
                    for command in entries[section] if entries.has_section(section) else ():
                        targets.add(Path(scheme["scripts"]) / command)
            if member.endswith(".dist-info/METADATA"):
                targets.update(
                    target.parent / extra for extra in ("INSTALLER", "REQUESTED", "direct_url.json")
                )
    prefix = Path(sys.prefix).resolve()
    for target in targets:
        if not target.resolve().is_relative_to(prefix) or target.exists() or target.is_symlink():
            raise ValueError(f"additive dependency would overwrite an existing path: {target}")
    return info | {"paths": sorted(str(path) for path in targets)}


def removable_dependencies(rows):
    removable = []
    for row in rows:
        try:
            distribution = metadata.distribution(row["name"])
        except metadata.PackageNotFoundError:
            if any(Path(path).exists() for path in row["paths"]):
                raise ValueError(
                    f"partial dependency {row['name']} requires repair; preserving uncertain files"
                ) from None
            continue
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        if (
            distribution.version != row["version"]
            or direct.get("url") != Path(row["path"]).as_uri()
        ):
            raise ValueError(
                f"dependency {row['name']} changed after installation; preserving operator changes"
            )
        wheel_path = Path(row["path"])
        if hashlib.sha256(wheel_path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("private dependency wheel no longer proves installation ownership")
        # uv records a private file URL with empty archive_info. Verify its
        # retained wheel directly, then compare immutable payloads as well as
        # the installed RECORD (which owns generated entry points).
        with zipfile.ZipFile(wheel_path) as wheel:
            for member in wheel.namelist():
                if member.endswith("/") or member.endswith(".dist-info/RECORD"):
                    continue
                parts = Path(member).parts
                if parts[0].endswith(".data"):
                    if parts[1] == "scripts":
                        continue  # installer rewrites #!python; RECORD covers it
                    target = Path(sysconfig.get_paths()[parts[1]]).joinpath(*parts[2:])
                else:
                    target = Path(distribution.locate_file(member))
                if not target.is_file() or target.read_bytes() != wheel.read(member):
                    raise ValueError(f"dependency payload changed; preserving {target}")
        for file in distribution.files or ():
            if file.hash is None:
                continue
            target = Path(distribution.locate_file(file))
            if (
                not target.is_file()
                or file.hash.mode != "sha256"
                or base64.urlsafe_b64encode(hashlib.sha256(target.read_bytes()).digest())
                .rstrip(b"=")
                .decode()
                != file.hash.value
            ):
                raise ValueError(f"installed dependency file changed; preserving {target}")
        removable.append(row["name"])
    return removable


def main():
    operation = sys.argv[1]
    if operation == "inspect":
        pair = versions()
        result = {
            "runtime": pair["cozy-runtime"],
            "tensorfs": pair["tensorfs"],
            "eligible": all(stable(v) for v in pair.values()),
            **software_state(sys.argv[2]),
        }
    elif operation == "plan":
        result = plan()
    elif operation == "dependencies":
        result = dependencies(sys.argv[2:])
    elif operation == "dependency-files":
        result = [dependency_files(path) for path in sys.argv[2:]]
    elif operation == "removable-dependencies":
        result = removable_dependencies(json.loads(sys.argv[2]))
    elif operation == "probe":
        result = probe(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        raise ValueError("unknown startup update operation")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
