#!/usr/bin/env python3
"""Validate an immutable new publication, not ordinary source development.

Only the tag-triggered publish workflow calls this guard. Source branches may
retain a released manifest version; scripts/build-dev-wheel.py gives their
unpublished artifacts a source/donor/recipe-derived local version.

The push must create the matching tag on this exact clean checkout, and PyPI
must not already know the version (including a partial release). Existing
filenames remain protected by PyPI and the publishing action; this preflight
also prevents filling an existing version with artifacts from another source.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import tomllib
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPI_URL = "https://pypi.org"


def git(*args: str) -> str:
    done = subprocess.run(("git", "-C", str(ROOT), *args), capture_output=True, text=True)
    if done.returncode != 0:
        raise ValueError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout.strip()


def commit(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("release event must identify an exact Git object")
    return git("rev-parse", "--verify", f"{value}^{{commit}}")


def unpublished(version: str) -> None:
    url = f"{PYPI_URL}/pypi/cozy-runtime/{urllib.parse.quote(version, safe='')}/json"
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            if response.status != 200:
                raise ValueError(f"cannot establish PyPI version absence: HTTP {response.status}")
    except urllib.error.HTTPError as error:
        code = error.code
        error.close()
        if code == 404:
            return
        raise ValueError(f"cannot establish PyPI version absence: HTTP {code}") from error
    raise ValueError(f"cozy-runtime {version} already exists on PyPI; never replace a release")


def main() -> int:
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.source.resolve()
    event = json.loads(pathlib.Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    manual = os.environ["GITHUB_EVENT_NAME"] == "workflow_dispatch"
    if manual:
        tag = event.get("inputs", {}).get("release_tag", "")
        if not isinstance(tag, str) or not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag):
            raise ValueError("manual publication requires an existing version tag")
        ref = "refs/tags/" + tag
    else:
        ref = os.environ["GITHUB_REF"]
        if (
            os.environ["GITHUB_EVENT_NAME"] != "push"
            or not ref.startswith("refs/tags/v")
            or event["ref"] != ref
            or event["created"] is not True
            or event["deleted"] is not False
            or event["before"] != "0" * 40
        ):
            raise ValueError("publication requires a new tag push or an explicit manual tag")
    with (ROOT / "pyproject.toml").open("rb") as source:
        version = tomllib.load(source)["project"]["version"]
    if not isinstance(version, str) or not version:
        raise ValueError("pyproject project.version must be a nonempty string")
    tag = ref.removeprefix("refs/tags/")
    if tag != f"v{version}":
        raise ValueError(f"tag {tag} != pyproject version {version}")
    head = git("rev-parse", "HEAD")
    expected = [git("rev-parse", "--verify", f"{ref}^{{commit}}")]
    if not manual:
        expected.extend((commit(os.environ["GITHUB_SHA"]), commit(event["after"])))
    if any(value != head for value in expected):
        raise ValueError("release tag, push event, and checked-out source must agree")
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("release checkout has uncommitted or untracked changes")
    unpublished(version)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            print(f"source={head}", file=stream)
    print(f"release guard green: {tag} at {head}, version absent from PyPI")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, ValueError) as error:
        raise SystemExit(f"release guard RED: {error}") from error
