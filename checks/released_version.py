#!/usr/bin/env python3
"""A released version number names ONE tree, forever.

Publication IS the tag push here: `.github/workflows/publish.yaml` triggers on `v*` and
there is no local publish path, so the `v*` tags ARE the released set. Checked
2026-09-03 in both this repository and its sibling: the PyPI release list and the `v*`
tags agree exactly, with no version on one side only. That is why this guard reads git
refs and never the network.

The defect it refuses: content merged while the manifest still declares an already
released number. Nothing rejected that, so a pin resolved to different bytes depending on
when the wheel was built, and the divergence surfaced hours later inside a baked image on
a rented pod. Four paid failures on 2026-09-03 were this one defect.

    if `v<declared version>` exists, HEAD's tree must be the tree that tag names.

Consequences, all deliberate:

  * A version that was never tagged is in development and may change freely. That is the
    ordinary case, and this guard is silent for it.
  * Bumping the version in the same commit that changes content PASSES, because the new
    number carries no tag. The correct workflow needs no exemption.
  * A tag whose publish failed still counts. A tag is a public immutable name that a pin,
    a bake, or `pip install git+...@v0.3.4` may already resolve; re-pointing it over
    changed content is exactly the defect. Tagged is released.
  * The comparison is the WHOLE tree, not the packaged subset: `v0.2.15` names a
    checkout, and a README fixed under that number makes the name ambiguous. The remedy
    is one line in the manifest.

This runs on every push and pull_request, ahead of the merge. It is deliberately NOT in
publish.yaml, where the tag already points at HEAD and the comparison is a tautology.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DIFF_LINES = 40


def git(*args: str, strip: bool = True) -> str:
    done = subprocess.run(("git", "-C", str(ROOT), *args), capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout.strip() if strip else done.stdout.rstrip("\n")


def declared_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as source:
        version = tomllib.load(source)["project"]["version"]
    if not isinstance(version, str):
        raise ValueError("pyproject project.version is not a string")
    return version


def red(message: str) -> int:
    print(f"released version guard RED: {message}", file=sys.stderr)
    return 1


def main() -> int:
    version = declared_version()
    tag = f"v{version}"

    # A guard that cannot see the released tags must refuse, not pass: a shallow or
    # tagless checkout is otherwise indistinguishable from a repository that has never
    # released anything.
    if git("rev-parse", "--is-shallow-repository") == "true":
        return red(
            "this is a shallow clone, so the released tags are absent and nothing can be\n"
            "  compared. Check the repository out with `fetch-depth: 0`."
        )

    released = git("tag", "--list", "--sort=v:refname", "v*").split()
    if not released:
        print(f"released version guard green: nothing is released yet, so {version} names no tree")
        return 0
    if tag not in released:
        print(
            f"released version guard green: {version} is unreleased and free to change "
            f"({len(released)} released, latest {released[-1]})"
        )
        return 0

    head_tree = git("rev-parse", "HEAD^{tree}")
    tag_tree = git("rev-parse", f"{tag}^{{tree}}")
    if head_tree == tag_tree:
        print(f"released version guard green: HEAD is exactly the tree {tag} names")
        return 0

    changed = git("diff", "--stat", tag, "HEAD", strip=False).splitlines()
    if len(changed) > DIFF_LINES:
        changed = [*changed[: DIFF_LINES - 1], f"  ... {len(changed) - DIFF_LINES + 1} more lines"]
    detail = "\n".join(f"    {line}" for line in changed)
    return red(
        f"{tag} is already released, and this tree is NOT the tree it names.\n"
        f"  {tag:<12} commit {git('rev-parse', '--short', tag + '^{commit}')}  tree {tag_tree}\n"
        f"  {'HEAD':<12} commit {git('rev-parse', '--short', 'HEAD')}  tree {head_tree}\n"
        f"  changed under the released number:\n{detail}\n"
        f"  Fix: bump `version` in pyproject.toml and re-run `uv lock`, which records it too.\n"
        f"  Never re-point {tag} — a released version number names one tree, forever."
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(red(str(error))) from error
