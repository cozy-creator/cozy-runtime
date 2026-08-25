#!/usr/bin/env python3
"""Boundary fence (boundaries.md): forbidden-dependency scan. Architecture enforcement, not a test."""
import pathlib, re, sys

DENY = ["tensorhub", "cozy_creator", "cozy-creator-v2", "varena", "runpod", "openrails"]
SCAN = ["pyproject.toml", "src/**/*.py"]

violations = []
for glob in SCAN:
    for p in sorted(pathlib.Path(".").glob(glob)):
        if not p.is_file():
            continue
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            s = line.strip()
            if s.startswith(("#", "//")):
                continue
            for d in DENY:
                if re.search(r"(?<![\w.-])" + re.escape(d) + r"(?![\w-])", s, re.I):
                    violations.append(f"{p}:{i}: forbidden dependency '{d}': {s}")

if violations:
    print("FENCE RED — forbidden dependencies (boundaries.md):", file=sys.stderr)
    for v in violations:
        print("  " + v, file=sys.stderr)
    sys.exit(1)
print(f"fence green ({len(DENY)} forbidden tokens, {len(SCAN)} scan globs)")
