#!/usr/bin/env python3
"""Validate the current pool's measured GPU caps from public TensorD tracing logs."""

import argparse
import json
import re
import sys


def events(text):
    """Read tracing JSON, including JSON-string log envelopes printed by the public CLI."""
    try:
        value = json.loads(text)
    except ValueError:
        for line in text.splitlines():
            start = line.find("{")
            if start < 0:
                continue
            try:
                value, _ = json.JSONDecoder().raw_decode(line[start:])
            except ValueError:
                continue
            yield from objects(value)
    else:
        yield from objects(value)


def objects(value):
    if isinstance(value, dict):
        fields = value.get("fields")
        if isinstance(fields, dict) and fields.get("message") == "GPU memory capacity":
            yield fields
        else:
            for child in value.values():
                if isinstance(child, str):
                    yield from events(child)
                elif isinstance(child, (dict, list)):
                    yield from objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from objects(child)


def measured_caps(text, requested, expected):
    if type(requested) is not int or requested <= 0 or type(expected) is not int or expected <= 0:
        raise ValueError("a positive requested byte cap and assigned GPU count are required")
    latest = {}
    for row in events(text):
        gpu = row.get("gpu")
        if (
            not isinstance(gpu, str)
            or re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", gpu)
            is None
        ):
            raise ValueError("capacity evidence has no physical GPU UUID")
        # TensorD's log stream is append ordered; a later pool replaces this UUID's row.
        latest[gpu.lower()] = row
    if len(latest) != expected:
        raise ValueError(f"capacity evidence covers {len(latest)} GPUs, expected {expected}")
    generations = set()
    for row in latest.values():
        generation = row.get("generation")
        if not isinstance(generation, str) or re.fullmatch(r"[0-9a-f]{32}", generation) is None:
            raise ValueError("capacity evidence has no pool generation")
        generations.add(generation)
        physical, configured, effective = (
            row.get(key)
            for key in ("physical_bytes", "configured_limit_bytes", "effective_limit_bytes")
        )
        if (
            any(type(value) is not int for value in (physical, configured, effective))
            or physical <= 0
            or configured != requested
            or effective != min(requested, physical)
        ):
            raise ValueError(f"GPU {row['gpu']} has unproved or incorrect effective capacity")
    if len(generations) != 1:
        raise ValueError(
            "assigned GPUs do not have complete coverage from the same pool generation"
        )
    return list(latest.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gib", type=int, required=True)
    parser.add_argument("--gpus", type=int, required=True)
    args = parser.parse_args()
    try:
        rows = measured_caps(sys.stdin.read(), args.gib << 30, args.gpus)
    except ValueError as error:
        print(f"GPU cap not proved: {error}", file=sys.stderr)
        return 1
    for row in rows:
        print(f"GPU cap proved: {json.dumps(row, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
