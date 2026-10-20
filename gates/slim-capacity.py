#!/usr/bin/env python3
"""Validate the current pool's measured GPU caps from public TensorD tracing logs."""

import argparse
import json
import re
import sys


TIMESTAMP = r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z"
PLAIN = re.compile(
    TIMESTAMP + r"[ \t]+INFO tensord::gpu_service: GPU memory capacity(?:[ \t]+(.*))?"
)
NUMBERS = ("physical_bytes", "configured_limit_bytes", "effective_limit_bytes")


def plain_row(line):
    match = PLAIN.fullmatch(line)
    if match is None:
        return None
    fields = {}
    for term in (match[1] or "").split():
        key, separator, value = term.partition("=")
        if not separator or key in fields:
            raise ValueError("malformed native GPU capacity fields")
        fields[key] = value
    if set(fields) != {"gpu", "generation", *NUMBERS}:
        raise ValueError("incomplete native GPU capacity fields")
    for key in NUMBERS:
        if re.fullmatch(r"[0-9]+", fields[key]) is None:
            raise ValueError("invalid native GPU capacity number")
        fields[key] = int(fields[key])
    return {"message": "GPU memory capacity", **fields}


def events(text):
    """Read whole native tracing rows or the public CLI's TensorD log envelope."""
    try:
        value = json.loads(text)
    except ValueError:
        for line in text.splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                row = plain_row(line)
                if row is not None:
                    yield row
            else:
                yield from objects(value)
    else:
        yield from objects(value)


def objects(value):
    if isinstance(value, dict):
        fields = value.get("fields")
        if isinstance(fields, dict):
            if (
                value.get("target") == "tensord::gpu_service"
                and value.get("level") == "INFO"
                and fields.get("message") == "GPU memory capacity"
                and isinstance(value.get("timestamp"), str)
                and re.fullmatch(TIMESTAMP, value["timestamp"]) is not None
            ):
                yield fields
            # Other tracing messages are opaque, including captured executor output.
        elif (value.get("log") == "tensord" and isinstance(value.get("machine"), str)
              and value["machine"] and isinstance(value.get("text"), str)):
            yield from events(value["text"])
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
