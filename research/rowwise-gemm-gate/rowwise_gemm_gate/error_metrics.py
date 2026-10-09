"""Post-timing numerical diagnostics; never changes the strict exact-byte gate."""


def compare(torch, actual, reference, tile_rows=None):
    finite = torch.ones((), device=actual.device, dtype=torch.bool)
    sums = torch.zeros(2, device=actual.device, dtype=torch.float64)
    maximum = torch.zeros((), device=actual.device, dtype=torch.float64)
    different = torch.zeros((), device=actual.device, dtype=torch.int64)
    max_ulp = torch.zeros((), device=actual.device, dtype=torch.int32)
    ulp_counts = torch.zeros(3, device=actual.device, dtype=torch.int64)
    tail_different = torch.zeros((), device=actual.device, dtype=torch.int64)
    examples = []
    tail_start = actual.shape[0] - actual.shape[0] % tile_rows if tile_rows else actual.shape[0]
    for start in range(0, actual.shape[0], 128):
        a, b = actual[start : start + 128], reference[start : start + 128]
        finite.logical_and_(torch.isfinite(a).all() & torch.isfinite(b).all())
        mismatch = a != b
        different.add_(mismatch.sum())
        delta = a.double() - b.double()
        sums[0].add_(delta.square().sum())
        sums[1].add_(b.double().square().sum())
        maximum = torch.maximum(maximum, delta.abs().max())
        # This ordering makes adjacent finite BF16 values one unit apart and
        # treats positive/negative zero equally for ULP distance, unlike bytes.
        ai, bi = a.view(torch.int16).to(torch.int32), b.view(torch.int16).to(torch.int32)
        ai = torch.where(ai < 0, -32768 - ai, ai)
        bi = torch.where(bi < 0, -32768 - bi, bi)
        ulp = (ai - bi).abs()
        max_ulp = torch.maximum(max_ulp, ulp.max())
        for index, mask in enumerate((ulp == 1, ulp == 2, ulp > 2)):
            ulp_counts[index].add_(mask.sum())
        if start + len(a) > tail_start:
            tail_different.add_(mismatch[max(0, tail_start - start) :].sum())
        if len(examples) < 16:
            for row, column in mismatch.nonzero()[: 16 - len(examples)].tolist():
                examples.append(
                    {
                        "row": start + row,
                        "column": column,
                        "actual": float(a[row, column].item()),
                        "reference": float(b[row, column].item()),
                        "bf16_ulp": int(ulp[row, column].item()),
                    }
                )
    numerator, denominator = sums.tolist()
    count, tail = int(different.item()), int(tail_different.item())
    return {
        "bytes_equal": bool(torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))),
        "finite": bool(finite.item()),
        "max_abs": float(maximum.item()),
        "different_elements": count,
        "relative_l2": (numerator / denominator) ** 0.5
        if denominator
        else (0.0 if numerator == 0 else None),
        "bf16_ulp_max": int(max_ulp.item()),
        "bf16_ulp_counts": dict(
            zip(("1", "2", "greater_than_2"), ulp_counts.tolist(), strict=True)
        ),
        "difference_examples": examples,
        "tail_start_row": tail_start if tile_rows else None,
        "tail_different_elements": tail if tile_rows else None,
        "full_tile_different_elements": count - tail if tile_rows else None,
        "acceptance": "Informational errors only; exact-byte admission is unchanged",
    }
