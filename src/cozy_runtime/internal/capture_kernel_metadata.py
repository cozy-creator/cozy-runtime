"""Temporary xs-138 read-only CUDA graph metadata, queried only after capture ends."""
from __future__ import annotations

from collections import Counter
from importlib.metadata import version
from typing import Any


def snapshot(raw_graph: int) -> dict[str, Any]:
    """Names and launch geometry only. Never read tensor storage or kernel argument arrays."""
    result: dict[str, Any] = {
        "schema": "xs138.cuda-kernel-metadata/1",
        "complete": False,
        "ordering": "kernel multiset; cuGraphGetNodes enumeration is not execution order",
        "kernel_arguments_read": False,
    }
    kernels: Counter[tuple[Any, ...]] = Counter()
    types: Counter[int] = Counter()
    count = 0
    children = 0
    try:
        # NVIDIA's generated bindings own CUDA_KERNEL_NODE_PARAMS and the versioned symbol.
        # CUDA13.0 headers alias cuGraphKernelNodeGetParams to _v2; do not copy a ctypes layout.
        from cuda.bindings import driver

        result["cuda_bindings"] = version("cuda-bindings")

        def query(name: str, *args: Any) -> tuple[Any, ...]:
            values = getattr(driver, name)(*args)
            status = int(values[0])
            if status:
                raise RuntimeError(f"{name}: CUresult {status}")
            return tuple(values[1:])

        def visit(graph: Any, depth: int) -> None:
            nonlocal count, children
            # Same graph bounds as native/execution/control.cu::inspect.
            if depth > 32:
                raise RuntimeError("graph exceeds native inspection depth32")
            _, expected = query("cuGraphGetNodes", graph, 0)
            expected = int(expected)
            if expected > 1_000_000 - count:
                raise RuntimeError("graph exceeds native inspection node limit")
            nodes, observed = query("cuGraphGetNodes", graph, expected) if expected else ([], 0)
            if int(observed) != expected or len(nodes) != expected:
                raise RuntimeError("graph node count changed during owned metadata query")
            count += expected
            for node in nodes:
                (kind,) = query("cuGraphNodeGetType", node)
                types[int(kind)] += 1
                if kind == driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_GRAPH:
                    (child,) = query("cuGraphChildGraphNodeGetGraph", node)
                    children += 1
                    visit(child, depth + 1)
                elif kind == driver.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL:
                    (params,) = query("cuGraphKernelNodeGetParams", node)
                    if int(params.func):
                        (name,) = query("cuFuncGetName", params.func)
                    elif int(params.kern):
                        (name,) = query("cuKernelGetName", params.kern)
                    else:
                        raise RuntimeError("kernel node exposes neither function nor kernel handle")
                    if not isinstance(name, bytes):
                        raise RuntimeError("CUDA binding returned an unexpected kernel-name type")
                    # kernelParams/extra and their pointed-to argument memory are never accessed.
                    key = (
                        depth, name.decode("utf-8", errors="replace"),
                        int(params.gridDimX), int(params.gridDimY), int(params.gridDimZ),
                        int(params.blockDimX), int(params.blockDimY), int(params.blockDimZ),
                        int(params.sharedMemBytes),
                    )
                    kernels[key] += 1

        visit(driver.CUgraph(raw_graph), 0)
        result["complete"] = True
    except Exception as error:
        # Diagnostic failure must not change graph admission, replay routing or model arithmetic.
        result["error"] = f"{type(error).__name__}: {error}"
    result.update(
        nodes=count,
        child_graphs=children,
        node_types={str(kind): total for kind, total in sorted(types.items())},
        kernel_nodes=sum(kernels.values()),
        kernels=[
            {"depth": key[0], "name": key[1], "grid": list(key[2:5]),
             "block": list(key[5:8]), "shared_mem_bytes": key[8], "count": total}
            for key, total in sorted(kernels.items())
        ],
    )
    return result
