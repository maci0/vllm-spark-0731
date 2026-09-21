"""Verify the side-by-side b12x install.

Checks that the 1.2.6 package our overlays and vLLM are written against still works
under its own name, and that the reference's 0.15.3 generation is reachable as
b12x_ref with the entry points a port would need.

Run inside the image: python3 check_b12x_side.py
"""

import sys
from importlib.metadata import version

problems = 0


def report(label: str, ok: bool, detail: str = "") -> None:
    global problems
    suffix = f" ({detail})" if detail else ""
    print(f"{'ok  ' if ok else 'FAIL'} {label}{suffix}")
    if not ok:
        problems += 1


OURS = [
    ("b12x.attention.compressed_sparse_mla", ["Caps", "plan", "bind", "run"]),
    ("b12x.attention.dsa_indexer", ["plan_paged_schedule"]),
    ("b12x.gemm.wo_projection", ["Caps", "plan", "bind", "run"]),
]

import b12x

report("b12x kept working", True, b12x.__file__)
try:
    report("b12x version", True, version("b12x"))
except Exception as exc:
    report("b12x version", False, f"{type(exc).__name__}: {exc}")

for module_name, names in OURS:
    try:
        module = __import__(module_name, fromlist=names)
    except Exception as exc:
        report(f"{module_name} kept working", False, f"{type(exc).__name__}: {exc}")
        continue
    missing = [name for name in names if not hasattr(module, name)]
    report(f"{module_name} kept working", not missing, f"missing={missing}" if missing else "")

try:
    import b12x_ref
except Exception as exc:
    report("b12x_ref importable", False, f"{type(exc).__name__}: {exc}")
else:
    report("b12x_ref importable", True, b12x_ref.__file__)

REF_ENTRY_POINTS = [
    ("b12x_ref.attention.mla", "compressed_mla_decode_forward"),
    ("b12x_ref.attention.mla", "sparse_mla_decode_forward"),
    ("b12x_ref.attention.mla", "MLASparseDecodeMetadata"),
    ("b12x_ref.gemm", "wo_projection"),
    ("b12x_ref.attention.indexer", "IndexerPagedTiledLogitsKernelBinding"),
    ("b12x_ref.moe", "fused"),
]

for module_name, attr in REF_ENTRY_POINTS:
    try:
        module = __import__(module_name, fromlist=[attr])
        report(f"{module_name}.{attr}", hasattr(module, attr))
    except Exception as exc:
        report(f"{module_name}.{attr}", False, f"{type(exc).__name__}: {exc}")

print(f"problems={problems}")
sys.exit(1 if problems else 0)
