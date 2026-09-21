"""Check the exact b12x surface this repo's code imports against whatever b12x is installed.

Run inside an image: python3 check_b12x_api.py
Exit code is the number of failures, so it works as a gate.
"""

import sys

CHECKS = [
    ("b12x.attention.compressed_sparse_mla", ["Caps", "plan", "bind", "run"]),
    ("b12x.attention.dsa_indexer", ["plan_paged_schedule"]),
    ("b12x.gemm.wo_projection", ["Caps", "plan", "bind", "run"]),
]

failures = 0
for module_name, names in CHECKS:
    try:
        module = __import__(module_name, fromlist=names)
    except Exception as exc:
        print(f"MODULE FAIL {module_name}: {type(exc).__name__}: {exc}")
        failures += 1
        continue
    missing = [name for name in names if not hasattr(module, name)]
    if missing:
        print(f"NAME FAIL {module_name}: missing {missing}")
        failures += len(missing)
    else:
        print(f"ok {module_name}")

import b12x  # noqa: E402

print("b12x file:", getattr(b12x, "__file__", "?"))
try:
    from importlib.metadata import version

    print("b12x version:", version("b12x"))
except Exception as exc:
    print("b12x version unknown:", exc)
print(f"failures={failures}")
sys.exit(1 if failures else 0)
