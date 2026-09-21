#!/usr/bin/env python3
"""Compare the MoE MAX_ACTIVE_CLUSTERS tuning coverage of the two b12x generations.

The reference's b12x carries three decode regimes, ``micro`` / ``static`` /
``dynamic``, each with a generated ladder from routed rows to a cluster cap. Our
generation carries ``micro`` and ``dynamic``. A decode step's routed-row count is
rows * topk, which for DSV4-Flash lands in the hundreds, so the question is
whether the band the reference tunes with ``static`` is covered on our side.

Run inside ``vllm-spark-0731:main-029-proto-b12xref``:

    docker run --rm --gpus all --entrypoint python3 -v ~/bench:/bench \
        vllm-spark-0731:main-029-proto-b12xref /bench/probe_moe_tuning.py
"""

from __future__ import annotations

import json

ROWS = (8, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 640, 1024)


def look(table):
    def f(backend, rows):
        try:
            return table.lookup_max_active_clusters(
                regime="decode", backend=backend, routed_rows=rows
            )
        except Exception:  # noqa: BLE001
            return None
    return f


def main():
    import b12x.moe as ours_moe
    import b12x_ref.moe as ref_moe
    import b12x.moe._shared.tuning as ours
    import b12x_ref.moe.tuning as ref

    ours_mods = [n for n in dir(ours_moe.fused_moe) if "static" in n.lower()]
    ref_mods = [n for n in dir(ref_moe.fused) if "static" in n.lower()]
    print(json.dumps({"event": "static_kernel_module",
                      "ours": ours_mods, "ref": ref_mods}), flush=True)
    print(json.dumps({"event": "policies",
                      "ours": sorted(f"{a}/{b}" for a, b in ours.MAX_ACTIVE_CLUSTERS_POLICY),
                      "ref": sorted(f"{a}/{b}" for a, b in ref.MAX_ACTIVE_CLUSTERS_POLICY)}),
          flush=True)

    fo, fr = look(ours), look(ref)
    print("routed_rows  ours.micro  ours.dynamic  ref.micro  ref.static  ref.dynamic", flush=True)
    for rows in ROWS:
        cells = [
            fo("micro", rows), fo("dynamic", rows),
            fr("micro", rows), fr("static", rows), fr("dynamic", rows),
        ]
        print(
            f"{rows:11d}  " + "  ".join(f"{'-' if c is None else c:>10}" for c in cells),
            flush=True,
        )

    # Where a DSV4-Flash decode step lands. docs/knowledge/02-model.md: 256 routed
    # experts, 6 per token. q_rows = batch * (k+1) with k=7.
    TOPK = 6
    print(json.dumps({
        "event": "operating_points",
        "note": f"routed_rows = q_rows * topk, topk={TOPK}, q_rows = c * (7+1)",
        "points": [
            {"c": c, "q_rows": c * 8, "routed_rows": c * 8 * TOPK,
             "ours_policy_band": "dynamic (micro ends at 20)",
             "ref_policy_band": "static (20 < rows <= 640)"}
            for c in (1, 3, 5, 6)
        ],
    }), flush=True)


if __name__ == "__main__":
    main()
