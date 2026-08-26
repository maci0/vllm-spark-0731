#!/usr/bin/env python3
"""Verification harness for the vllm-spark-0731 docs corpus.

Checks (all must exit 0):
  1. markdown links resolve across README/HANDOFF/docs/patches — except the
     2 documented verbatim-archive exceptions
  2. backticked repo-path references (configs/…, scripts/…, patches/…, docs/…,
     docker/…, tests/…, outputs/…) resolve to existing files or globs
  3. config pin matrix: documented canonical facts match the pin files and
     Dockerfile.main
  4. key numbers and arithmetic: documented identities hold and the numbers
     appear in the corpus
  5. every documented --only overlay name exists in apply_overlays.py
  6. pytest tests/ passes

Run:  python3 scripts/verify-docs.py
Exit 0 = verified (with allowed exceptions), 1 = findings.

Corresponding audit record: outputs/verify-findings.md.
"""

from __future__ import annotations

import glob
import math
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------
# Check 1: markdown link integrity
# --------------------------------------------------------------------------

# Known-broken links that are deliberate: the source repo shipped these broken
# and we keep the field-notes verbatim. (docs/field-notes/README.md explains.)
VERBATIM_ARCHIVE_EXCEPTIONS = {
    ("docs/field-notes/dgx-spark/PROD_C5_SSD.md", "examples/prod-c5-ssd.yaml"),
    ("docs/field-notes/dgx-spark/TROUBLESHOOTING.md", "examples/prod-c5-ssd.yaml"),
}

# Backticked prose refs to files that never existed even in the source repo
# (kept verbatim in the archive; flagged in docs/field-notes/README.md).
BACKTICK_VERBATIM_EXCEPTIONS = {
    ("docs/field-notes/dgx-spark/UPSTREAM_GAPS.md", "examples/prod-c5-ssd.yaml"),
}

# Backticked paths that live in the *upstream vLLM repo*, not this one — the
# docs already say so ("vLLM's …"); deliberately not resolved locally.
KNOWN_EXTERNAL_REFS = {
    ("docs/PLAN-MAIN.md", "docker/Dockerfile"),
    ("docs/UPSTREAM.md", "docker/versions.json"),
}

MD_GLOBS = [
    "README.md",
    "HANDOFF.md",
    "docs/**/*.md",
    "patches/**/*.md",
]


def md_files() -> list[str]:
    out = []
    for g in MD_GLOBS:
        out.extend(glob.glob(os.path.join(ROOT, g), recursive=True))
    return sorted(set(out))


def check_links() -> list[str]:
    findings: list[str] = []
    link_re = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    for path in md_files():
        with open(path, encoding="utf-8") as fh:
            txt = fh.read()
        base = os.path.dirname(path)
        for m in link_re.finditer(txt):
            target = m.group(1).strip()
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            rel = os.path.normpath(os.path.join(base, target.split("#")[0]))
            relroot = os.path.relpath(rel, ROOT)
            if os.path.exists(rel):
                continue
            if (os.path.relpath(path, ROOT), target) in VERBATIM_ARCHIVE_EXCEPTIONS:
                continue
            findings.append(f"broken link: {os.path.relpath(path, ROOT)} -> {target}")
    return findings


# --------------------------------------------------------------------------
# Check 2: backticked repo-path references exist
# --------------------------------------------------------------------------

REPO_TOP_LEVEL = (
    "configs", "scripts", "patches", "docs", "docker", "tests", "outputs",
    "examples", "notes", "papers", "VERSION", "Dockerfile", "Dockerfile.main",
    "HANDOFF.md", "README.md", "LICENSE", ".gitignore", ".dockerignore",
)


def check_path_refs() -> list[str]:
    findings: list[str] = []
    tick_re = re.compile(r"`([^`]+)`")
    link_re = re.compile(r"\[[^\]]*\]\([^)]+\)")
    # Backticked text inside a markdown link label is display text, not a ref.
    for path in md_files():
        with open(path, encoding="utf-8") as fh:
            txt = link_re.sub("", fh.read())
        for m in tick_re.finditer(txt):
            ref = m.group(1).strip()
            first = ref.split("/", 1)[0]
            if first not in REPO_TOP_LEVEL:
                continue
            if any(ch in ref for ch in "<>{} \t"):
                continue  # template / multi-path / prose
            rel = os.path.relpath(path, ROOT)
            if (rel, ref) in BACKTICK_VERBATIM_EXCEPTIONS or (rel, ref) in KNOWN_EXTERNAL_REFS:
                continue
            hits = glob.glob(os.path.join(ROOT, ref))
            if hits:
                continue
            findings.append(f"missing path ref: {rel} -> `{ref}`")
    return findings


# --------------------------------------------------------------------------
# Check 3: config pin matrix vs canonical facts
# --------------------------------------------------------------------------

PIN_DIR = os.path.join(ROOT, "configs")


def parse_pin(name: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(os.path.join(PIN_DIR, name), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"')
            m = re.fullmatch(r"\$\{([A-Z0-9_]+):-?([^}]*)\}", v)
            if m:
                v = m.group(2)  # ${VAR:-default} -> default
            out[k.strip()] = v
    return out


# Canonical facts derived from the audited docs (knowledge/06 serve table,
# knowledge/00-index pins, HANDOFF live table) — the "documented truth".
PIN_MATRIX = {
    "pin.main.env": {
        "IMAGE": "vllm-spark-0731:main-b12x",
        "KV_CACHE_DTYPE": "nvfp4_ds_mla",
        "ATTENTION_BACKEND": "B12X_MLA_SPARSE",
        "DRAFT_ATTENTION_BACKEND": "B12X_MLA_SPARSE",
        "MOE_BACKEND": "b12x",
        "LINEAR_BACKEND": "b12x",
        "GPU_MEMORY_UTILIZATION": "0.8",
        "MAX_NUM_SEQS": "32",
        "DEEPGEMM_COMMIT": "a6b593d2826719dcf4892609af7b84ee23aaf32a",
        "DG_JIT_USE_NVRTC": "0",
    },
    "pin.nvfp4.env": {
        "IMAGE": "vllm-spark-0731:v0.28.0rc2-b12x",
        "KV_CACHE_DTYPE": "nvfp4_ds_mla",
        "ATTENTION_BACKEND": "FLASHINFER_MLA_SPARSE_DSV4",
        "MOE_BACKEND": "b12x",
        "LINEAR_BACKEND": "b12x",
        "CUDAGRAPH_MODE": "PIECEWISE",
    },
    "pin.env": {
        "IMAGE": "vllm-spark-0731:v0.28.0rc2-b12x",
        "KV_CACHE_DTYPE": "fp8_ds_mla",
        "ATTENTION_BACKEND": "FLASHINFER_MLA_SPARSE_DSV4",
        "MOE_BACKEND": "b12x",
        # LINEAR_BACKEND deliberately unset (documented in 06-deployment table)
    },
    "pin.eugr-b12x.env": {
        "IMAGE": "ghcr.io/spark-arena/dgx-vllm-eugr-nightly-b12x:2026081903",
        "KV_CACHE_DTYPE": "fp8_ds_mla",
        "ATTENTION_BACKEND": "B12X_MLA_SPARSE",
        "MOE_BACKEND": "b12x",
        "GPU_MEMORY_UTILIZATION": "0.89",
        "MAX_MODEL_LEN": "1048576",
    },
    "pin.golden.env": {
        "IMAGE": "ghcr.io/anemll/dspark-vllm-gx10:0.1.1",
        "KV_CACHE_DTYPE": "nvfp4_ds_mla",
        "MOE_BACKEND": "flashinfer_b12x",
        "GPU_MEMORY_UTILIZATION": "0.82",
        "MAX_CUDAGRAPH_CAPTURE_SIZE": "36",
        # ATTENTION_BACKEND / LINEAR_BACKEND deliberately unset (image default)
    },
}


def check_pins() -> list[str]:
    findings: list[str] = []
    for pin, expected in PIN_MATRIX.items():
        actual = parse_pin(pin)
        for key, want in expected.items():
            got = actual.get(key)
            if got != want:
                findings.append(f"pin mismatch: {pin} {key}: expected {want!r}, got {got!r}")
    # Dockerfile.main build pins (FROM resolves ${CUDA_IMAGE} via its ARG)
    df = os.path.join(ROOT, "docker", "Dockerfile.main")
    with open(df, encoding="utf-8") as fh:
        dftxt = fh.read()
    args = dict(
        re.findall(r"^ARG\s+([A-Z0-9_]+)=(\S+)", dftxt, re.M)
    )
    for arg, want in [
        ("DEEPGEMM_COMMIT", "a6b593d2826719dcf4892609af7b84ee23aaf32a"),
        ("TORCH_REF", "release/2.14"),
        ("CUTLASS_DSL_VERSION", "4.7.0"),
        ("TRITON_VERSION", "3.7.1"),
    ]:
        got = args.get(arg)
        if got != want:
            findings.append(f"Dockerfile.main ARG {arg}: expected {want!r}, got {got!r}")
    base = re.search(r"^FROM\s+(\S+)", dftxt, re.M)
    got_base = base.group(1) if base else None
    if got_base and got_base.startswith("${"):
        got_base = args.get(got_base.strip("${}"))
    if got_base != "nvidia/cuda:13.3.1-cudnn-devel-ubuntu24.04":
        findings.append(f"Dockerfile.main FROM: expected nvidia/cuda:13.3.1-…, got {got_base!r}")
    return findings


# --------------------------------------------------------------------------
# Check 4: key numbers and arithmetic
# --------------------------------------------------------------------------

# Each entry: (computed_value, expected_value, search_token, doc_that_must_mention_it)
NUMBER_CHECKS = [
    (61 * 584, 35624, "35,624", "docs/knowledge/02-model.md"),
    (7650 / 61, 125.4, "7,650", "docs/knowledge/04-quantization-kv.md"),
    (11317 / 61, 185.5, "11,317", "docs/knowledge/04-quantization-kv.md"),
    (math.ceil(133 / 64) * 64, 192, "192", "docs/knowledge/03-kernels-attention.md"),
    (155.43 / 2, 77.7, "77.7", "docs/knowledge/02-model.md"),
]
CANONICAL_NUMBERS = [
    "97,737", "94,516", "2,047,170", "65.2", "216.8", "26.90", "17.81", "52.12",
]


def check_numbers() -> list[str]:
    findings: list[str] = []
    for value, expected, token, doc in NUMBER_CHECKS:
        if round(value, 1) != expected:  # docs state rounded values
            findings.append(f"arithmetic check FAILS: {token} (computed {value}, doc says {expected})")
        p = os.path.join(ROOT, doc)
        with open(p, encoding="utf-8") as fh:
            if token not in fh.read():
                findings.append(f"missing number token {token!r} in {doc}")
    corpus = ""
    for path in md_files():
        with open(path, encoding="utf-8") as fh:
            corpus += fh.read()
    for num in CANONICAL_NUMBERS:
        if num not in corpus:
            findings.append(f"canonical number {num!r} appears nowhere in the corpus")
    return findings


# --------------------------------------------------------------------------
# Check 5: documented --only overlay names exist in apply_overlays.py
# --------------------------------------------------------------------------

DOCUMENTED_ONLY = [
    # patches/README.md table + knowledge/08 + UPSTREAM backport table
    "b12x-sparse", "o-proj-b12x", "indexer-store-page64", "indexer-b12x-schedule",
    "indexer-mqa", "mqa-packed-gather", "mqa-paged-kernel", "flashinfer-eidx-contig",
    "triton-e8m0-sm12x", "einsum-sm12x", "sm12x-kv-insert", "instanttensor-hybrid",
    "dsv4-block64", "dspark-backbone-cg", "dspark-backbone-none", "ar-piecewise-ws",
]


def check_overlay_names() -> list[str]:
    findings: list[str] = []
    src = os.path.join(ROOT, "patches", "apply_overlays.py")
    with open(src, encoding="utf-8") as fh:
        txt = fh.read()
    m = re.search(r'--only"\s*,\s*choices=\[(.*?)\]', txt, re.S)
    if not m:
        return ["could not parse --only choices from apply_overlays.py"]
    choices = set(re.findall(r'"([a-z0-9-]+)"', m.group(1)))
    for name in DOCUMENTED_ONLY:
        if name not in choices:
            findings.append(f"documented --only name {name!r} missing from apply_overlays.py")
    return findings


# --------------------------------------------------------------------------
# Check 6: pytest
# --------------------------------------------------------------------------


def check_tests() -> list[str]:
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        tail = (res.stdout + res.stderr).strip().splitlines()[-5:]
        return [f"pytest failed ({res.returncode}): " + " | ".join(tail)]
    return []


# --------------------------------------------------------------------------


def main() -> int:
    checks = [
        ("links", check_links),
        ("path refs", check_path_refs),
        ("pin matrix", check_pins),
        ("numbers", check_numbers),
        ("overlay names", check_overlay_names),
        ("pytest", check_tests),
    ]
    total = 0
    for name, fn in checks:
        findings = fn()
        status = "OK" if not findings else f"FAIL ({len(findings)})"
        print(f"[{status}] {name}")
        for f in findings:
            print(f"    - {f}")
        total += len(findings)
    print(f"\n{total} finding(s) total")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
