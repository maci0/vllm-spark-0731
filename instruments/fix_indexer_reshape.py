"""Replace the sync_packed_indexer_k gather so it stops materialising a cache copy.

`flat_tokens = raw.reshape(-1, _TOKEN_BYTES)` is asked for on a strided slice of the
packed cache. Row i of the flattened view is `raw[i // block_size, i % block_size]`,
so the flatten is only ever a way to turn a row index into a (block, offset) pair.
When the slice cannot merge (the trace shows the reshape collapsing into
aten::clone -> aten::copy_), ATen copies the whole page view first.

The replacement is built from the installed source with one text substitution, so
the rest of the body cannot drift from the shipped one, and exec'd against the real
module globals so `_packed_sidecars` stays shared with lookup_packed_indexer_k.

Two nsys captures with identical total device time (62.9 s) show the instantiation
this removes: `elementwise_kernel<128, 4, gpu_kernel_impl_nocast<direct_copy...[unsigned
char]>>` at 9.471 s over 54390 launches (15.05 percent of device time) is present in
the control capture and absent from the fix capture, where ATen elementwise falls from
18.19 to 2.95 percent.

DSV4_FIX_INDEXER_FLAG names a marker file. When set, the fast path is used only while
that file exists, which allows an A/B within one boot, where NCCL variance between
boots (127 us against 479 us per all-reduce in those same two captures) cannot
confound the result.
"""
import inspect
import os
import time

import torch

OLD = """    flat_tokens = raw.reshape(-1, _TOKEN_BYTES)
    tok_idx = clamped.clamp(max=flat_tokens.shape[0] - 1)
    tok = flat_tokens[tok_idx]"""

NEW = """    n_rows = int(raw.shape[0]) * block_size
    tok_idx = clamped.clamp(max=n_rows - 1)
    tok = raw[tok_idx // block_size, tok_idx % block_size]"""


def install(log=print):
    from vllm.utils import sm12x_b12x_kernels as m

    orig = m.sync_packed_indexer_k
    src = inspect.getsource(orig)
    if OLD not in src:
        raise RuntimeError("sync_packed_indexer_k no longer matches the expected body")
    if not src.startswith("def "):
        raise RuntimeError("unexpected source prefix: %r" % src[:40])
    if NEW in src:
        raise RuntimeError("body already carries the replacement")

    ns = m.__dict__
    exec(compile(src.replace(OLD, NEW, 1), "<sync_packed_indexer_k-fixed>", "exec"), ns)
    fixed = ns["sync_packed_indexer_k"]

    checks = int(os.environ.get("DSV4_FIX_INDEXER_CHECK", "3"))
    flag = os.environ.get("DSV4_FIX_INDEXER_FLAG", "")
    state = {"calls": 0, "checked": 0, "mismatch": 0, "arm": None, "t0": time.time()}

    def fast_now():
        if not flag:
            return True
        return os.path.exists(flag)

    def wrapper(kv_cache, slot_mapping):
        state["calls"] += 1
        if flag and state["arm"] != fast_now():
            state["arm"] = fast_now()
            log("[fixreshape] arm=%s at call=%d after %.1fs pid=%d"
                % ("fast" if state["arm"] else "shipped", state["calls"],
                   time.time() - state["t0"], os.getpid()))
        out = fixed(kv_cache, slot_mapping) if fast_now() else orig(kv_cache, slot_mapping)
        if state["checked"] < checks and slot_mapping is not None and slot_mapping.numel():
            state["checked"] += 1
            before = out.clone() if out is not None else None
            again = orig(kv_cache, slot_mapping)
            same = before is not None and again is not None and torch.equal(before, again)
            if not same:
                state["mismatch"] += 1
            log("[fixreshape] check%d identical=%s dim=%d cache=%s %s bs=%d T=%d"
                % (state["checked"], same, kv_cache.dim(), tuple(kv_cache.shape),
                   kv_cache.stride(), int(kv_cache.shape[1]), int(slot_mapping.numel())))
        if state["calls"] % 2048 == 0:
            log("[fixreshape] calls=%d mismatches=%d after %.1fs pid=%d"
                % (state["calls"], state["mismatch"], time.time() - state["t0"],
                   os.getpid()))
        return out

    m.sync_packed_indexer_k = wrapper
    log("[fixreshape] installed pid=%d checks=%d" % (os.getpid(), checks))
    return fixed
