#!/usr/bin/env python3
"""Numerics probes for SM12x DSV4 paths that still emit 的超 loops.

Compares CuteDSL indexer-Q vs Triton vs a Python GPT-J/UE8M0 reference,
and Triton SWA KV insert vs the same reference (footer-scale 584B pages).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
INDEX_H = 64
INDEX_D = 128
INDEX_ROPE = 64
FP8_MAX = 448.0
EPS = 1e-6
BLOCK = 64
BPT = 584
TOKEN_DATA = 576  # 448 fp8 NoPE + 64 bf16 RoPE


def gptj_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """GPT-J interleaved RoPE on the last ROPE_DIM dims. x: [..., HEAD]."""
    nope, rot = x[..., : x.shape[-1] - ROPE_DIM], x[..., -ROPE_DIM:]
    even = rot[..., 0::2]
    odd = rot[..., 1::2]
    new_even = even * cos - odd * sin
    new_odd = even * sin + odd * cos
    rot2 = torch.stack((new_even, new_odd), dim=-1).reshape_as(rot)
    return torch.cat((nope, rot2), dim=-1)


def rmsnorm_no_weight(x: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * rms).to(x.dtype)


def ue8m0_scale(amax: torch.Tensor) -> torch.Tensor:
    scale = torch.maximum(amax, torch.tensor(1e-4, device=amax.device, dtype=amax.dtype))
    scale = scale / FP8_MAX
    return torch.exp2(torch.ceil(torch.log2(scale.clamp(min=2**-126))))


def report(name: str, a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().flatten()
    bf = b.float().flatten()
    cos = F.cosine_similarity(af, bf, dim=0).item()
    mad = (af - bf).abs().max().item()
    print(f"{name}: cosine={cos:.8f} max_abs={mad:.6g}")
    return cos


def indexer_q_python(q, weights, pos, cos_sin, softmax_scale, head_scale):
    T, H, D = q.shape
    rope = INDEX_ROPE
    nope = D - rope
    qf = q.float()
    out = qf.clone()
    for t in range(T):
        c = cos_sin[pos[t], : rope // 2].float()
        s = cos_sin[pos[t], rope // 2 :].float()
        rot = qf[t, :, nope:]
        even = rot[:, 0::2]
        odd = rot[:, 1::2]
        new_even = even * c - odd * s
        new_odd = even * s + odd * c
        rot2 = torch.stack((new_even, new_odd), dim=-1).reshape(H, rope)
        # fp32 -> bf16 -> fp32 before absmax, matching Triton comment
        rot2 = rot2.to(torch.bfloat16).float()
        out[t, :, nope:] = rot2
        out[t, :, :nope] = qf[t, :, :nope]
    amax = out.abs().amax(dim=-1)
    scale = ue8m0_scale(amax)
    q_fp8 = (out / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    w_out = weights.float() * scale * softmax_scale * head_scale
    return q_fp8, w_out


def test_indexer_q() -> None:
    from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
        _fused_indexer_q_rope_quant_kernel,
        fused_indexer_q_rope_quant,
    )
    from vllm.platforms import current_platform
    from vllm.utils.import_utils import has_cutedsl

    torch.manual_seed(0)
    T, H, D = 8, INDEX_H, INDEX_D
    device = "cuda"
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device=device)
    weights = torch.randn(T, H, dtype=torch.bfloat16, device=device)
    pos = torch.arange(T, dtype=torch.int64, device=device)
    cos_sin = torch.randn(4096, INDEX_ROPE, dtype=torch.bfloat16, device=device)
    softmax_scale = D**-0.5
    head_scale = 1.0

    print("has_cutedsl", has_cutedsl(), "fp8", current_platform.fp8_dtype())

    q_ref, w_ref = indexer_q_python(q, weights, pos, cos_sin, softmax_scale, head_scale)

    q_cute, w_cute = fused_indexer_q_rope_quant(
        pos, q.clone(), cos_sin, weights.clone(), softmax_scale, head_scale, use_fp4=False
    )

    fp8_dtype = current_platform.fp8_dtype()
    q_triton = torch.empty_like(q, dtype=fp8_dtype)
    w_triton = torch.empty_like(weights, dtype=torch.float32)
    _fused_indexer_q_rope_quant_kernel[(T, H)](
        pos,
        q,
        q.stride(0),
        q.stride(1),
        cos_sin,
        cos_sin.stride(0),
        cos_sin.shape[-1] // 2,
        q_triton,
        q_triton.stride(0),
        q_triton.stride(1),
        D,
        weights,
        weights.stride(0),
        softmax_scale,
        head_scale,
        w_triton,
        w_triton.stride(0),
        FP8_MAX=FP8_MAX,
        USE_FNUZ=False,
        num_warps=1,
    )

    q_cute_f = q_cute.float()
    q_triton_f = q_triton.float()
    q_ref_f = q_ref.float()
    print("--- indexer Q ---")
    report("cutedsl vs python Q", q_cute_f, q_ref_f)
    report("triton vs python Q", q_triton_f, q_ref_f)
    report("cutedsl vs triton Q", q_cute_f, q_triton_f)
    report("cutedsl vs python W", w_cute, w_ref)
    report("triton vs python W", w_triton, w_ref)
    report("cutedsl vs triton W", w_cute, w_triton)
    cute_u8 = q_cute.view(torch.uint8)
    triton_u8 = q_triton.view(torch.uint8)
    print("cutedsl/triton fp8 byte agree", (cute_u8 == triton_u8).float().mean().item())


def python_qnorm_rope(q, kv, pos, cos_sin, eps):
    """q: [T,H,512] kv: [T,512]. Returns roped q and kv."""
    T, H, _ = q.shape
    qn = rmsnorm_no_weight(q, eps)
    q_out = qn.clone()
    kv_out = kv.clone()
    for t in range(T):
        c = cos_sin[pos[t], : ROPE_DIM // 2].float()
        s = cos_sin[pos[t], ROPE_DIM // 2 :].float()
        q_out[t] = gptj_rope(qn[t].float(), c, s).to(q.dtype)
        kv_out[t] = gptj_rope(kv[t].float(), c, s).to(kv.dtype)
    return q_out, kv_out


def python_footer_insert(kv_roped: torch.Tensor, cache: torch.Tensor, slot, block_size: int):
    """Write 448B fp8 NoPE + 128B bf16 RoPE, then 8B UE8M0 scales at block footer."""
    T = slot.shape[0]
    n_blocks = cache.shape[0]
    cache_2d = cache.view(n_blocks, -1)
    nope = kv_roped[:, :NOPE_DIM].float()
    rope = kv_roped[:, NOPE_DIM:].contiguous()
    # 7 quant blocks of 64 on 448 NoPE dims
    n_q = NOPE_DIM // 64
    scales = []
    fp8_parts = []
    for i in range(n_q):
        chunk = nope[:, i * 64 : (i + 1) * 64]
        amax = chunk.abs().amax(dim=-1)
        sc = ue8m0_scale(amax)
        scales.append(sc)
        fp8_parts.append((chunk / sc.unsqueeze(-1)).to(torch.float8_e4m3fn))
    fp8 = torch.cat(fp8_parts, dim=-1)
    scale_u8 = torch.stack(scales, dim=-1)  # [T, 7]
    # pack each scale as UE8M0 exponent byte: log2(scale)+127
    exp = (torch.log2(scale_u8.clamp(min=2**-126)) + 127).round().to(torch.uint8)
    pad = torch.zeros(T, 1, dtype=torch.uint8, device=kv_roped.device)
    scale_bytes = torch.cat([exp, pad], dim=-1)  # 8 bytes

    for t in range(T):
        s = int(slot[t].item())
        blk = s // block_size
        off = s % block_size
        row = cache_2d[blk]
        data_off = off * TOKEN_DATA
        row[data_off : data_off + NOPE_DIM] = fp8[t].view(torch.uint8)
        rope_bytes = rope[t].view(torch.uint8)
        row[data_off + NOPE_DIM : data_off + TOKEN_DATA] = rope_bytes
        scale_off = block_size * TOKEN_DATA + off * 8
        row[scale_off : scale_off + 8] = scale_bytes[t]


def test_kv_insert() -> None:
    from vllm.models.deepseek_v4.xpu.xpu_qnorm_rope_kv_fp8_insert import (
        xpu_qnorm_rope_kv_fp8_insert,
    )

    torch.manual_seed(1)
    T, H = 4, 32
    device = "cuda"
    q = torch.randn(T, H, HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv = torch.randn(T, HEAD_DIM, dtype=torch.bfloat16, device=device)
    cache = torch.zeros(4, BLOCK, BPT, dtype=torch.uint8, device=device)
    slot = torch.arange(T, dtype=torch.int64, device=device)
    pos = torch.arange(T, dtype=torch.int64, device=device)
    cos_sin = torch.randn(4096, ROPE_DIM, dtype=torch.float32, device=device)

    q_tri = q.clone()
    cache_tri = cache.clone()
    xpu_qnorm_rope_kv_fp8_insert(q_tri, kv, cache_tri, slot, pos, cos_sin, EPS, BLOCK)
    torch.cuda.synchronize()

    q_ref, kv_ref = python_qnorm_rope(q, kv, pos, cos_sin, EPS)
    cache_ref = cache.clone()
    python_footer_insert(kv_ref, cache_ref, slot, BLOCK)

    print("--- SWA KV insert block_size=64 ---")
    report("Q after insert vs python", q_tri, q_ref)
    # compare NoPE fp8 bytes and rope bf16 bytes of token 0
    c_tri = cache_tri.view(4, -1)
    c_ref = cache_ref.view(4, -1)
    report("cache bytes vs python", c_tri.float(), c_ref.float())
    agree = (c_tri == c_ref).float().mean().item()
    print("cache byte agree", agree)
    # token 0 data: first 448 vs python
    print("tok0 nope u8 agree", (c_tri[0, :NOPE_DIM] == c_ref[0, :NOPE_DIM]).float().mean().item())
    print(
        "tok0 rope u8 agree",
        (c_tri[0, NOPE_DIM:TOKEN_DATA] == c_ref[0, NOPE_DIM:TOKEN_DATA]).float().mean().item(),
    )
    scale_off = BLOCK * TOKEN_DATA
    print(
        "tok0 scale u8",
        c_tri[0, scale_off : scale_off + 8].tolist(),
        "ref",
        c_ref[0, scale_off : scale_off + 8].tolist(),
    )


def dequant_footer_page(page: torch.Tensor, n_tokens: int, block_size: int) -> torch.Tensor:
    """page: [block_bytes] uint8 footer-scale DSV4."""
    rows = []
    for off in range(n_tokens):
        data = page[off * TOKEN_DATA : (off + 1) * TOKEN_DATA]
        fp8 = data[:NOPE_DIM].view(torch.float8_e4m3fn).float()
        rope = data[NOPE_DIM:TOKEN_DATA].contiguous().view(torch.bfloat16).float()
        sc_off = block_size * TOKEN_DATA + off * 8
        exp = page[sc_off : sc_off + 7].float()
        sc = torch.exp2(exp - 127.0)
        nope = torch.cat([fp8[i * 64 : (i + 1) * 64] * sc[i] for i in range(7)])
        rows.append(torch.cat([nope, rope]))
    return torch.stack(rows)


def test_flashinfer_dsv4() -> None:
    from vllm.models.deepseek_v4.xpu.xpu_qnorm_rope_kv_fp8_insert import (
        xpu_qnorm_rope_kv_fp8_insert,
    )
    from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_sparse_mla_dsv4

    torch.manual_seed(2)
    T, H = 8, 32
    device = "cuda"
    q = torch.randn(T, H, HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv = torch.randn(T, HEAD_DIM, dtype=torch.bfloat16, device=device)
    cache = torch.zeros(2, BLOCK, BPT, dtype=torch.uint8, device=device)
    slot = torch.arange(T, dtype=torch.int64, device=device)
    pos = torch.arange(T, dtype=torch.int64, device=device)
    cos_sin = torch.randn(4096, ROPE_DIM, dtype=torch.float32, device=device)
    xpu_qnorm_rope_kv_fp8_insert(q, kv, cache, slot, pos, cos_sin, EPS, BLOCK)
    torch.cuda.synchronize()

    k_ref = dequant_footer_page(cache.view(2, -1)[0], T, BLOCK)
    q_last = q[-1].float()  # insert mutated q in-place
    scale = HEAD_DIM**-0.5
    scores = torch.einsum("hd,sd->hs", q_last, k_ref) * scale
    attn = torch.softmax(scores, dim=-1)
    out_k = torch.einsum("hs,sd->hd", attn, k_ref)
    out_nope = torch.einsum("hs,sd->hd", attn, k_ref[:, :NOPE_DIM])

    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    swa = cache.unsqueeze(-2)  # NHD [pages, block, 1, 584]
    q_in = q[-1:].contiguous()  # [1, H, 512]
    out = torch.empty(1, H, HEAD_DIM, dtype=torch.bfloat16, device=device)
    topk = 128
    indices = torch.full((1, topk), -1, dtype=torch.int32, device=device)
    indices[0, :T] = torch.arange(T, dtype=torch.int32, device=device)
    lens = torch.tensor([T], dtype=torch.int32, device=device)
    print("--- FlashInfer DSV4 vs python (V=K and V=NoPE) ---")
    print("dispatch (H,topk)", (H, topk), "in table", (H, topk) in {(32, 128)})
    try:
        flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
            query=q_in,
            swa_kv_cache=swa,
            workspace_buffer=workspace,
            sparse_indices=indices,
            compressed_kv_cache=None,
            out=out,
            bmm1_scale=scale,
            sinks=None,
            kv_layout="NHD",
            swa_topk_lens=lens,
        )
        torch.cuda.synchronize()
        of = out[0].float()
        report("FI vs python V=K", of, out_k)
        # pad nope to 512 for shape match
        nope_pad = torch.zeros_like(out_k)
        nope_pad[:, :NOPE_DIM] = out_nope
        report("FI vs python V=NoPE-pad", of, nope_pad)
        report("FI vs python V=NoPE on 448", of[:, :NOPE_DIM], out_nope)
        print("FI out rms", of.pow(2).mean().sqrt().item(), "ref V=K rms", out_k.pow(2).mean().sqrt().item())
        print("FI sample", of[0, :8].tolist())
        print("ref V=K sample", out_k[0, :8].tolist())
    except Exception as e:
        print("flashinfer decode failed:", type(e).__name__, e)


def test_inv_rope_fp8() -> None:
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
        fused_inv_rope_fp8_quant,
    )

    torch.manual_seed(3)
    T, G, Hpg = 8, 4, 8
    H = G * Hpg
    device = "cuda"
    o = torch.randn(T, H, HEAD_DIM, dtype=torch.bfloat16, device=device)
    pos = torch.arange(T, dtype=torch.int64, device=device)
    cos_sin = torch.randn(4096, ROPE_DIM, dtype=torch.float32, device=device)

    o_fp8, o_scale = fused_inv_rope_fp8_quant(
        o,
        pos,
        cos_sin,
        n_groups=G,
        heads_per_group=Hpg,
        nope_dim=NOPE_DIM,
        rope_dim=ROPE_DIM,
        tma_aligned_scales=False,
    )
    print(
        "inv-rope fp8",
        tuple(o_fp8.shape),
        o_fp8.dtype,
        "scale",
        tuple(o_scale.shape),
        o_scale.dtype,
    )

    inv = o.float().clone()
    half = ROPE_DIM // 2
    for t in range(T):
        c = cos_sin[pos[t], :half]
        s = cos_sin[pos[t], half:]
        rot = inv[t, :, NOPE_DIM:]
        even = rot[:, 0::2]
        odd = rot[:, 1::2]
        new_even = even * c + odd * s
        new_odd = odd * c - even * s
        rot2 = torch.stack((new_even, new_odd), dim=-1).reshape(H, ROPE_DIM)
        inv[t, :, NOPE_DIM:] = rot2

    packed = o_fp8.float()
    sc = o_scale
    print("packed", tuple(packed.shape), "scale", tuple(sc.shape))
    if sc.shape[0] == G:
        sc = sc.transpose(0, 1)
    nblk = HEAD_DIM // 128
    packed_h = packed.reshape(T, G, Hpg, nblk, 128)
    sc_h = sc.reshape(T, G, Hpg, nblk)
    dq = (packed_h * sc_h.unsqueeze(-1)).reshape(T, H, HEAD_DIM)
    print("--- inv-rope + FP8 dequant vs python inverse RoPE ---")
    report("full head", dq, inv)
    report("nope only", dq[..., :NOPE_DIM], inv[..., :NOPE_DIM])
    report("rope only", dq[..., NOPE_DIM:], inv[..., NOPE_DIM:])


def test_flashinfer_extra() -> None:
    from vllm.models.deepseek_v4.xpu.xpu_qnorm_rope_kv_fp8_insert import (
        xpu_qnorm_rope_kv_fp8_insert,
    )
    from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_sparse_mla_dsv4

    torch.manual_seed(2)
    T, H = 8, 32
    device = "cuda"
    q = torch.randn(T, H, HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv = torch.randn(T, HEAD_DIM, dtype=torch.bfloat16, device=device)
    cache = torch.zeros(2, BLOCK, BPT, dtype=torch.uint8, device=device)
    extra = torch.zeros(2, BLOCK, BPT, dtype=torch.uint8, device=device)
    slot = torch.arange(T, dtype=torch.int64, device=device)
    pos = torch.arange(T, dtype=torch.int64, device=device)
    cos_sin = torch.randn(4096, ROPE_DIM, dtype=torch.float32, device=device)
    xpu_qnorm_rope_kv_fp8_insert(q, kv, cache, slot, pos, cos_sin, EPS, BLOCK)
    extra.copy_(cache)
    torch.cuda.synchronize()

    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    swa = cache.unsqueeze(-2)
    extra_p = extra.unsqueeze(-2)
    q_in = q[-1:].contiguous()
    topk = 128
    indices = torch.full((1, topk), -1, dtype=torch.int32, device=device)
    indices[0, :T] = torch.arange(T, dtype=torch.int32, device=device)
    lens = torch.tensor([T], dtype=torch.int32, device=device)
    extra_topk = 512
    extra_idx_neg = torch.full((1, extra_topk), -1, dtype=torch.int32, device=device)
    extra_len0 = torch.tensor([0], dtype=torch.int32, device=device)
    extra_idx_one = extra_idx_neg.clone()
    extra_idx_one[0, 0] = 0
    extra_len1 = torch.tensor([1], dtype=torch.int32, device=device)

    def run(extra_cache, extra_i, extra_l):
        out = torch.empty(1, H, HEAD_DIM, dtype=torch.bfloat16, device=device)
        flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
            query=q_in,
            swa_kv_cache=swa,
            workspace_buffer=workspace,
            sparse_indices=indices,
            compressed_kv_cache=extra_cache,
            out=out,
            bmm1_scale=HEAD_DIM**-0.5,
            sinks=None,
            kv_layout="NHD",
            swa_topk_lens=lens,
            extra_sparse_indices=extra_i,
            extra_sparse_topk_lens=extra_l,
        )
        torch.cuda.synchronize()
        return out.float()

    print("--- FlashInfer extra compressed indices ---")
    out_swa = run(None, None, None)
    out_neg = run(extra_p, extra_idx_neg, extra_len0)
    out_one = run(extra_p, extra_idx_one, extra_len1)
    report("SWA-only vs extra all-1 lens=0", out_swa, out_neg)
    report("SWA-only vs extra 1 valid", out_swa, out_one)
    print("swa rms", out_swa.pow(2).mean().sqrt().item(), "neg rms", out_neg.pow(2).mean().sqrt().item(), "one rms", out_one.pow(2).mean().sqrt().item())


def main() -> int:
    torch.cuda.init()
    print("device", torch.cuda.get_device_name(0), "mem", torch.cuda.mem_get_info())
    test_indexer_q()
    test_kv_insert()
    test_flashinfer_dsv4()
    test_inv_rope_fp8()
    test_flashinfer_extra()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
