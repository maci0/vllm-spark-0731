
### c6 split, and the sparse-MLA autotune asymmetry checked and closed, 2026-09-13

With the profiler window in the c6 region the sample lines read: `execute_model` gpu 97 to 108 ms,
`sample_tokens` gpu 22.7 to 37.3 ms, `sample` gpu 2.5 to 3.8 ms, against a c6 step interval of
38.65 ms. So about three engine steps of GPU work are in flight at once at c6, and the level that the
completion criterion judges is throughput-bound on total GPU work rather than on per-layer latency.
That is why the c1 region split did not predict the c6 result: at c1 the MoE is 65 % of the *layer
latency*, at c6 it is a much smaller share of the *layer work*.

`VLLM_PROFILE_DECODE_STEPS` did not take effect: the window still reports `steps=12`, so the limit
is not reaching the worker even though `VLLM_PROFILE_DECODE` and `VLLM_PROFILE_CAPTURE` do. Recorded
as an instrument limitation rather than worked around.

One asymmetry looked promising and is now closed. The reference runs a FlashInfer SM120 sparse MLA
decode autotune and caches 24 configs, then hits that cache at runtime:

    [flashinfer_sparse_mla_warmup.py:124] Autotuning FlashInfer SM120 sparse MLA DSv4 decode with cache: ...
    [Autotuner]: Loaded 24 configs ...
    [Autotuner]: Config cache hit for sparse_mla_sm120_decode_dsv4 (runner=SparseMlaDecodeV3Runner)

Our log carries that autotune **zero** times, so our sparse MLA decode runs at defaults. The gate is
in `flashinfer_sparse_mla_warmup.py:225`: the autotune runs only when the attention backend's
`get_name()` is one of the DeepSeek-V4 sparse MLA backends, and `b12x_sparse.py` reports
`B12X_MLA_SPARSE`, which is not in that set. So with the default backend we skip it.

It still does not explain the gap, because the arm that passes that gate measured a wash: `attnfi`,
which sets `ATTENTION_BACKEND` and `DRAFT_ATTENTION_BACKEND` to `FLASHINFER_MLA_SPARSE_DSV4` and so
does run the autotune, gave 37.8 / 78.5 / 104.6 / 119.8 against 38.4 / 80.8 / 105.2 / 122.3. Tuned
FlashInfer sparse MLA lands where our untuned b12x MLA already is. So attention is out on both the
implementation and the tuning axis.
