"""Validate the LD_PRELOAD interposer in isolation, before it goes near vLLM.

Runs a few large 1-byte elementwise operations on the GPU, which are the shape the interposer
is meant to catch, and prints whether the process survived.
"""
import torch

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
dev = "cuda"
big = torch.empty((13718, 64, 132), dtype=torch.uint8, device=dev)
dst = torch.empty_like(big)

# A large contiguous 1-byte copy: blockX 128, grid in the hundreds of thousands.
dst.copy_(big)
torch.cuda.synchronize()
print("copy_ ok")

# Advanced indexing into the same buffer, the scatter shape seen in sync_packed_indexer_k.
idx = torch.arange(0, big.shape[0], 7, device=dev, dtype=torch.int64)
sliced = big[idx]
torch.cuda.synchronize()
print("index_select ok", tuple(sliced.shape))

# A strided slice, which earlier bench_copy measured at 722 us.
tmp = torch.empty((48, 64, 64), dtype=torch.uint8, device=dev)
tmp.copy_(big[:48, :, :64])
torch.cuda.synchronize()
print("strided copy ok")
