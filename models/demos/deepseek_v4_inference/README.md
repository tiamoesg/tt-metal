# DeepSeek-V4 Inference (Tenstorrent)

Co-designing a DeepSeek-V4 inference path for the Tenstorrent mesh, targeting
**low-batch / per-user (agentic) serving** — the regime where the mesh's
high-per-user-throughput-at-low-batch profile beats batched-HBM GPUs, *enabled by*
V4's aggressive KV compression keeping long contexts resident.

## Status

| Piece | State |
|---|---|
| `config.py` — V4 config (full + small) | ✅ done |
| `reference_model.py` — **CPU PyTorch reference** (ground truth) | ✅ done, runnable |
| `test_reference.py` — CPU smoke tests | ✅ done |
| `tt/` — TT-NN device implementation | ⏳ next |

This is **phase 1: the runnable reference.** It is architecture-faithful to the
official DeepSeek-V4 `model.py` but strips production concerns so it runs anywhere
(no Tenstorrent device): FP8/FP4 → bf16/fp32, no tensor/expert/pipeline parallelism,
and full-sequence forward instead of the incremental KV-cache decode (generation is
autoregressive-by-recompute). It is the validation target for the TT-NN port — the
same role `reference/` plays for `models/demos/deepseek_v3`.

```bash
# CPU, no device needed:
pytest models/demos/deepseek_v4_inference/test_reference.py
```

## What's implemented (the V4 architecture)

- **Hybrid attention** per layer via `compress_ratios` (`4`=CSA, `128`=HCA, `0`=dense)
- **KV compression**: overlapping two-series (CSA) and single-series (HCA) softmax pooling
- **Lightning indexer + top-k** sparse selection (DeepSeek Sparse Attention)
- **Sliding-window** branch, **attention sink**, **partial RoPE** with the `−i` output trick
- **mHC** residual mixing (`hc_pre`/`hc_post`, Sinkhorn-constrained `B`)
- **DeepSeekMoE** FFN (sqrtsoftplus gate, routed + shared SwiGLU experts)

Correctness-over-speed simplifications (intentional in the reference; replaced in the
TT port): sparse attention is a **dense masked** reference (`[S, G]` keep-mask, no
gather); the `hc_split_sinkhorn` element split is reconstructed and flagged inline —
reconcile with the official `kernel.py` before loading their checkpoints.

## Phase 2 — the TT-NN port plan

Build on `models/demos/deepseek_v3_b1` (the batch-1, per-user decode engine). It
already gives us: pipeline-parallel decode across pods, DRAM-streamed experts,
KV-in-DRAM, CCL reductions, the host pipeline/runner. The new work is the attention
block, validated op-by-op against this reference:

1. **Compressor kernel** (softmax-gated pooling; overlapping for CSA).
2. **Lightning indexer + top-k**, then the **hard one: the sparse gather decode** —
   per query, gather the top-k compressed KV blocks from DRAM and run masked SDPA
   over just them (extend `b1`'s `flash_mla` + `kv_cache_update`).
3. **mHC** cross-layer mixing (changes the `pipeline_block` residual handling).
4. **sqrtsoftplus gate** (reuse `b1`'s `deepseek_moe_gate` + DRAM-streamed experts).
5. **Partial / strided / inverse RoPE** variants.

The reusable C++ math (faithful spec, not drop-in) lives in `tt-train`
(`sources/ttml/modules/{compressed_sparse_attention,heavily_compressed_attention,
lightning_indexer,deepseek_moe,manifold_hyper_connections}`, `optimizers/muon_v4`).

## Reference implementations to build on (github.com/deepseek-ai)

Port the algorithms and credit them (MIT/Apache):

- **DeepSeek-V3.2-Exp (DSA)** — lightning indexer + top-k sparse attention; the
  direct ancestor of CSA. *Primary reference for the sparse-gather kernel.*
- **FlashMLA** — MLA decode kernel (the algorithm behind `b1`'s `flash_mla`).
- **DeepGEMM** — FP8 GEMM with fine-grained scaling (V4 is fp8/fp4).
- **DeepEP** — expert-parallel all-to-all dispatch/combine for MoE.
- **DualPipe** + **profile-data** — bidirectional pipeline comp/comm overlap.
- **EPLB / LPLB** — expert-parallel load balancing (hot-expert skew).
- **TileKernels** — tilelang kernels (V4's own kernels are TileLang).
- **awesome-deepseek-agent** — agent stacks to target for hosting.

## Reality checks (measured from the shipping `deepseek_v3_b1` engine)

- KV lives in **DRAM**, not SRAM — but tiny (MLA latent `1×576`/token). The lever is
  "make KV tiny via compression," which V4 pushes further (~10% of V3.2).
- MoE routed experts are **streamed from DRAM**, not pinned in SRAM (too big at scale).
- The spatial online-softmax accumulation is real (`sdpa_reduce_to_all` over a ring).
