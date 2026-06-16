# DeepSeek-V4 Inference — Roadmap & Sequencing

Living tracker for the V4-on-Tenstorrent inference port. **Work top-to-bottom; do not
start a step until its dependencies are ✅ and its predecessor's validation gate
passes.** Each step names its validation gate and the DeepSeek-AI reference to port.

Status legend: [x] done · [~] doing · [>] next · [ ] later · [!] blocked

---

## Phase 0 — CPU reference (the ground truth)  ✅
Everything downstream validates against this. No device needed.

- ✅ `config.py` — V4 config (`full()` from config.json, `small()` for CPU)
- ✅ `reference_model.py` — architecture-faithful PyTorch (CSA/HCA, indexer+top-k, mHC, MoE, RoPE)
- ✅ `test_reference.py` — CPU smoke (shape, finiteness, causality, generate, score-funcs)
- **Gate:** `pytest test_reference.py` green in the tt-metal torch env. *(Not yet run — written blind; Phase 1 step 1 is to actually run it.)*

## Phase 1 — Make the reference trustworthy
- ⏳ **1.1** Run `test_reference.py` in the torch env; fix transcription bugs.
- ⬜ **1.2** Reconcile against official sources, replacing the two flagged assumptions:
  - `hc_split_sinkhorn` element split / scale-index mapping ← official `kernel.py`
  - compressed-block RoPE position convention (currently `s*ratio`) ← `model.py` Compressor
- ⬜ **1.3** Optional: load a real V4 checkpoint into the reference and match logits vs the
  official `model.py` on a fixed prompt (proves faithfulness end-to-end).
- **Gate:** reference logits match official `model.py` (PCC > 0.99) on a fixed prompt.

## Phase 2 — TT-NN device modules (op-by-op, each PCC-checked vs Phase 0)
Build under `tt/`, mirroring `models/demos/deepseek_v3/tt`. Validate **each module
in isolation** against the reference before composing. Order by dependency:

- 🔨 **2.1 RoPE variants** — partial + strided (`s*ratio`) + inverse (`−i`). **Implemented**
  (`tt/rope.py`, interleaved-pair convention via constant rotate-matrix) + PCC test
  (`tt/test_rope.py`); pending device run. *(Smallest; unblocks attention.)*
- 🔨 **2.2 Compressor** — softmax-gated pooling (overlapping CSA / single HCA). **Implemented**
  (`tt/compressor.py` + PCC test `tt/test_compressor.py`); overlap index mapping verified in
  pure Python; pending device run. Ref: model.py.
- ⬜ **2.3 Lightning indexer + top-k** — low-rank q, ReLU-weighted scores, `ttnn::topk`. Ref: **DeepSeek-V3.2-Exp (DSA)**.
- ⬜ **2.4 ⭐ Sparse-gather decode (the hard one)** — gather top-k compressed blocks from
  DRAM + masked SDPA + sink. Extend `deepseek_v3_b1`'s `flash_mla` + `kv_cache_update`.
  Ref: **DSA kernels** + **FlashMLA**. *This step decides whether sparsity pays off.*
- ⬜ **2.5 Sliding-window branch + attention sink** — fold into 2.4's SDPA.
- ⬜ **2.6 Grouped output projection** — `wo_a`/`wo_b`.
- ⬜ **2.7 mHC** — `hc_pre`/`hc_post` + Sinkhorn; widens residual to `hc_mult` streams.
- ⬜ **2.8 DeepSeekMoE** — sqrtsoftplus gate (reuse `b1` `deepseek_moe_gate`) + DRAM-streamed experts.
- **Gate (each):** module PCC > 0.99 vs reference at `small()` shapes.

## Phase 3 — Assemble the decode engine (on the `b1` skeleton)
- ⬜ **3.1** V4 `Block` + `Transformer` wiring (reuse `b1` pipeline/runner/stage).
- ⬜ **3.2** Incremental **KV-cache decode**: compressed-block append every `m` tokens +
  sliding window of recent uncompressed; replaces the reference's recompute path.
- ⬜ **3.3** End-to-end prefill + decode on one device slice; greedy-match vs reference.
- **Gate:** generated token IDs match the reference greedily for N steps.

## Phase 4 — Scale-out & performance
- ⬜ **4.1** Pipeline-parallel across pods (`stage_to_slice` mapping). Ref: **DualPipe** + **profile-data**.
- ⬜ **4.2** Expert parallelism + load balancing. Ref: **DeepEP**, **EPLB/LPLB**.
- ⬜ **4.3** FP8/FP4 weights + fine-grained scaling. Ref: **DeepGEMM**; align kernels with **TileKernels**.
- **Gate:** per-user decode tok/s at batch 1 measured on target cluster; KV stays resident.

## Phase 5 — Serving for agents
- ⬜ **5.1** vLLM-style generator (cf. `deepseek_v3/tt/generator_vllm.py`) → OpenAI-compatible endpoint.
- ⬜ **5.2** MTP / speculative decode for per-user latency (cf. `deepseek_v3/tt/mtp.py`).
- ⬜ **5.3** Agent-hosting harness. Ref: **awesome-deepseek-agent**.
- **Gate:** host an agent loop end-to-end; measure interactive latency.

---

## Critical path (the one-line sequence)
`run reference (1.1) → RoPE (2.1) → compressor (2.2) → indexer+topk (2.3) →`
**`sparse-gather decode (2.4)`** `→ mHC (2.7) + MoE (2.8) → assemble (3.1) → KV-cache decode (3.2) → serve (5.1)`

The ⭐ at **2.4** is the highest-risk, highest-value kernel — everything before it is
setup, everything after assumes it works. De-risk it early (a standalone spike against
`b1`'s `flash_mla`) before committing to the full assembly.

## Cross-cutting reality checks (don't regress these)
- KV in **DRAM but tiny** (compression is the lever), experts **streamed from DRAM**.
- Validate every TT op against Phase 0 at `small()` before scaling shapes.
- Honest caveat: all C++/TT code in this effort is written against header-verified APIs
  and **not yet compiled/run on hardware** — Phase 1.1 (running the CPU reference) is the
  first thing that actually executes.
