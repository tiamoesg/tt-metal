# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Single-device speed harness for DeepSeek-V4 inference (verify on 1 Wormhole).

The full V4 model (~1 TB FP8 / ~500 GB FP4) does not fit on one Wormhole, but the
quantity every Galaxy projection rests on -- the fraction of peak DRAM bandwidth the
real V4 kernels achieve (eta) -- is a single-chip property that transfers across the
Tensix architecture (Wormhole -> Blackhole). This harness measures, at full() dims:

  * per-kernel decode latency + achieved GB/s + %-of-peak  (eta)
  * a composed per-layer decode time  (attention block + active MoE experts)
  * the MoE batch-amortization curve   (tok/s vs concurrent streams)

then projects per-user tok/s on a Blackhole Galaxy via the published hardware ratios,
parameterized by the one thing a single chip cannot measure: multi-chip pipeline
efficiency P.

Run (Wormhole n150/n300):
    python -m models.demos.deepseek_v4_inference.tt.bench
    python -m models.demos.deepseek_v4_inference.tt.bench --dtype bfp4 --device-bw 288
"""

from __future__ import annotations

import argparse

# Published peak DRAM bandwidth (GB/s) per board, for %-of-peak and projection.
PEAK_BW = {"wh-n150": 288.0, "wh-n300": 576.0, "bh-p150": 512.0, "galaxy": 16000.0}
WH_TO_BH_BW = 512.0 / 288.0      # per-chip bandwidth scaling Wormhole -> Blackhole
GALAXY_CHIPS = 32
BYTES = {"bf16": 2.0, "bfp8": 1.0, "bfp4": 0.5}


def _time(fn, *, warmup=3, iters=20):
    """Median wall-clock ms of `fn` with device sync. Imported lazily (needs ttnn)."""
    import time

    import ttnn

    for _ in range(warmup):
        out = fn()
        ttnn.synchronize_device(out.device()) if hasattr(out, "device") else None
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        dev = out.device() if hasattr(out, "device") else None
        if dev is not None:
            ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def _report(name, ms, bytes_moved, peak_bw):
    gbps = (bytes_moved / 1e9) / (ms / 1e3)
    eta = gbps / peak_bw
    print(f"  {name:28s} {ms:8.3f} ms   {gbps:8.1f} GB/s   eta={eta*100:5.1f}% of {peak_bw:.0f}")
    return eta


def bench(device, peak_bw, dtype):
    """Measure unit-kernel etas + a composed per-layer decode time. Returns (eta, t_layer_ms)."""
    import torch

    import ttnn
    from models.demos.deepseek_v4_inference.config import V4Config
    from models.demos.deepseek_v4_inference.tt.moe import TtExpert
    from models.demos.deepseek_v4_inference.tt.sink_attention import sink_attention

    cfg = V4Config.full()
    nbytes = BYTES[dtype]
    to_dev = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print(f"\nfull() dims: dim={cfg.dim} heads={cfg.n_heads} head_dim={cfg.head_dim} "
          f"experts={cfg.n_routed_experts}(+{cfg.n_shared_experts}) active={cfg.n_activated_experts} "
          f"layers={cfg.n_layers}  dtype={dtype}\n")

    # --- one MoE expert (the decode unit cost: 3 matmuls, SwiGLU) ---
    inter = cfg.moe_inter_dim
    w = lambda o, i: torch.randn(o, i) * 0.02
    expert = TtExpert(device, w(inter, cfg.dim), w(cfg.dim, inter), w(inter, cfg.dim), cfg.swiglu_limit)
    x1 = to_dev(torch.randn(1, 1, 32, cfg.dim))           # 1 token (pad to tile)
    expert_bytes = nbytes * 3 * cfg.dim * inter
    eta_e = _report("moe expert (1 tok)", _time(lambda: expert(x1)), expert_bytes, peak_bw)

    # --- attention decode core (sink attention over kept KV) ---
    h, s, k, hd = cfg.n_heads, 32, cfg.index_topk, cfg.head_dim
    q = to_dev(torch.randn(1, h, s, hd) * 0.1)
    keys = to_dev(torch.randn(1, 1, k, hd) * 0.1)
    keep = to_dev((torch.rand(1, 1, s, k) > 0.5).float())
    sink = to_dev(torch.zeros(1, h, 1, 1))
    attn_bytes = nbytes * h * k * hd * 2                   # q.kv + w.kv streamed KV
    eta_a = _report("sink attention", _time(lambda: sink_attention(q, keys, keep, sink, hd ** -0.5)),
                    attn_bytes, peak_bw)

    # --- composed per-layer decode time (active experts + shared + attention) ---
    active = cfg.n_activated_experts + cfg.n_shared_experts
    t_e = _time(lambda: expert(x1))
    t_a = _time(lambda: sink_attention(q, keys, keep, sink, hd ** -0.5))
    # attention QKV/O projections ~ a few dim x dim matmuls; approximate with 4 expert-sized reads
    t_layer = active * t_e + t_a + 4 * (nbytes * cfg.dim * cfg.dim / 1e9) / (peak_bw * max(eta_e, 0.3)) * 1e3
    eta = (eta_e + eta_a) / 2
    print(f"\n  composed t_layer ~= {t_layer:.3f} ms  (mean eta ~= {eta*100:.1f}%)")
    return eta, t_layer


def bench_amortization(device, dtype, batches=(1, 2, 4, 8, 16, 32)):
    """MoE expert-read amortization: tok/s vs batch on a single expert (unit curve)."""
    import torch

    import ttnn
    from models.demos.deepseek_v4_inference.config import V4Config
    from models.demos.deepseek_v4_inference.tt.moe import TtExpert

    cfg = V4Config.full()
    inter = cfg.moe_inter_dim
    w = lambda o, i: torch.randn(o, i) * 0.02
    expert = TtExpert(device, w(inter, cfg.dim), w(cfg.dim, inter), w(inter, cfg.dim), cfg.swiglu_limit)
    to_dev = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print("\n  MoE expert amortization (weights read once, applied to B tokens):")
    for b in batches:
        x = to_dev(torch.randn(1, 1, b * 32, cfg.dim))    # B tokens (tile-padded)
        ms = _time(lambda: expert(x))
        print(f"    batch {b:3d}   {ms:8.3f} ms   {b / (ms / 1e3):10.1f} tok/s/expert")


def project(eta, t_layer_ms, n_layers, active_gb, p_values=(0.4, 0.6, 0.8)):
    """Project per-user tok/s on a Blackhole Galaxy from the Wormhole per-layer time."""
    t_model_wh = n_layers * t_layer_ms                    # whole model, 1 WH chip
    t_model_bh = t_model_wh / WH_TO_BH_BW                 # per-chip BW scaling, same eta
    print(f"\n  projection (measured eta carried through):")
    print(f"    1 WH chip whole-model decode  ~= {t_model_wh:8.1f} ms/token")
    print(f"    1 BH chip whole-model decode  ~= {t_model_bh:8.1f} ms/token")
    for p in p_values:
        t_user = t_model_bh / (GALAXY_CHIPS * p)
        print(f"    Galaxy (32 BH, P={p:.1f})         ~= {1000 / t_user:8.1f} tok/s/user")
    print(f"\n  (active weights/token ~= {active_gb:.1f} GB FP4; BW-bound ceiling check: "
          f"{PEAK_BW['galaxy'] * eta / active_gb:.0f} tok/s/user at eta={eta*100:.0f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=list(BYTES), default="bfp8")
    ap.add_argument("--device-bw", type=float, default=PEAK_BW["wh-n150"], help="peak DRAM GB/s of THIS card")
    args = ap.parse_args()

    import ttnn
    from models.demos.deepseek_v4_inference.config import V4Config

    device = ttnn.open_device(device_id=0)
    try:
        eta, t_layer = bench(device, args.device_bw, args.dtype)
        bench_amortization(device, args.dtype)
        cfg = V4Config.full()
        active_gb = BYTES["bfp4"] * (cfg.n_activated_experts + cfg.n_shared_experts) * 3 * cfg.dim * \
            cfg.moe_inter_dim * cfg.n_layers / 1e9
        project(eta, t_layer, cfg.n_layers, active_gb)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
