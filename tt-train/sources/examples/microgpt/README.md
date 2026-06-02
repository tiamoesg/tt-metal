# microgpt

A port of Andrej Karpathy's [`microgpt.py`](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
to Tenstorrent via `tt-train`. microgpt is the successor to nanoGPT — Karpathy's
"most atomic way to train and run inference for a GPT," the complete algorithm
with everything else stripped away. This is that essence, running on Tenstorrent
hardware through the `ttml` autograd library.

It trains a tiny character-level GPT on the [makemore `names`](https://github.com/karpathy/makemore)
dataset and then generates new, plausible-sounding names.

## Architecture

Follows GPT-2 with microgpt's deliberate simplifications:

- **RMSNorm** instead of LayerNorm
- **ReLU** instead of GeLU
- **no biases** in the MLP and head
- token + trainable positional embeddings, an initial RMSNorm, `n_layer` blocks
  of `RMSNorm → MHA → residual` and `RMSNorm → Linear → ReLU → Linear → residual`,
  then a linear head to vocabulary logits.

Defaults are tiny: `n_layer=1`, `n_embd=128`, `n_head=4`, `block_size=32`
(~0.3M parameters). The only deviations from `microgpt.py` are sizes chosen to be
multiples of 32 for Tenstorrent tiles, and full-sequence attention with a causal
mask instead of a scalar per-position KV-cache loop. The algorithm is identical.

## Run

Build the project (see the top-level `tt-train/README.md`), then:

```bash
# Train with AdamW (default) and sample names
TT_LOGGER_LEVEL=FATAL ./build/sources/examples/microgpt/microgpt

# Train with the Muon optimizer instead
TT_LOGGER_LEVEL=FATAL ./build/sources/examples/microgpt/microgpt --optimizer muon --lr 0.02

# More steps / different batch size
./build/sources/examples/microgpt/microgpt --steps 4000 --batch_size 128
```

The `names.txt` dataset is downloaded automatically at configure time into
`tt-train/data/`.

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `-d, --data` | `data/names.txt` | dataset path |
| `-o, --optimizer` | `adamw` | `adamw` or `muon` |
| `-s, --steps` | `2000` | training steps |
| `-b, --batch_size` | `64` | names per step |
| `-l, --lr` | `0.01` | learning rate |
| `--samples` | `20` | number of names to generate after training |
