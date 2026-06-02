# conversational

A tiny chat model on Tenstorrent / `tt-train`: a small (~10M parameter) GPT over
a ~5k-token BPE vocabulary that learns to hold a short conversation. This is the
"small conversationalist" recipe end to end:

1. **Pretrain** — next-token prediction on a plain-text corpus.
2. **SFT** — supervised fine-tuning on `{prompt, response}` pairs, with the loss
   **masked to the response tokens only** (`ops::cross_entropy_loss_masked`), so
   the model learns to *produce* replies rather than to also model the prompt.
3. **Chat** — generate a response to a prompt.

It reuses the production `ttml::models::gpt2` Transformer, the BPE tokenizer, and
either AdamW or **Muon**.

## 1. Prepare data + tokenizer

```bash
pip install tokenizers
python sources/examples/conversational/prepare_data.py
```

With no arguments this builds a self-contained, TinyStories-style toy corpus and
chat set plus a ~5k BPE `tokenizer.json` under `sources/examples/conversational/data/`,
so the whole pipeline runs with no downloads. Point `--corpus your.txt` and
`--chat your.jsonl` at real data for something substantial.

## 2. Train and chat

```bash
BIN=./build/sources/examples/conversational/conversational

# Pretrain language modeling on the corpus
$BIN --mode pretrain --data sources/examples/conversational/data/pretrain.txt --optimizer muon

# Supervised fine-tune on chat pairs (masked loss on responses)
$BIN --mode sft --data sources/examples/conversational/data/chat.jsonl

# Talk to it
$BIN --mode chat --prompt "hello"
```

The tokenizer defaults to `data/tokenizer.json`; checkpoints default to
`conversational.msgpack` (pretrain writes it, sft loads and refines it, chat
loads it).

## Model / scale

Defaults: `embedding_dim=256`, `num_blocks=6`, `num_heads=8`, `block_size=256`,
vocab ~5k → roughly **10M parameters**. This is the TinyStories-class regime
where small models over a constrained vocabulary produce coherent text. Trains in
minutes-to-tens-of-minutes of compute. Scale up `--block_size`, the corpus, and
the chat set for a stronger conversationalist.

## The masked-loss detail

During SFT each example is laid out as `User: <prompt>\nAssistant: <response>\n`.
The loss mask is `1.0` exactly on the positions whose **target** is a response
token and `0.0` on the prompt and padding, so gradients flow only from the
assistant's words. That single change is what turns a language model into a
chat model.
