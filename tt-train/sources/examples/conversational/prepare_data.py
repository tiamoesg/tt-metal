#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Prepare data for the `conversational` example.

Trains a small (~5k) byte-level BPE tokenizer and writes:
  - tokenizer.json   : load this with ttml::tokenizers::BPETokenizer
  - pretrain.txt     : the plain-text corpus used for --mode pretrain
  - chat.jsonl       : {"prompt", "response"} pairs used for --mode sft

By default it builds a tiny, self-contained TinyStories-style toy corpus and a
handful of chat pairs so the pipeline runs end to end with no downloads. Point
`--corpus` at your own text and `--chat` at your own jsonl for something real.

Requires: pip install tokenizers
"""

import argparse
import json
import os
import random

random.seed(0)

# A deliberately small, consistent vocabulary toy corpus (TinyStories spirit).
SUBJECTS = ["the cat", "a dog", "the girl", "the boy", "a bird", "the sun", "my friend", "the dragon"]
VERBS = ["runs", "jumps", "sleeps", "sings", "plays", "eats", "smiles", "flies"]
PLACES = ["in the park", "by the river", "at home", "in the garden", "on the hill", "under the tree"]
MOODS = ["happily", "quietly", "slowly", "brightly", "softly", "gently"]


def make_story():
    s = []
    for _ in range(random.randint(2, 4)):
        s.append(f"{random.choice(SUBJECTS)} {random.choice(VERBS)} {random.choice(PLACES)} {random.choice(MOODS)}.")
    return " ".join(s).capitalize()


CHAT_TEMPLATES = [
    ("hello", "Hello! How are you today?"),
    ("how are you", "I am doing well, thank you for asking."),
    ("what is your name", "I am a small language model."),
    ("tell me a story", lambda: make_story()),
    ("what do you like", "I like telling simple little stories."),
    ("goodbye", "Goodbye! Have a wonderful day."),
    ("what can you do", "I can chat with you and tell short stories."),
    ("are you happy", "Yes, I am happy to talk with you."),
]


def build_toy_corpus(path, num_stories):
    with open(path, "w") as f:
        for _ in range(num_stories):
            f.write(make_story() + "\n")
    print(f"wrote toy corpus: {path} ({num_stories} stories)")


def build_toy_chat(path, num_examples):
    with open(path, "w") as f:
        for _ in range(num_examples):
            prompt, response = random.choice(CHAT_TEMPLATES)
            resp = response() if callable(response) else response
            f.write(json.dumps({"prompt": prompt, "response": resp}) + "\n")
    print(f"wrote toy chat set: {path} ({num_examples} pairs)")


def train_tokenizer(corpus_path, out_path, vocab_size):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>"])
    tokenizer.train([corpus_path], trainer)
    tokenizer.save(out_path)
    print(f"trained tokenizer: {out_path} (vocab_size~{vocab_size})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "data"))
    ap.add_argument("--vocab_size", type=int, default=5000)
    ap.add_argument("--num_stories", type=int, default=20000)
    ap.add_argument("--num_chat", type=int, default=4000)
    ap.add_argument("--corpus", default=None, help="optional: use your own pretrain corpus .txt")
    ap.add_argument("--chat", default=None, help="optional: use your own chat .jsonl")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pretrain_path = args.corpus or os.path.join(args.out_dir, "pretrain.txt")
    chat_path = args.chat or os.path.join(args.out_dir, "chat.jsonl")
    tokenizer_path = os.path.join(args.out_dir, "tokenizer.json")

    if args.corpus is None:
        build_toy_corpus(pretrain_path, args.num_stories)
    if args.chat is None:
        build_toy_chat(chat_path, args.num_chat)

    train_tokenizer(pretrain_path, tokenizer_path, args.vocab_size)

    print("\nDone. Next:")
    print("  # pretrain")
    print(f"  ./build/sources/examples/conversational/conversational --mode pretrain --data {pretrain_path}")
    print("  # supervised fine-tune on chat pairs (masked loss)")
    print(f"  ./build/sources/examples/conversational/conversational --mode sft --data {chat_path}")
    print("  # chat")
    print('  ./build/sources/examples/conversational/conversational --mode chat --prompt "hello"')


if __name__ == "__main__":
    main()
