#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Drive the full conversational recipe: prepare data + tokenizer, pretrain,
# supervised fine-tune (masked loss on responses), then chat.
#
# Usage:
#   ./train_conversational.sh                  # full pipeline: prep -> pretrain -> sft -> chat
#   ./train_conversational.sh -b               # build the target first
#   ./train_conversational.sh -p               # prep + pretrain only
#   ./train_conversational.sh -f               # sft only (assumes a pretrained checkpoint)
#   ./train_conversational.sh -c "hello"       # chat only, with the given prompt
#   OPTIMIZER=muon ./train_conversational.sh   # use Muon instead of AdamW
#
# Steps / batch size / lr live in the env overrides below or the binary's flags.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_TRAIN_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

BIN="${BIN:-${TT_TRAIN_ROOT}/build/sources/examples/conversational/conversational}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
OPTIMIZER="${OPTIMIZER:-adamw}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-2000}"
SFT_STEPS="${SFT_STEPS:-1000}"
CHECKPOINT="${CHECKPOINT:-conversational.msgpack}"

DO_BUILD=0
RUN_PREP=1
RUN_PRETRAIN=1
RUN_SFT=1
RUN_CHAT=1
CHAT_PROMPT="hello"

while getopts "bpfc:h" opt; do
    case "$opt" in
        b) DO_BUILD=1 ;;
        p) RUN_SFT=0; RUN_CHAT=0 ;;                        # prep + pretrain only
        f) RUN_PREP=0; RUN_PRETRAIN=0; RUN_CHAT=0 ;;       # sft only
        c) RUN_PREP=0; RUN_PRETRAIN=0; RUN_SFT=0; CHAT_PROMPT="$OPTARG" ;;  # chat only
        h)
            grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown option. Use -h for help." >&2
            exit 1
            ;;
    esac
done

export TT_LOGGER_LEVEL="${TT_LOGGER_LEVEL:-FATAL}"

if [[ "${DO_BUILD}" -eq 1 ]]; then
    echo ">>> Building conversational (Release)..."
    cmake --build "${TT_TRAIN_ROOT}/build" --config Release --target conversational
fi

if [[ "${RUN_PREP}" -eq 1 ]]; then
    echo ">>> Preparing data + tokenizer (data dir: ${DATA_DIR})..."
    python3 "${SCRIPT_DIR}/prepare_data.py" --out_dir "${DATA_DIR}"
fi

if [[ ! -x "${BIN}" ]]; then
    echo "ERROR: conversational binary not found at ${BIN}" >&2
    echo "Build it first (see tt-train/README.md), pass BIN=/path/to/conversational, or run with -b." >&2
    exit 1
fi

TOKENIZER="${DATA_DIR}/tokenizer.json"

if [[ "${RUN_PRETRAIN}" -eq 1 ]]; then
    echo ">>> Pretraining (${OPTIMIZER}, ${PRETRAIN_STEPS} steps)..."
    "${BIN}" --mode pretrain \
        --tokenizer "${TOKENIZER}" \
        --data "${DATA_DIR}/pretrain.txt" \
        --checkpoint "${CHECKPOINT}" \
        --optimizer "${OPTIMIZER}" \
        --steps "${PRETRAIN_STEPS}"
fi

if [[ "${RUN_SFT}" -eq 1 ]]; then
    echo ">>> Supervised fine-tuning (masked loss, ${SFT_STEPS} steps)..."
    "${BIN}" --mode sft \
        --tokenizer "${TOKENIZER}" \
        --data "${DATA_DIR}/chat.jsonl" \
        --checkpoint "${CHECKPOINT}" \
        --optimizer "${OPTIMIZER}" \
        --steps "${SFT_STEPS}"
fi

if [[ "${RUN_CHAT}" -eq 1 ]]; then
    echo ">>> Chat (prompt: \"${CHAT_PROMPT}\")..."
    "${BIN}" --mode chat \
        --tokenizer "${TOKENIZER}" \
        --checkpoint "${CHECKPOINT}" \
        --prompt "${CHAT_PROMPT}"
fi

echo ">>> Done."
