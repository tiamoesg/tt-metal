#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Train the char-level NanoGPT (Shakespeare) example with the Muon optimizer.
#
# This drives the full train cycle:
#   1. (optional) build the nano_gpt target
#   2. run training with the Muon config; checkpoints to configs' model_path and
#      auto-resumes if that checkpoint already exists
#   3. run a sample generation from the trained checkpoint
#
# Usage:
#   ./train_muon.sh                 # train then sample, using the default config
#   ./train_muon.sh -e              # evaluation/generation only (skip training)
#   ./train_muon.sh -b              # build the target first, then train
#   CONFIG=... ./train_muon.sh      # override the config file
#
# The number of steps, batch size, learning rate, and checkpoint path all live
# in the YAML config (configs/training_shakespeare_nanogpt_muon.yaml).

set -euo pipefail

# Resolve paths relative to the tt-train root (two levels up from this script's
# directory: sources/examples/nano_gpt -> tt-train).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_TRAIN_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CONFIG="${CONFIG:-${TT_TRAIN_ROOT}/configs/training_shakespeare_nanogpt_muon.yaml}"
BIN="${BIN:-${TT_TRAIN_ROOT}/build/sources/examples/nano_gpt/nano_gpt}"
ENABLE_WANDB="${ENABLE_WANDB:-0}"   # set to 1 to log to Weights & Biases
RUN_NAME="${RUN_NAME:-nano_gpt_muon}"

DO_BUILD=0
EVAL_ONLY=0
while getopts "beh" opt; do
    case "$opt" in
        b) DO_BUILD=1 ;;
        e) EVAL_ONLY=1 ;;
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
    echo ">>> Building nano_gpt (Release)..."
    cmake --build "${TT_TRAIN_ROOT}/build" --config Release --target nano_gpt
fi

if [[ ! -x "${BIN}" ]]; then
    echo "ERROR: nano_gpt binary not found at ${BIN}" >&2
    echo "Build it first (see tt-train/README.md) or pass BIN=/path/to/nano_gpt, or run with -b." >&2
    exit 1
fi

if [[ "${EVAL_ONLY}" -eq 0 ]]; then
    echo ">>> Training with Muon: ${CONFIG}"
    "${BIN}" -c "${CONFIG}" -w "${ENABLE_WANDB}" -n "${RUN_NAME}"
fi

echo ">>> Sampling from the trained model..."
"${BIN}" -c "${CONFIG}" -e 1

echo ">>> Done."
