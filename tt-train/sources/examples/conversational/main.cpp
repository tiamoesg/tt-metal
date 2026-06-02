// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// conversational: train a tiny chat model on Tenstorrent / tt-train.
//
// A small (~10M parameter) GPT over a ~5k-token BPE vocabulary that learns to
// hold a short conversation. Demonstrates the full small-LM recipe:
//   --mode pretrain : next-token training on a plain-text corpus
//   --mode sft      : supervised fine-tuning on {prompt, response} pairs, with
//                     the loss masked to the response tokens only
//   --mode chat     : generate a response to a prompt
//
// The masked supervised fine-tuning is the key piece: ops::cross_entropy_loss_masked
// computes the loss on assistant tokens only, so the model learns to *produce*
// responses rather than to also model the prompt.

#include <fmt/format.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <cmath>
#include <core/ttnn_all_includes.hpp>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "models/gpt2.hpp"
#include "ops/losses.hpp"
#include "optimizers/adamw.hpp"
#include "optimizers/muon.hpp"
#include "serialization/msgpack_file.hpp"
#include "serialization/serialization.hpp"
#include "tokenizers/bpe_tokenizer.hpp"

namespace {

constexpr auto kUserTag = "User: ";
constexpr auto kAssistantTag = "\nAssistant:";

struct Config {
    std::string mode = "chat";
    std::string tokenizer_path;
    std::string data_path;  // corpus (.txt) for pretrain, jsonl for sft
    std::string checkpoint = "conversational.msgpack";
    std::string optimizer_type = "adamw";
    std::string prompt;  // for chat mode
    uint32_t embedding_dim = 256;
    uint32_t num_heads = 8;
    uint32_t num_blocks = 6;
    uint32_t block_size = 256;
    uint32_t batch_size = 32;
    uint32_t steps = 2000;
    float learning_rate = 3e-4F;
    float temperature = 0.8F;
    uint32_t max_new_tokens = 128;
};

uint32_t round_up_32(uint32_t v) {
    return (v + 31U) / 32U * 32U;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error(fmt::format("Could not open file: {}", path));
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void save_checkpoint(ttml::autograd::ModuleBase& model, const std::string& path) {
    ttml::serialization::MsgPackFile file;
    ttml::serialization::write_module(file, "model", &model);
    file.serialize(path);
}

void load_checkpoint(ttml::autograd::ModuleBase& model, const std::string& path) {
    ttml::serialization::MsgPackFile file;
    file.deserialize(path);
    ttml::serialization::read_module(file, "model", &model);
}

ttml::models::gpt2::TransformerConfig make_model_config(const Config& cfg, uint32_t vocab_size) {
    ttml::models::gpt2::TransformerConfig mc;
    mc.num_heads = cfg.num_heads;
    mc.embedding_dim = cfg.embedding_dim;
    mc.dropout_prob = 0.0F;
    mc.num_blocks = cfg.num_blocks;
    mc.vocab_size = round_up_32(vocab_size);
    mc.max_sequence_length = cfg.block_size;
    return mc;
}

// Lower-triangular causal mask [1, 1, S, S].
ttml::autograd::TensorPtr make_causal_mask(uint32_t block_size, ttnn::distributed::MeshDevice* device) {
    std::vector<float> mask;
    mask.reserve(static_cast<size_t>(block_size) * block_size);
    for (uint32_t i = 0; i < block_size; ++i) {
        for (uint32_t j = 0; j < block_size; ++j) {
            mask.push_back(i >= j ? 1.0F : 0.0F);
        }
    }
    return ttml::autograd::create_tensor(
        ttml::core::from_vector(mask, ttnn::Shape({1, 1, block_size, block_size}), device));
}

std::unique_ptr<ttml::optimizers::OptimizerBase> make_optimizer(
    const Config& cfg, const ttml::serialization::NamedParameters& params) {
    if (cfg.optimizer_type == "muon") {
        ttml::optimizers::MuonConfig mc;
        mc.lr = cfg.learning_rate;
        return std::make_unique<ttml::optimizers::Muon>(params, mc);
    }
    ttml::optimizers::AdamWConfig ac;
    ac.lr = cfg.learning_rate;
    return std::make_unique<ttml::optimizers::AdamW>(params, ac);
}

// ----------------------------------------------------------------------------
// Pretraining: next-token prediction over a packed token stream.
// ----------------------------------------------------------------------------
void run_pretrain(Config& cfg, const std::shared_ptr<ttml::models::gpt2::Transformer>& model) {
    auto& ctx = ttml::autograd::ctx();
    auto* device = &ctx.get_device();
    ttml::tokenizers::BPETokenizer tokenizer(cfg.tokenizer_path);

    fmt::print("Tokenizing corpus...\n");
    auto tokens = tokenizer.encode(read_file(cfg.data_path));
    if (tokens.size() <= cfg.block_size + 1U) {
        throw std::runtime_error("Pretraining corpus is too small for the configured block size.");
    }
    fmt::print("Corpus tokens: {}\n", tokens.size());

    auto optimizer = make_optimizer(cfg, model->parameters());
    auto mask = make_causal_mask(cfg.block_size, device);

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> start_dist(0, tokens.size() - cfg.block_size - 2U);

    for (uint32_t step = 0; step < cfg.steps; ++step) {
        std::vector<uint32_t> inputs(static_cast<size_t>(cfg.batch_size) * cfg.block_size);
        std::vector<uint32_t> targets(static_cast<size_t>(cfg.batch_size) * cfg.block_size);
        for (uint32_t b = 0; b < cfg.batch_size; ++b) {
            size_t start = start_dist(rng);
            for (uint32_t t = 0; t < cfg.block_size; ++t) {
                inputs[b * cfg.block_size + t] = tokens[start + t];
                targets[b * cfg.block_size + t] = tokens[start + t + 1U];
            }
        }

        auto input_tensor = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
            inputs, ttnn::Shape({cfg.batch_size, 1, 1, cfg.block_size}), device, ttnn::Layout::ROW_MAJOR));
        auto target_tensor = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
            targets, ttnn::Shape({cfg.batch_size, cfg.block_size}), device, ttnn::Layout::ROW_MAJOR));

        optimizer->zero_grad();
        auto logits = (*model)(input_tensor, mask);
        auto loss = ttml::ops::cross_entropy_loss(logits, target_tensor);
        float loss_value = ttml::core::to_vector(loss->get_value())[0];
        loss->backward();
        optimizer->step();
        ctx.reset_graph();

        if ((step + 1U) % 50U == 0U || step == 0U) {
            fmt::print("[pretrain] step {:5d}/{:5d} | loss {:.4f}\n", step + 1U, cfg.steps, loss_value);
        }
    }
    save_checkpoint(*model, cfg.checkpoint);
    fmt::print("Saved checkpoint to {}\n", cfg.checkpoint);
}

// ----------------------------------------------------------------------------
// Supervised fine-tuning: learn to produce responses. The loss is masked to the
// response tokens only -- the model is never penalized for "predicting" the
// prompt, only for its replies.
// ----------------------------------------------------------------------------
struct ChatExample {
    std::string prompt;
    std::string response;
};

std::vector<ChatExample> load_chat_jsonl(const std::string& path) {
    std::vector<ChatExample> examples;
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error(fmt::format("Could not open sft data: {}", path));
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) {
            continue;
        }
        auto obj = nlohmann::json::parse(line);
        examples.push_back(ChatExample{obj.at("prompt").get<std::string>(), obj.at("response").get<std::string>()});
    }
    return examples;
}

void run_sft(Config& cfg, const std::shared_ptr<ttml::models::gpt2::Transformer>& model) {
    auto& ctx = ttml::autograd::ctx();
    auto* device = &ctx.get_device();
    ttml::tokenizers::BPETokenizer tokenizer(cfg.tokenizer_path);

    if (!cfg.checkpoint.empty() && std::ifstream(cfg.checkpoint).good()) {
        load_checkpoint(*model, cfg.checkpoint);
        fmt::print("Loaded pretrained checkpoint {}\n", cfg.checkpoint);
    }

    auto examples = load_chat_jsonl(cfg.data_path);
    fmt::print("SFT examples: {}\n", examples.size());

    auto optimizer = make_optimizer(cfg, model->parameters());
    auto mask = make_causal_mask(cfg.block_size, device);

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> ex_dist(0, examples.size() - 1U);

    const uint32_t S = cfg.block_size;
    for (uint32_t step = 0; step < cfg.steps; ++step) {
        std::vector<uint32_t> inputs(static_cast<size_t>(cfg.batch_size) * S, 0U);
        std::vector<uint32_t> targets(static_cast<size_t>(cfg.batch_size) * S, 0U);
        std::vector<float> loss_mask(static_cast<size_t>(cfg.batch_size) * S, 0.0F);  // [N, H] -> [N,1,H,1]

        for (uint32_t b = 0; b < cfg.batch_size; ++b) {
            const auto& ex = examples[ex_dist(rng)];
            auto prefix = tokenizer.encode(std::string(kUserTag) + ex.prompt + kAssistantTag);
            auto reply = tokenizer.encode(" " + ex.response + "\n");
            std::vector<uint32_t> full = prefix;
            full.insert(full.end(), reply.begin(), reply.end());

            const size_t prefix_len = prefix.size();
            for (uint32_t t = 0; t < S; ++t) {
                size_t in_idx = t;
                size_t tgt_idx = t + 1U;
                inputs[b * S + t] = (in_idx < full.size()) ? full[in_idx] : 0U;
                targets[b * S + t] = (tgt_idx < full.size()) ? full[tgt_idx] : 0U;
                // Train only where the *target* is a response token (and real, not padding).
                loss_mask[b * S + t] = (tgt_idx >= prefix_len && tgt_idx < full.size()) ? 1.0F : 0.0F;
            }
        }

        auto input_tensor = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
            inputs, ttnn::Shape({cfg.batch_size, 1, 1, S}), device, ttnn::Layout::ROW_MAJOR));
        auto target_tensor = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
            targets, ttnn::Shape({cfg.batch_size, S}), device, ttnn::Layout::ROW_MAJOR));
        auto mask_tensor = ttml::autograd::create_tensor(
            ttml::core::from_vector(loss_mask, ttnn::Shape({cfg.batch_size, 1, S, 1}), device));

        optimizer->zero_grad();
        auto logits = (*model)(input_tensor, mask);
        auto loss = ttml::ops::cross_entropy_loss_masked(logits, target_tensor, mask_tensor);
        float loss_value = ttml::core::to_vector(loss->get_value())[0];
        loss->backward();
        optimizer->step();
        ctx.reset_graph();

        if ((step + 1U) % 25U == 0U || step == 0U) {
            fmt::print("[sft] step {:5d}/{:5d} | loss {:.4f}\n", step + 1U, cfg.steps, loss_value);
        }
    }
    save_checkpoint(*model, cfg.checkpoint);
    fmt::print("Saved fine-tuned checkpoint to {}\n", cfg.checkpoint);
}

// ----------------------------------------------------------------------------
// Chat: generate a response to a single prompt.
// ----------------------------------------------------------------------------
uint32_t sample_token(const std::vector<float>& logits, uint32_t vocab_size, float temperature, std::mt19937& rng) {
    float max_logit = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < vocab_size; ++i) {
        max_logit = std::max(max_logit, logits[i] / temperature);
    }
    std::vector<float> probs(vocab_size);
    float sum = 0.0F;
    for (uint32_t i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp(logits[i] / temperature - max_logit);
        sum += probs[i];
    }
    std::uniform_real_distribution<float> dist(0.0F, sum);
    float r = dist(rng);
    float cum = 0.0F;
    for (uint32_t i = 0; i < vocab_size; ++i) {
        cum += probs[i];
        if (r <= cum) {
            return i;
        }
    }
    return vocab_size - 1U;
}

void run_chat(Config& cfg, const std::shared_ptr<ttml::models::gpt2::Transformer>& model) {
    auto& ctx = ttml::autograd::ctx();
    auto* device = &ctx.get_device();
    ttml::tokenizers::BPETokenizer tokenizer(cfg.tokenizer_path);

    if (std::ifstream(cfg.checkpoint).good()) {
        load_checkpoint(*model, cfg.checkpoint);
    } else {
        fmt::print("Warning: no checkpoint at {}; the model is untrained.\n", cfg.checkpoint);
    }
    model->eval();

    const uint32_t vocab_size = tokenizer.get_vocab_size();
    const uint32_t logits_width = round_up_32(vocab_size);
    auto mask = make_causal_mask(cfg.block_size, device);

    auto context = tokenizer.encode(std::string(kUserTag) + cfg.prompt + kAssistantTag + " ");
    std::mt19937 rng(std::random_device{}());

    fmt::print("Assistant:");
    for (uint32_t n = 0; n < cfg.max_new_tokens; ++n) {
        // Window the context to the last block_size tokens.
        std::vector<uint32_t> window(cfg.block_size, 0U);
        const size_t ctx_len = std::min<size_t>(context.size(), cfg.block_size);
        const size_t offset = context.size() - ctx_len;
        for (size_t i = 0; i < ctx_len; ++i) {
            window[i] = context[offset + i];
        }

        auto input_tensor = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
            window, ttnn::Shape({1, 1, 1, cfg.block_size}), device, ttnn::Layout::ROW_MAJOR));
        auto logits = (*model)(input_tensor, mask);
        auto logits_vec = ttml::core::to_vector(logits->get_value());  // [block_size * logits_width]
        ctx.reset_graph();

        const size_t last = ctx_len - 1U;
        std::vector<float> next(
            logits_vec.begin() + static_cast<long>(last) * logits_width,
            logits_vec.begin() + static_cast<long>(last) * logits_width + vocab_size);
        uint32_t next_token = sample_token(next, vocab_size, cfg.temperature, rng);

        auto piece = tokenizer.decode({next_token});
        if (piece.find('\n') != std::string::npos) {
            break;  // end of the assistant turn
        }
        fmt::print("{}", piece);
        std::fflush(stdout);
        context.push_back(next_token);
    }
    fmt::print("\n");
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"conversational: a tiny chat model on Tenstorrent"};
    Config cfg;
    cfg.tokenizer_path = std::string(DATA_FOLDER) + "/tokenizer.json";

    app.add_option("--mode", cfg.mode, "pretrain | sft | chat")->default_val(cfg.mode);
    app.add_option("--tokenizer", cfg.tokenizer_path, "Path to BPE tokenizer.json")->default_val(cfg.tokenizer_path);
    app.add_option("--data", cfg.data_path, "Corpus (.txt) for pretrain, or chat .jsonl for sft");
    app.add_option("--checkpoint", cfg.checkpoint, "Model checkpoint path")->default_val(cfg.checkpoint);
    app.add_option("--optimizer", cfg.optimizer_type, "adamw | muon")->default_val(cfg.optimizer_type);
    app.add_option("--prompt", cfg.prompt, "Prompt (chat mode)");
    app.add_option("--steps", cfg.steps, "Training steps")->default_val(cfg.steps);
    app.add_option("--batch_size", cfg.batch_size, "Batch size")->default_val(cfg.batch_size);
    app.add_option("--lr", cfg.learning_rate, "Learning rate")->default_val(cfg.learning_rate);
    app.add_option("--block_size", cfg.block_size, "Context length")->default_val(cfg.block_size);
    CLI11_PARSE(app, argc, argv);

    auto& ctx = ttml::autograd::ctx();
    ctx.open_device();

    ttml::tokenizers::BPETokenizer tokenizer(cfg.tokenizer_path);
    const uint32_t vocab_size = tokenizer.get_vocab_size();
    fmt::print("Vocab size: {} (padded to {})\n", vocab_size, round_up_32(vocab_size));

    auto model = ttml::models::gpt2::create(make_model_config(cfg, vocab_size));

    if (cfg.mode == "pretrain") {
        run_pretrain(cfg, model);
    } else if (cfg.mode == "sft") {
        run_sft(cfg, model);
    } else if (cfg.mode == "chat") {
        run_chat(cfg, model);
    } else {
        fmt::print("Unknown mode: {}. Use pretrain | sft | chat.\n", cfg.mode);
    }

    ctx.close_device();
    return 0;
}
