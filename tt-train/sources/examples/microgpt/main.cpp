// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// microgpt: a port of Andrej Karpathy's microgpt.py to Tenstorrent / tt-train.
// "The most atomic way to train and run inference for a GPT" -- here expressed
// with ttml autograd tensors so the whole thing runs on Tenstorrent hardware.
//
// It trains a tiny character-level GPT on the makemore `names` dataset and then
// hallucinates new names. Following GPT-2, blessed among the GPTs, with the same
// minor differences microgpt makes: RMSNorm instead of LayerNorm, no biases in
// the MLP / head, ReLU instead of GeLU.
//
// Differences from microgpt.py, all for Tenstorrent tile efficiency (the
// algorithm is identical, only the sizes change): the embedding dim is a
// multiple of 32 and attention is computed over the full padded sequence with a
// causal mask (rather than a scalar per-position KV-cache loop).

#include <fmt/format.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <cmath>
#include <core/ttnn_all_includes.hpp>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/embedding_module.hpp"
#include "modules/linear_module.hpp"
#include "modules/multi_head_attention.hpp"
#include "modules/positional_embeddings.hpp"
#include "modules/rms_norm_module.hpp"
#include "ops/binary_ops.hpp"
#include "ops/losses.hpp"
#include "ops/unary_ops.hpp"
#include "optimizers/adamw.hpp"
#include "optimizers/muon.hpp"

namespace {

// ----------------------------------------------------------------------------
// Configuration. Tiny by design -- this is the essence, not the efficiency.
// ----------------------------------------------------------------------------
struct MicroGptConfig {
    uint32_t n_layer = 1;       // depth (microgpt.py uses 1)
    uint32_t n_embd = 128;      // width; multiple of 32 for Tenstorrent tiles
    uint32_t n_head = 4;        // head_dim = n_embd / n_head = 32 (one tile)
    uint32_t block_size = 32;   // max context (longest name is ~15 chars); mult of 32
    uint32_t num_steps = 2000;  // training steps
    uint32_t batch_size = 64;   // documents (names) per step
    float learning_rate = 0.01F;
    float temperature = 0.7F;  // sampling creativity
};

// ----------------------------------------------------------------------------
// Dataset + tokenizer. Each document is a name; characters become token ids
// 0..n-1 and a special Beginning-of-Sequence (BOS) token wraps every name.
// ----------------------------------------------------------------------------
struct NamesData {
    std::vector<std::string> docs;
    std::vector<char> uchars;       // sorted unique characters
    std::map<char, uint32_t> ctoi;  // char -> token id
    uint32_t bos = 0;               // BOS token id
    uint32_t vocab_size = 0;

    // tokens for one name, surrounded by BOS on both sides: [BOS] c0 c1 ... [BOS]
    std::vector<uint32_t> tokenize(const std::string& name) const {
        std::vector<uint32_t> tokens;
        tokens.reserve(name.size() + 2);
        tokens.push_back(bos);
        for (char ch : name) {
            tokens.push_back(ctoi.at(ch));
        }
        tokens.push_back(bos);
        return tokens;
    }
};

NamesData load_names(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error(fmt::format("Could not open names file: {}", path));
    }
    NamesData data;
    std::set<char> charset;
    std::string line;
    while (std::getline(file, line)) {
        // strip trailing whitespace/CR
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        data.docs.push_back(line);
        for (char ch : line) {
            charset.insert(ch);
        }
    }
    data.uchars.assign(charset.begin(), charset.end());
    for (uint32_t i = 0; i < data.uchars.size(); ++i) {
        data.ctoi[data.uchars[i]] = i;
    }
    data.bos = static_cast<uint32_t>(data.uchars.size());
    data.vocab_size = data.bos + 1U;

    std::mt19937 rng(42);
    std::shuffle(data.docs.begin(), data.docs.end(), rng);
    fmt::print("num docs: {}, vocab size: {}\n", data.docs.size(), data.vocab_size);
    return data;
}

// ----------------------------------------------------------------------------
// A single transformer block: RMSNorm -> Attention -> residual, then
// RMSNorm -> Linear -> ReLU -> Linear -> residual. No biases in the MLP.
// ----------------------------------------------------------------------------
class MicroBlock : public ttml::autograd::ModuleBase {
    std::shared_ptr<ttml::modules::RMSNormLayer> m_norm1;
    std::shared_ptr<ttml::modules::MultiHeadAttention> m_attn;
    std::shared_ptr<ttml::modules::RMSNormLayer> m_norm2;
    std::shared_ptr<ttml::modules::LinearLayer> m_fc1;
    std::shared_ptr<ttml::modules::LinearLayer> m_fc2;

public:
    MicroBlock(uint32_t n_embd, uint32_t n_head) {
        m_norm1 = std::make_shared<ttml::modules::RMSNormLayer>(n_embd);
        m_attn = std::make_shared<ttml::modules::MultiHeadAttention>(n_embd, n_head, /* dropout */ 0.0F);
        m_norm2 = std::make_shared<ttml::modules::RMSNormLayer>(n_embd);
        m_fc1 = std::make_shared<ttml::modules::LinearLayer>(n_embd, 4U * n_embd, /* has_bias */ false);
        m_fc2 = std::make_shared<ttml::modules::LinearLayer>(4U * n_embd, n_embd, /* has_bias */ false);

        create_name("micro_block");
        register_module(m_norm1, "norm1");
        register_module(m_attn, "attn");
        register_module(m_norm2, "norm2");
        register_module(m_fc1, "fc1");
        register_module(m_fc2, "fc2");
    }

    ttml::autograd::TensorPtr operator()(
        const ttml::autograd::TensorPtr& input, const ttml::autograd::TensorPtr& mask) override {
        auto residual = input;
        auto x = (*m_norm1)(input);
        x = (*m_attn)(x, mask);
        x = ttml::ops::add(x, residual);

        residual = x;
        auto h = (*m_norm2)(x);
        h = (*m_fc1)(h);
        h = ttml::ops::relu(h);
        h = (*m_fc2)(h);
        return ttml::ops::add(h, residual);
    }
};

// ----------------------------------------------------------------------------
// The model: token + position embeddings, an initial RMSNorm, the blocks, and
// a linear head to logits over the vocabulary.
// ----------------------------------------------------------------------------
class MicroGpt : public ttml::autograd::ModuleBase {
    std::shared_ptr<ttml::modules::Embedding> m_tok_emb;
    std::shared_ptr<ttml::modules::TrainablePositionalEmbedding> m_pos_emb;
    std::shared_ptr<ttml::modules::RMSNormLayer> m_initial_norm;
    std::vector<std::shared_ptr<MicroBlock>> m_blocks;
    std::shared_ptr<ttml::modules::LinearLayer> m_lm_head;

public:
    MicroGpt(const MicroGptConfig& config, uint32_t vocab_size) {
        const uint32_t vocab_padded = (vocab_size + 31U) / 32U * 32U;

        m_tok_emb = std::make_shared<ttml::modules::Embedding>(vocab_padded, config.n_embd);
        m_pos_emb =
            std::make_shared<ttml::modules::TrainablePositionalEmbedding>(ttml::modules::PositionalEmbeddingConfig{
                .embedding_dim = config.n_embd,
                .sequence_length = config.block_size,
                .dropout_prob = 0.0F,
                .use_dropout_seed_per_device = false});
        m_initial_norm = std::make_shared<ttml::modules::RMSNormLayer>(config.n_embd);
        for (uint32_t i = 0; i < config.n_layer; ++i) {
            m_blocks.push_back(std::make_shared<MicroBlock>(config.n_embd, config.n_head));
        }
        // Head emits a tile-aligned number of logits; the extra padded classes
        // are never valid targets and are ignored at sampling time.
        m_lm_head = std::make_shared<ttml::modules::LinearLayer>(config.n_embd, vocab_padded, /* has_bias */ false);

        create_name("microgpt");
        register_module(m_tok_emb, "tok_emb");
        register_module(m_pos_emb, "pos_emb");
        register_module(m_initial_norm, "initial_norm");
        for (uint32_t i = 0; i < m_blocks.size(); ++i) {
            register_module(m_blocks[i], fmt::format("block_{}", i));
        }
        register_module(m_lm_head, "lm_head");
    }

    ttml::autograd::TensorPtr operator()(
        const ttml::autograd::TensorPtr& tokens, const ttml::autograd::TensorPtr& mask) override {
        auto x = (*m_tok_emb)(tokens);
        x = (*m_pos_emb)(x);
        x = (*m_initial_norm)(x);  // not redundant: matters for backward via the residual stream
        for (auto& block : m_blocks) {
            x = (*block)(x, mask);
        }
        return (*m_lm_head)(x);
    }
};

// Lower-triangular causal mask of shape [1, 1, S, S] (1 = attend, 0 = masked).
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

// Assemble one batch of inputs and shifted targets, padding short names with BOS.
std::pair<std::vector<uint32_t>, std::vector<uint32_t>> make_batch(
    const NamesData& data, const std::vector<size_t>& doc_indices, uint32_t block_size) {
    std::vector<uint32_t> inputs;
    std::vector<uint32_t> targets;
    inputs.reserve(doc_indices.size() * block_size);
    targets.reserve(doc_indices.size() * block_size);

    for (size_t doc_idx : doc_indices) {
        auto tokens = data.tokenize(data.docs[doc_idx]);
        // padded[0..block_size] (one extra for the shifted target), BOS-filled.
        std::vector<uint32_t> padded(block_size + 1U, data.bos);
        for (size_t i = 0; i < tokens.size() && i < padded.size(); ++i) {
            padded[i] = tokens[i];
        }
        for (uint32_t pos = 0; pos < block_size; ++pos) {
            inputs.push_back(padded[pos]);
            targets.push_back(padded[pos + 1U]);
        }
    }
    return {inputs, targets};
}

uint32_t sample_from_logits(
    const std::vector<float>& logits, uint32_t vocab_size, float temperature, std::mt19937& rng) {
    std::vector<float> probs(vocab_size);
    float max_logit = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < vocab_size; ++i) {
        max_logit = std::max(max_logit, logits[i] / temperature);
    }
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

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"microgpt: tiny GPT on the names dataset, running on Tenstorrent"};

    MicroGptConfig config;
    std::string data_path = std::string(DATA_FOLDER) + "/names.txt";
    std::string optimizer_type = "adamw";  // or "muon"
    uint32_t num_samples = 20;
    app.add_option("-d,--data", data_path, "Path to names.txt")->default_val(data_path);
    app.add_option("-o,--optimizer", optimizer_type, "Optimizer: adamw or muon")->default_val(optimizer_type);
    app.add_option("-s,--steps", config.num_steps, "Training steps")->default_val(config.num_steps);
    app.add_option("-b,--batch_size", config.batch_size, "Batch size")->default_val(config.batch_size);
    app.add_option("-l,--lr", config.learning_rate, "Learning rate")->default_val(config.learning_rate);
    app.add_option("--samples", num_samples, "Number of names to generate")->default_val(num_samples);
    CLI11_PARSE(app, argc, argv);

    auto& ctx = ttml::autograd::ctx();
    ctx.open_device();
    auto* device = &ctx.get_device();

    // Let there be a Dataset and a model.
    auto data = load_names(data_path);
    auto model = std::make_shared<MicroGpt>(config, data.vocab_size);
    fmt::print(
        "MicroGPT: n_layer={}, n_embd={}, n_head={}, block_size={}\n",
        config.n_layer,
        config.n_embd,
        config.n_head,
        config.block_size);

    // Let there be an optimizer (Adam, blessed; or Muon, the orthogonalizer).
    std::unique_ptr<ttml::optimizers::OptimizerBase> optimizer;
    if (optimizer_type == "muon") {
        ttml::optimizers::MuonConfig muon_config;
        muon_config.lr = config.learning_rate;
        optimizer = std::make_unique<ttml::optimizers::Muon>(model->parameters(), muon_config);
        fmt::print("Optimizer: Muon (lr={})\n", muon_config.lr);
    } else {
        ttml::optimizers::AdamWConfig adamw_config;
        adamw_config.lr = config.learning_rate;
        optimizer = std::make_unique<ttml::optimizers::AdamW>(model->parameters(), adamw_config);
        fmt::print("Optimizer: AdamW (lr={})\n", adamw_config.lr);
    }

    auto mask = make_causal_mask(config.block_size, device);

    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> doc_dist(0, data.docs.size() - 1U);

    // Repeat in sequence.
    for (uint32_t step = 0; step < config.num_steps; ++step) {
        std::vector<size_t> doc_indices(config.batch_size);
        for (auto& idx : doc_indices) {
            idx = doc_dist(rng);
        }
        auto [inputs, targets] = make_batch(data, doc_indices, config.block_size);

        auto input_tensor = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
            inputs, ttnn::Shape({config.batch_size, 1, 1, config.block_size}), device, ttnn::Layout::ROW_MAJOR));
        auto target_tensor = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
            targets, ttnn::Shape({config.batch_size, config.block_size}), device, ttnn::Layout::ROW_MAJOR));

        optimizer->zero_grad();
        auto logits = (*model)(input_tensor, mask);
        auto loss = ttml::ops::cross_entropy_loss(logits, target_tensor);
        float loss_value = ttml::core::to_vector(loss->get_value())[0];
        loss->backward();
        optimizer->step();
        ctx.reset_graph();

        if ((step + 1U) % 50U == 0U || step == 0U) {
            fmt::print("step {:4d} / {:4d} | loss {:.4f}\n", step + 1U, config.num_steps, loss_value);
        }
    }

    // Inference: may the model babble new names back to us.
    fmt::print("\n--- inference (new, hallucinated names) ---\n");
    model->eval();
    for (uint32_t s = 0; s < num_samples; ++s) {
        std::vector<uint32_t> generated;  // generated character token ids
        for (uint32_t pos = 0; pos < config.block_size - 1U; ++pos) {
            // Build the current context: BOS followed by what we've generated, BOS-padded.
            std::vector<uint32_t> ctx_tokens(config.block_size, data.bos);
            for (uint32_t i = 0; i < generated.size(); ++i) {
                ctx_tokens[i + 1U] = generated[i];
            }
            auto ctx_tensor = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
                ctx_tokens, ttnn::Shape({1, 1, 1, config.block_size}), device, ttnn::Layout::ROW_MAJOR));

            auto logits = (*model)(ctx_tensor, mask);
            auto logits_vec = ttml::core::to_vector(logits->get_value());  // [block_size * logits_width]
            ctx.reset_graph();

            // Logits for the next token are at the current last filled position.
            // Rows are logits_width wide (tile-aligned); only the first
            // vocab_size entries are real tokens.
            const uint32_t logits_width = (data.vocab_size + 31U) / 32U * 32U;
            const uint32_t last_pos = static_cast<uint32_t>(generated.size());  // BOS is at index 0
            std::vector<float> next_logits(
                logits_vec.begin() + static_cast<long>(last_pos) * logits_width,
                logits_vec.begin() + static_cast<long>(last_pos) * logits_width + data.vocab_size);

            uint32_t next_token = sample_from_logits(next_logits, data.vocab_size, config.temperature, rng);
            if (next_token == data.bos) {
                break;
            }
            generated.push_back(next_token);
        }

        std::string name;
        for (uint32_t token : generated) {
            name.push_back(data.uchars[token]);
        }
        fmt::print("sample {:2d}: {}\n", s + 1U, name);
    }

    ctx.close_device();
    return 0;
}
