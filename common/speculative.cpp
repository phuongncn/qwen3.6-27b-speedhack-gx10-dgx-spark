#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"
#include "suffix-tree.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <map>
#include <queue>
#include <unordered_map>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::vector<enum common_speculative_type> common_speculative_types = {
    COMMON_SPECULATIVE_TYPE_NONE,
    COMMON_SPECULATIVE_TYPE_DRAFT,
    COMMON_SPECULATIVE_TYPE_EAGLE3,
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,
    COMMON_SPECULATIVE_TYPE_SUFFIX,
    COMMON_SPECULATIVE_TYPE_COPYSPEC,
    COMMON_SPECULATIVE_TYPE_RECYCLE,
    COMMON_SPECULATIVE_TYPE_DFLASH
};

const std::map<std::string, enum common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft",         COMMON_SPECULATIVE_TYPE_DRAFT},
    {"eagle3",        COMMON_SPECULATIVE_TYPE_EAGLE3},
    {"ngram_simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram_map_k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram_map_k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram_mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram_cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE},
    {"suffix",        COMMON_SPECULATIVE_TYPE_SUFFIX},
    {"copyspec",      COMMON_SPECULATIVE_TYPE_COPYSPEC},
    {"recycle",       COMMON_SPECULATIVE_TYPE_RECYCLE},
    {"dflash",        COMMON_SPECULATIVE_TYPE_DFLASH}
};

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_DBG("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_DBG("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (
        llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)
    ) {
        LOG_DBG("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_state
struct common_speculative_state {
    const enum common_speculative_type type;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_state(enum common_speculative_type type) : type(type) {}

    virtual ~common_speculative_state() = default;

    virtual void begin(const llama_tokens & prompt) = 0;

    virtual void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) = 0;

    virtual void draft_tree(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            int tree_budget,
            common_speculative_tree & tree) {
        // default: flat draft, no tree
        llama_tokens result;
        draft(params, prompt_tgt, id_last, result);
        tree.n_nodes = (int)result.size();
        tree.tokens = std::move(result);
        tree.parents.resize(tree.n_nodes + 1);
        tree.parents[0] = -1;
        for (int i = 0; i < tree.n_nodes; ++i) {
            tree.parents[i + 1] = i; // linear chain
            tree.depths.push_back(i + 1);
        }
        // build child_maps + visibility for linear chain
        tree.child_maps.resize(tree.n_nodes + 1);
        for (int i = 0; i < tree.n_nodes; ++i) {
            tree.child_maps[i][tree.tokens[i]] = i + 1;
        }
        int n = tree.n_nodes + 1;
        tree.visibility.assign(n * n, false);
        for (int i = 0; i < n; ++i) {
            // each node sees itself and all ancestors
            int cur = i;
            while (cur >= 0) {
                tree.visibility[i * n + cur] = true;
                cur = tree.parents[cur];
            }
        }
        GGML_UNUSED(tree_budget);
    }

    virtual void accept(uint16_t n_accepted) = 0;

    // called after verification decode with logits still in ctx
    // batch_tokens: tokens that were in the batch [id_last, draft0, draft1, ...]
    // n_accepted: how many were accepted (ids.size(), including the bonus token)
    virtual void update_logits(llama_context * /*ctx*/, const llama_tokens & /*batch_tokens*/, int /*n_accepted*/) {}
};

struct common_speculative_state_draft : public common_speculative_state {
    llama_context * ctx_tgt; // only used for retokenizing from ctx_dft
    llama_context * ctx_dft;

    common_sampler * smpl;

    llama_batch  batch;
    llama_tokens prompt_dft;

    bool vocab_cmpt = true; // whether retokenization is needed
    std::unordered_map<std::string, std::string> vocab_map;

    common_speculative_state_draft(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_dft,
            const std::vector<std::pair<std::string, std::string>> & replacements)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
    {
        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
        smpl = nullptr;

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }
        {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        }

        vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        LOG_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_WRN("the target and draft vocabs are not compatible - tokens will be translated between the two\n");

            for (const auto & pair : replacements) {
                vocab_map[pair.first] = pair.second;
            }
        }
    }

    ~common_speculative_state_draft() override {
        llama_perf_context_print(ctx_dft);

        llama_free(ctx_dft);

        common_sampler_free(smpl);

        llama_batch_free(batch);
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        auto * spec = this;

        auto & batch      = spec->batch;
        auto & ctx_tgt    = spec->ctx_tgt;
        auto & ctx_dft    = spec->ctx_dft;
        auto & smpl       = spec->smpl;
        auto & prompt_dft = spec->prompt_dft;

        auto * mem_dft = llama_get_memory(ctx_dft);

        int reuse_i = 0;
        int reuse_n = 0;

        const int n_ctx = llama_n_ctx(ctx_dft) - params.n_max;

        llama_tokens prompt_cnv;
        if (!spec->vocab_cmpt) {
            std::string text;

            text = common_detokenize(ctx_tgt, prompt_tgt, true);
            text = replace_to_dft(text);

            LOG_DBG("%s: main->draft detokenized string: '%s'\n", __func__, text.c_str());

            prompt_cnv = common_tokenize(ctx_dft, text, false, true);

            // convert id_last to draft vocab. llama_detokenize is called directly to avoid an allocation
            const auto * model_tgt = llama_get_model(ctx_tgt);
            const auto * vocab_tgt = llama_model_get_vocab(model_tgt);

            int32_t n_chars = llama_detokenize(vocab_tgt, &id_last, 1, nullptr, 0, false, false);
            GGML_ASSERT(n_chars < 0 && "failed to detokenize id_last");

            text.resize(-n_chars);
            llama_detokenize(vocab_tgt, &id_last, 1, text.data(), text.size(), false, false);
            text = replace_to_dft(text);

            LOG_DBG("main->draft detokenized id_last(%d): '%s'\n", id_last, text.c_str());
            id_last = common_tokenize(ctx_dft, text, false, true)[0];
        }

        const llama_tokens & prompt_cur = spec->vocab_cmpt ? prompt_tgt : prompt_cnv;

        const int i_start = std::max<int>(0, (int) prompt_cur.size() - n_ctx);

        // reuse as much as possible from the old draft context
        // ideally, the draft context should be as big as the target context and we will always reuse the entire prompt
        for (int i = 0; i < (int) prompt_dft.size(); ++i) {
            int cur = 0;
            while (i_start + cur < (int) prompt_cur.size() &&
                    i       + cur < (int) prompt_dft.size() &&
                    prompt_cur[i_start + cur] == prompt_dft[i + cur]) {
                cur++;
            }

            if ((cur >= 256 || n_ctx >= (int) prompt_cur.size()) && cur > reuse_n) {
                reuse_i = i;
                reuse_n = cur;
            }
        }

        LOG_DBG("%s: reuse_i = %d, reuse_n = %d, prompt = %d\n", __func__, reuse_i, reuse_n, (int) prompt_dft.size());

        result.clear();
        result.reserve(params.n_max);

        if (reuse_n == 0) {
            llama_memory_clear(mem_dft, false);
            prompt_dft.clear();
        } else {
            // this happens when a previous draft has been discarded (for example, due to being too small), but the
            // target model agreed with it. in this case, we simply pass back the previous results to save compute
            if (reuse_i + reuse_n < (int) prompt_dft.size() && prompt_dft[reuse_i + reuse_n] == id_last) {
                for (int i = reuse_i + reuse_n + 1; i < (int) prompt_dft.size(); ++i) {
                    result.push_back(prompt_dft[i]);

                    if (params.n_max <= (int) result.size()) {
                        break;
                    }
                }

                return;
            }

            if (reuse_i > 0) {
                llama_memory_seq_rm (mem_dft, 0, 0, reuse_i);
                llama_memory_seq_add(mem_dft, 0, reuse_i, -1, -reuse_i);

                prompt_dft.erase(prompt_dft.begin(), prompt_dft.begin() + reuse_i);
            }

            if (reuse_n < (int) prompt_dft.size()) {
                llama_memory_seq_rm (mem_dft, 0, reuse_n, -1);
                prompt_dft.erase(prompt_dft.begin() + reuse_n, prompt_dft.end());
            }
        }

        // prepare a batch to evaluate any new tokens in the prompt
        common_batch_clear(batch);

        for (size_t i = i_start + reuse_n; i < prompt_cur.size(); ++i) {
            //LOG_DBG("i = %d, i_start = %d, reuse_n = %d, i - i_start = %d, id = %6d\n", i, i_start, reuse_n, i - i_start, prompt_cur[i]);
            common_batch_add(batch, prompt_cur[i], i - i_start, { 0 }, false);

            prompt_dft.push_back(prompt_cur[i]);
        }

        // we should rarely end-up here during normal decoding
        if (batch.n_tokens > 0) {
            //LOG_DBG("%s: draft prompt batch: %s\n", __func__, string_from(ctx, batch).c_str());

            llama_decode(ctx_dft, batch);
        }

        const llama_pos n_past = prompt_dft.size();

        LOG_DBG("%s: n_past = %d\n", __func__, n_past);

        common_batch_clear(batch);
        common_batch_add  (batch, id_last, n_past, { 0 }, true);

        prompt_dft.push_back(id_last);

        LOG_DBG("%s: draft prompt: %s\n", __func__, string_from(ctx_dft, prompt_dft).c_str());

        llama_decode(ctx_dft, batch);

        common_sampler_reset(smpl);

        // sample n_draft tokens from the draft model
        for (int i = 0; i < params.n_max; ++i) {
            common_batch_clear(batch);

            common_sampler_sample(smpl, ctx_dft, 0, true);

            const auto * cur_p = common_sampler_get_candidates(smpl, true);

            for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                LOG_DBG(" - draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                        k, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
            }

            // add drafted token for each sequence
            const llama_token id = cur_p->data[0].id;

            common_sampler_accept(smpl, id, true);

            result.push_back(id);

            if (params.n_max <= (int) result.size()) {
                break;
            }

            // only collect very high-confidence draft tokens
            if (cur_p->data[0].p < params.p_min) {
                break;
            }

            common_batch_add(batch, id, n_past + i + 1, { 0 }, true);

            // evaluate the drafted tokens on the draft model
            llama_decode(ctx_dft, batch);

            prompt_dft.push_back(id);
        }

        if (!spec->vocab_cmpt) {
            std::string detokenized = common_detokenize(ctx_dft, result, true);
            detokenized = replace_to_tgt(detokenized);
            LOG_DBG("draft->main detokenized string: '%s'\n", detokenized.c_str());
            result = common_tokenize(ctx_tgt, detokenized, false, true);
            if (result.size() > (size_t)params.n_max) {
                result.resize(params.n_max);
            }
        }
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }

    std::string replace_to_dft(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.first);
            while (pos != std::string::npos) {
                result.replace(pos, pair.first.length(), pair.second);
                pos = result.find(pair.first, pos + pair.second.length());
            }
        }

        return result;
    }

    std::string replace_to_tgt(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.second);
            while (pos != std::string::npos) {
                result.replace(pos, pair.second.length(), pair.first);
                pos = result.find(pair.second, pos + pair.first.length());
            }
        }

        return result;
    }
};

struct common_speculative_state_eagle3 : public common_speculative_state {
    common_speculative_state_eagle3(enum common_speculative_type type) : common_speculative_state(type) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & draft_tokens) override {
        // TODO: implement
        GGML_UNUSED(params);
        GGML_UNUSED(prompt_tgt);
        GGML_UNUSED(id_last);
        GGML_UNUSED(draft_tokens);
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_state_ngram_simple : public common_speculative_state {
    common_ngram_simple_config config;

    common_speculative_state_ngram_simple(
            enum common_speculative_type type,
            common_ngram_simple_config config)
        : common_speculative_state(type), config(config) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {

        result = common_ngram_simple_draft(config, prompt_tgt, id_last);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }
};

struct common_speculative_state_ngram_map_k : public common_speculative_state {
    // draft ngram map for speculative decoding without draft model
    common_ngram_map map;

    common_speculative_state_ngram_map_k(
            enum common_speculative_type type,
            common_ngram_map map)
        : common_speculative_state(type), map(std::move(map)) {}

    void begin(const llama_tokens & prompt) override {
        common_ngram_map_begin(map, prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        common_ngram_map_draft(map, prompt_tgt, id_last, result);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted) override {
        common_ngram_map_accept(map, n_accepted);
    }
};

struct common_speculative_state_ngram_mod : public common_speculative_state {
    common_ngram_mod & mod;

    // the last position in the prompt that was added to the ngram container
    size_t i_last = 0;

    // length of the last drafted n‑gram (number of tokens returned by draft)
    size_t n_draft_last = 0;

    // consecutive accept rounds with low acceptance fraction (< 0.5)
    int n_low = 0;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    common_speculative_state_ngram_mod(enum common_speculative_type type, common_ngram_mod & mod)
        : common_speculative_state(type), mod(mod), verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));
    }

    void begin(const llama_tokens & prompt) override {
        i_last = 0;

        n_draft_last = 0;

        const size_t n = mod.get_n();

        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(params);

        n_draft_last = 0;

        const size_t cur_len = prompt_tgt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (i_last + 32 < cur_len) {
            for (size_t i = i_last; i < cur_len - n; ++i) {
                mod.add(prompt_tgt.data() + i);
            }

            i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt_tgt[cur_len - n + 1 + i];
        }
        result[n - 1] = id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n‑gram for later acceptance analysis
        n_draft_last = result.size();
    }

    void accept(uint16_t n_accepted) override {
        if (verbose) {
            LOG_INF("%s: accepted %d tokens from %zu drafted tokens\n", __func__, n_accepted, n_draft_last);
        }

        // compute acceptance fraction if we have a recorded draft length
        if (n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)n_draft_last;
            if (f_acc < 0.5) {
                n_low++;
                if (n_low >= 3) {
                    LOG_WRN("%s: low acceptance streak (%d) – resetting ngram_mod\n", __func__, n_low);

                    mod.reset();
                    n_low = 0;
                }
            } else {
                n_low = 0;
            }
        }
    }
};

struct common_speculative_state_ngram_cache : public common_speculative_state {
    uint16_t n_draft;
    bool save_dynamic;
    bool save_static;

    common_ngram_cache ngram_cache_context;
    common_ngram_cache ngram_cache_dynamic;
    common_ngram_cache ngram_cache_static;

    size_t cache_size = 0; // number of tokens in n-gram cache

    common_speculative_state_ngram_cache(
            const enum common_speculative_type type,
            const std::string & path_static,
            const std::string & path_dynamic,
            uint16_t            n_draft,
            bool                save_dynamic,
            bool                save_static)
        : common_speculative_state(type)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        if (!path_static.empty()) {
            try {
                ngram_cache_static = common_ngram_cache_load(path_static);
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);
            } catch (...) {
                LOG_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(params);

        if (cache_size < prompt_tgt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt_tgt.size() + 1 - cache_size);
            for (size_t j = cache_size; j < prompt_tgt.size(); ++j) {
                tokens_new.push_back(prompt_tgt[j]);
            }
            tokens_new.push_back(id_last); // add the last token

            // Update context ngram cache with new prompt_tgt:
            common_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            cache_size = prompt_tgt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt_tgt.size() + 1);
        for (size_t j = 0; j < prompt_tgt.size(); ++j) {
            inp.push_back(prompt_tgt[j]);
        }
        inp.push_back(id_last);

        result.push_back(id_last);

        common_ngram_cache_draft(inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                ngram_cache_context,
                ngram_cache_dynamic,
                ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    void accept(uint16_t n_accepted) override {
        // TODO: noop
        GGML_UNUSED(n_accepted);
    }
};

struct common_speculative_state_suffix : public common_speculative_state {
    SuffixTree tree;
    static constexpr int SEQ_ID = 1;

    int32_t max_depth;
    int32_t n_max;
    float   spec_factor;
    float   spec_offset;
    float   min_prob;

    size_t tree_size = 0;  // number of tokens fed to the tree (prompt_tgt.size() + 1)

    common_speculative_state_suffix(
            const enum common_speculative_type type,
            int32_t max_depth,
            int32_t n_max,
            float   spec_factor,
            float   spec_offset,
            float   min_prob)
        : common_speculative_state(type)
        , tree(max_depth)
        , max_depth(max_depth)
        , n_max(n_max)
        , spec_factor(spec_factor)
        , spec_offset(spec_offset)
        , min_prob(min_prob)
    {}

    void begin(const llama_tokens & prompt) override {
        tree = SuffixTree(max_depth);
        tree_size = 0;
        if (!prompt.empty()) {
            tree.extend(SEQ_ID, prompt.data(), prompt.size());
            tree_size = prompt.size();
        }
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(params);

        // feed new tokens to suffix tree (same pattern as ngram_cache)
        if (tree_size < prompt_tgt.size() + 1) {
            for (size_t j = tree_size; j < prompt_tgt.size(); ++j) {
                tree.append(SEQ_ID, prompt_tgt[j]);
            }
            tree.append(SEQ_ID, id_last);
            tree_size = prompt_tgt.size() + 1;
        }

        // build full context for pattern matching
        std::vector<int32_t> context;
        context.reserve(prompt_tgt.size() + 1);
        for (size_t i = 0; i < prompt_tgt.size(); i++) {
            context.push_back(prompt_tgt[i]);
        }
        context.push_back(id_last);

        if (context.size() < 2) { return; }

        SuffixDraft draft = tree.speculate(
            context.data(), context.size(),
            n_max, spec_factor, spec_offset, min_prob, false);

        for (size_t i = 0; i < draft.token_ids.size(); i++) {
            result.push_back(draft.token_ids[i]);
        }
    }

    void accept(uint16_t n_accepted) override {
        GGML_UNUSED(n_accepted);
    }
};

// CopySpec: draft by copying matching subsequences from the prompt context.
// Builds a rolling-hash index of all gamma-length windows in the prompt.
// On each draft call, hashes the last gamma tokens of output and looks up matches.
struct common_speculative_state_copyspec : public common_speculative_state {
    static constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
    static constexpr uint64_t FNV_PRIME  = 1099511628211ULL;

    int32_t gamma; // window size for matching

    // hash of gamma-length window -> position after the window in the prompt
    std::unordered_multimap<uint64_t, int32_t> index;
    llama_tokens prompt_tokens;

    common_speculative_state_copyspec(enum common_speculative_type type, int32_t gamma)
        : common_speculative_state(type)
        , gamma(gamma)
    {}

    static uint64_t hash_window(const llama_token * tokens, int32_t len) {
        uint64_t h = FNV_OFFSET;
        for (int32_t i = 0; i < len; i++) {
            h ^= (uint64_t)(uint32_t)tokens[i];
            h *= FNV_PRIME;
        }
        return h;
    }

    void begin(const llama_tokens & prompt) override {
        index.clear();
        prompt_tokens = prompt;
        if ((int32_t)prompt.size() <= gamma) {
            return;
        }
        for (int32_t i = 0; i <= (int32_t)prompt.size() - gamma; i++) {
            uint64_t h = hash_window(prompt.data() + i, gamma);
            index.emplace(h, i + gamma);
        }
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        // build the full context (prompt_tgt + id_last)
        const int32_t ctx_len = (int32_t)prompt_tgt.size() + 1;
        if (ctx_len < gamma) {
            return;
        }

        // hash the last gamma tokens of context
        std::vector<llama_token> window(gamma);
        const int32_t start = ctx_len - gamma;
        for (int32_t i = 0; i < gamma; i++) {
            const int32_t pos = start + i;
            window[i] = (pos < (int32_t)prompt_tgt.size()) ? prompt_tgt[pos] : id_last;
        }
        uint64_t h = hash_window(window.data(), gamma);

        // find longest match in prompt
        int32_t best_pos = -1;
        int32_t best_len = 0;
        auto range = index.equal_range(h);
        for (auto it = range.first; it != range.second; ++it) {
            int32_t pos = it->second;
            // verify hash match (collision check)
            if (pos < gamma || pos > (int32_t)prompt_tokens.size()) {
                continue;
            }
            bool match = true;
            for (int32_t j = 0; j < gamma; j++) {
                if (prompt_tokens[pos - gamma + j] != window[j]) {
                    match = false;
                    break;
                }
            }
            if (!match) {
                continue;
            }
            // count how many tokens we can copy from this position
            int32_t avail = std::min(params.n_max, (int32_t)prompt_tokens.size() - pos);
            if (avail > best_len) {
                best_len = avail;
                best_pos = pos;
            }
        }

        if (best_pos < 0) {
            return;
        }

        for (int32_t i = 0; i < best_len; i++) {
            result.push_back(prompt_tokens[best_pos + i]);
        }
    }

    void accept(uint16_t n_accepted) override {
        GGML_UNUSED(n_accepted);
    }
};

// Token Recycling: adjacency matrix tracking top-k successors per token.
// Seeded from observed bigrams, then updated from model logits after each
// verification decode. Logit-based entries have much higher scores and
// dominate the adjacency matrix after the first few iterations.
struct common_speculative_state_recycle : public common_speculative_state {
    int32_t k; // top-k successors per token

    // adjacency: token -> vector of (score, successor) pairs, sorted by score descending
    // scores: bigram observations use small integer counts (1, 2, ...),
    //         logit-derived entries use logit values (typically 10-30+ for top tokens)
    std::unordered_map<llama_token, std::vector<std::pair<float, llama_token>>> adj;

    size_t n_fed = 0;
    int32_t n_vocab = 0;

    common_speculative_state_recycle(enum common_speculative_type type, int32_t k)
        : common_speculative_state(type)
        , k(k)
    {}

    void set_successors(llama_token tok, const float * logits, int32_t vocab_size) {
        // partial sort to find top-k logits
        std::vector<std::pair<float, llama_token>> top(k, std::make_pair(-INFINITY, (llama_token)-1));
        for (int32_t i = 0; i < vocab_size; i++) {
            if (logits[i] > top[k-1].first) {
                top[k-1] = std::make_pair(logits[i], (llama_token)i);
                // bubble up
                for (int32_t j = k-2; j >= 0; j--) {
                    if (top[j+1].first > top[j].first) {
                        std::swap(top[j], top[j+1]);
                    } else {
                        break;
                    }
                }
            }
        }
        // remove unfilled slots
        while (!top.empty() && top.back().second < 0) {
            top.pop_back();
        }
        adj[tok] = std::move(top);
    }

    void add_bigram(llama_token a, llama_token b) {
        auto & succs = adj[a];
        for (size_t i = 0; i < succs.size(); i++) {
            if (succs[i].second == b) {
                succs[i].first += 1.0f;
                while (i > 0 && succs[i].first > succs[i-1].first) {
                    std::swap(succs[i], succs[i-1]);
                    i--;
                }
                return;
            }
        }
        if ((int32_t)succs.size() < k) {
            succs.push_back(std::make_pair(1.0f, b));
        }
    }

    void begin(const llama_tokens & prompt) override {
        adj.clear();
        n_fed = 0;
        for (size_t i = 0; i + 1 < prompt.size(); i++) {
            add_bigram(prompt[i], prompt[i + 1]);
        }
        n_fed = prompt.size();
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        // feed new bigrams from generated tokens
        if (n_fed < prompt_tgt.size() + 1) {
            size_t start = (n_fed > 0) ? n_fed - 1 : 0;
            for (size_t i = start; i < prompt_tgt.size(); i++) {
                llama_token next = (i + 1 < prompt_tgt.size()) ? prompt_tgt[i + 1] : id_last;
                add_bigram(prompt_tgt[i], next);
            }
            n_fed = prompt_tgt.size() + 1;
        }

        // greedy walk through adjacency matrix
        llama_token cur = id_last;
        for (int32_t i = 0; i < params.n_max; i++) {
            auto it = adj.find(cur);
            if (it == adj.end() || it->second.empty()) {
                break;
            }
            cur = it->second[0].second;
            result.push_back(cur);
        }
    }

    void accept(uint16_t n_accepted) override {
        GGML_UNUSED(n_accepted);
    }

    void update_logits(llama_context * ctx, const llama_tokens & batch_tokens, int n_accepted) override {
        if (n_vocab == 0) {
            const llama_model * model = llama_get_model(ctx);
            n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        }
        // update adjacency from logits for each position that had logits computed
        // batch_tokens[i] is the token at position i; logits at i predict its successor
        const int n_positions = std::min(n_accepted, (int)batch_tokens.size());
        for (int i = 0; i < n_positions; i++) {
            const float * logits = llama_get_logits_ith(ctx, i);
            if (logits) {
                set_successors(batch_tokens[i], logits, n_vocab);
            }
        }
    }
};

// DFlash block-diffusion speculative decoding
// Uses an external drafter model conditioned on target hidden states via KV injection
struct common_speculative_state_dflash : public common_speculative_state {
    llama_context * ctx_tgt;
    llama_context * ctx_dft;
    llama_model   * model_dft;

    int block_size;
    llama_token mask_token_id;
    int n_target_layers;
    int n_embd;
    int n_target_features;

    // accumulated fused target hidden states [n_embd, committed_len] per capture slot
    // we store raw hidden states per layer, then concatenate on demand
    std::vector<std::vector<float>> hidden_per_layer; // [n_target_layers][n_embd * committed_len]
    int committed_len = 0;

    // S3: cached interleaved concat_hidden — only new tokens get interleaved on each call
    std::vector<float> cached_concat;
    int cached_concat_len = 0; // how many tokens are already interleaved in cached_concat

    // A2: sliding window limit for drafter context (0 = unlimited)
    static constexpr int ctx_window = 512;

    llama_batch batch_dft;

    common_speculative_state_dflash(
            llama_context * ctx_tgt_,
            llama_context * ctx_dft_,
            llama_model   * model_dft_)
        : common_speculative_state(COMMON_SPECULATIVE_TYPE_DFLASH)
        , ctx_tgt(ctx_tgt_)
        , ctx_dft(ctx_dft_)
        , model_dft(model_dft_)
    {
        block_size        = llama_model_dflash_block_size(model_dft_);
        mask_token_id     = (llama_token) llama_model_dflash_mask_token_id(model_dft_);
        n_target_layers   = llama_model_dflash_n_target_layers(model_dft_);
        n_embd            = llama_model_n_embd(model_dft_);
        n_target_features = llama_model_dflash_n_target_features(model_dft_);

        hidden_per_layer.resize(n_target_layers);

        // tok_embd/output sharing must happen BEFORE context creation
        // (done in speculative-simple.cpp before common_speculative_init)

        // configure target context to capture hidden states
        std::vector<int32_t> capture_layers(n_target_layers);
        llama_model_dflash_target_layer_ids(model_dft_, capture_layers.data(), n_target_layers);
        llama_set_dflash_capture(ctx_tgt, capture_layers.data(), n_target_layers);

        batch_dft = llama_batch_init(block_size, 0, 1);

        LOG_INF("dflash: block_size=%d, mask_token=%d, n_target_layers=%d, n_embd=%d\n",
                block_size, mask_token_id, n_target_layers, n_embd);
    }

    ~common_speculative_state_dflash() override {
        llama_batch_free(batch_dft);
        llama_free(ctx_dft);
    }

    // called after initial prefill — extract hidden states from target
    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
        capture_target_hiddens();
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(prompt_tgt);

        const int n_draft = std::min(block_size - 1, params.n_max);
        if (committed_len == 0) {
            return;
        }

        const int64_t t0 = ggml_time_us();

        // S3: incrementally update cached interleaved concat_hidden
        // only interleave tokens from cached_concat_len..committed_len
        if (cached_concat_len < committed_len) {
            cached_concat.resize((size_t)n_target_features * committed_len);
            for (int layer = 0; layer < n_target_layers; ++layer) {
                for (int t = cached_concat_len; t < committed_len; ++t) {
                    memcpy(&cached_concat[(size_t)(layer * n_embd) + (size_t)t * n_target_features],
                           &hidden_per_layer[layer][(size_t)t * n_embd],
                           n_embd * sizeof(float));
                }
            }
            cached_concat_len = committed_len;
        }

        // A2: sliding window — limit context to last ctx_window tokens
        const float * cross_data = cached_concat.data();
        int cross_len = committed_len;
        if (ctx_window > 0 && committed_len > ctx_window) {
            cross_data = cached_concat.data() + (size_t)(committed_len - ctx_window) * n_target_features;
            cross_len = ctx_window;
        }

        const int64_t t1 = ggml_time_us();

        // set cross data on drafter context
        llama_set_cross_data(ctx_dft, cross_data, n_target_features, cross_len);

        // build drafter batch: [id_last, mask, mask, ..., mask]
        // positions are relative to the context window fed to the drafter
        common_batch_clear(batch_dft);
        common_batch_add(batch_dft, id_last, cross_len, { 0 }, true);
        for (int i = 1; i < block_size; ++i) {
            common_batch_add(batch_dft, mask_token_id, cross_len + i, { 0 }, true);
        }

        const int64_t t2 = ggml_time_us();

        // run drafter forward pass
        int ret = llama_decode(ctx_dft, batch_dft);
        if (ret != 0) {
            LOG_ERR("dflash: drafter decode failed with %d\n", ret);
            return;
        }

        const int64_t t3 = ggml_time_us();

        // read argmax tokens for positions 1..block_size-1 (skip position 0 = staged_first)
        {
            int32_t * argmax = llama_get_logits_argmax(ctx_dft);
            if (argmax) {
                // GPU argmax path — only 64 bytes transferred instead of 15.9MB
                for (int i = 1; i < block_size && (int) result.size() < n_draft; ++i) {
                    result.push_back((llama_token) argmax[i]);
                }
            } else {
                // fallback: CPU argmax over full vocab
                const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_dft));
                for (int i = 1; i < block_size && (int) result.size() < n_draft; ++i) {
                    float * logits = llama_get_logits_ith(ctx_dft, i);
                    if (!logits) {
                        break;
                    }
                    llama_token best = (llama_token)(std::max_element(logits, logits + n_vocab) - logits);
                    result.push_back(best);
                }
            }
        }

        const int64_t t4 = ggml_time_us();

        LOG_INF("dflash draft breakdown (ctx=%d): concat=%.1fms cross=%.1fms decode=%.1fms argmax=%.1fms total=%.1fms\n",
                committed_len,
                (t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3, (t4 - t3) / 1e3, (t4 - t0) / 1e3);
    }

    void accept(uint16_t n_accepted) override {
        GGML_UNUSED(n_accepted);
    }

    void draft_tree(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            int tree_budget,
            common_speculative_tree & tree) override {
        const int n_draft = std::min((int) params.n_max, block_size - 1);
        if (n_draft <= 0 || committed_len == 0) {
            return;
        }

        // run drafter forward pass (same as flat draft)
        // --- begin shared draft setup ---
        const int64_t t0 = ggml_time_us();

        if (cached_concat_len < committed_len) {
            cached_concat.resize((size_t)n_target_features * committed_len);
            for (int layer = 0; layer < n_target_layers; ++layer) {
                for (int t = cached_concat_len; t < committed_len; ++t) {
                    memcpy(&cached_concat[(size_t)(layer * n_embd) + (size_t)t * n_target_features],
                           &hidden_per_layer[layer][(size_t)t * n_embd],
                           n_embd * sizeof(float));
                }
            }
            cached_concat_len = committed_len;
        }

        const float * cross_data = cached_concat.data();
        int cross_len = committed_len;
        if (ctx_window > 0 && committed_len > ctx_window) {
            cross_data = cached_concat.data() + (size_t)(committed_len - ctx_window) * n_target_features;
            cross_len = ctx_window;
        }

        llama_set_cross_data(ctx_dft, cross_data, n_target_features, cross_len);

        common_batch_clear(batch_dft);
        common_batch_add(batch_dft, id_last, cross_len, { 0 }, true);
        for (int i = 1; i < block_size; ++i) {
            common_batch_add(batch_dft, mask_token_id, cross_len + i, { 0 }, true);
        }

        int ret = llama_decode(ctx_dft, batch_dft);
        if (ret != 0) {
            LOG_ERR("dflash: drafter decode failed with %d\n", ret);
            return;
        }
        // --- end shared draft setup ---

        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_dft));
        const int draft_horizon = std::min(n_draft, block_size - 1);
        const int topk = std::min(tree_budget, n_vocab);
        const int depth_limit = draft_horizon;

        // collect top-K log-probs at each position
        // top_log_probs[pos][rank], top_token_ids[pos][rank]
        std::vector<std::vector<float>>       top_log_probs(depth_limit);
        std::vector<std::vector<llama_token>> top_token_ids(depth_limit);

        std::vector<std::pair<float, llama_token>> scored(n_vocab);

        for (int pos = 0; pos < depth_limit; ++pos) {
            float * logits = llama_get_logits_ith(ctx_dft, pos + 1); // pos+1 because batch[0] = id_last
            if (!logits) break;

            for (int v = 0; v < n_vocab; ++v) {
                scored[v] = { logits[v], (llama_token)v };
            }
            std::partial_sort(scored.begin(), scored.begin() + topk, scored.end(),
                [](const auto & a, const auto & b) { return a.first > b.first; });

            // log-sum-exp for normalization
            float max_val = scored[0].first;
            float sum_exp = 0.0f;
            for (int v = 0; v < n_vocab; ++v) {
                sum_exp += expf(logits[v] - max_val);
            }
            float log_z = max_val + logf(sum_exp);

            top_log_probs[pos].resize(topk);
            top_token_ids[pos].resize(topk);
            for (int k = 0; k < topk; ++k) {
                top_log_probs[pos][k] = scored[k].first - log_z;
                top_token_ids[pos][k] = scored[k].second;
            }
        }

        // build tree: main path first (top-1 at each depth), then branches via heap
        // main path comes first in node order so DeltaNet processes it before branches
        tree.tokens.clear();
        tree.parents.clear();
        tree.depths.clear();
        tree.child_maps.clear();
        tree.visibility.clear();

        tree.parents.push_back(-1); // root parent
        tree.child_maps.push_back({}); // root child_map
        tree.n_nodes = 0;
        tree.main_path_len = 0;

        // phase 1: lay down the full main path (top-1 at each depth)
        {
            float cum_logw = 0.0f;
            int parent = 0; // virtual root
            for (int d = 1; d <= depth_limit && topk > 0 && tree.n_nodes < tree_budget; ++d) {
                llama_token token_id = top_token_ids[d - 1][0];
                cum_logw += top_log_probs[d - 1][0];

                int current_idx = tree.n_nodes + 1;
                tree.tokens.push_back(token_id);
                tree.parents.push_back(parent);
                tree.depths.push_back(d);
                tree.child_maps.push_back({});
                tree.child_maps[parent][token_id] = current_idx;
                tree.n_nodes++;

                parent = current_idx;
            }
            tree.main_path_len = tree.n_nodes;
        }

        // phase 2: add branch nodes via heap (remaining budget)
        if (tree.n_nodes < tree_budget && topk > 1) {
            struct heap_entry {
                float neg_logw;
                int parent_idx;
                int depth;
                int rank;
                float logw;

                bool operator>(const heap_entry & o) const { return neg_logw > o.neg_logw; }
            };

            std::priority_queue<heap_entry, std::vector<heap_entry>, std::greater<heap_entry>> heap;

            // seed heap with rank-1 siblings at each main-path node
            float cum_logw = 0.0f;
            for (int d = 1; d <= tree.main_path_len; ++d) {
                cum_logw += top_log_probs[d - 1][0];
                // sibling: same depth, rank 1, same parent as main-path node at depth d
                int mp_parent = tree.parents[d]; // parent of main-path node d (1-based)
                float sib_logw = cum_logw - top_log_probs[d - 1][0] + top_log_probs[d - 1][1];
                heap.push({ -sib_logw, mp_parent, d, 1, sib_logw });
            }

            while (!heap.empty() && tree.n_nodes < tree_budget) {
                auto entry = heap.top();
                heap.pop();

                int depth = entry.depth;
                int rank  = entry.rank;

                llama_token token_id = top_token_ids[depth - 1][rank];
                int current_idx = tree.n_nodes + 1;

                tree.tokens.push_back(token_id);
                tree.parents.push_back(entry.parent_idx);
                tree.depths.push_back(depth);
                tree.child_maps.push_back({});
                tree.child_maps[entry.parent_idx][token_id] = current_idx;
                tree.n_nodes++;

                // push next sibling (same depth, rank+1)
                if (rank + 1 < topk) {
                    float sib_logw = entry.logw - top_log_probs[depth - 1][rank]
                                                + top_log_probs[depth - 1][rank + 1];
                    heap.push({ -sib_logw, entry.parent_idx, depth, rank + 1, sib_logw });
                }

                // push child (depth+1, rank 0) — branch continuation
                if (depth < depth_limit) {
                    float child_logw = entry.logw + top_log_probs[depth][0];
                    heap.push({ -child_logw, current_idx, depth + 1, 0, child_logw });
                }
            }
        }

        // build visibility matrix [(n_nodes+1) × (n_nodes+1)]
        int n = tree.n_nodes + 1;
        tree.visibility.assign(n * n, false);
        tree.visibility[0] = true; // root sees itself
        for (int i = 1; i < n; ++i) {
            int parent = tree.parents[i];
            // inherit parent's visibility row
            for (int j = 0; j < i; ++j) {
                tree.visibility[i * n + j] = tree.visibility[parent * n + j];
            }
            tree.visibility[i * n + i] = true; // see itself
        }

        const int64_t t1 = ggml_time_us();
        LOG_INF("ddtree: built tree with %d nodes (%d main + %d branch, budget %d) in %.1fms\n",
                tree.n_nodes, tree.main_path_len, tree.n_nodes - tree.main_path_len,
                tree_budget, (t1 - t0) / 1e3);

        GGML_UNUSED(prompt_tgt);
    }

    // called after target verification decode — capture and append new hidden states
    void update_logits(llama_context * ctx, const llama_tokens & batch_tokens, int n_accepted) override {
        GGML_UNUSED(ctx);
        GGML_UNUSED(batch_tokens);
        // n_accepted includes the bonus token: [id_last, draft0, ..., draftN-1] → accepted count
        // the verification batch had (1 + n_draft) tokens
        // only the first n_accepted tokens' hidden states should be kept
        append_target_hiddens(n_accepted);
    }

private:
    // called after initial prefill — grab all hidden states
    void capture_target_hiddens() {
        int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
        if (n_slots == 0) {
            return;
        }

        int64_t n_tokens = llama_get_layer_hidden_n_tokens(ctx_tgt, 0);
        if (n_tokens <= 0) {
            return;
        }

        // replace hidden state cache entirely (initial capture)
        committed_len = (int) n_tokens;
        cached_concat_len = 0; // invalidate incremental cache
        for (int layer = 0; layer < n_target_layers && layer < n_slots; ++layer) {
            float * data = llama_get_layer_hidden(ctx_tgt, layer);
            int64_t embd = llama_get_layer_hidden_n_embd(ctx_tgt, layer);
            int64_t ntok = llama_get_layer_hidden_n_tokens(ctx_tgt, layer);

            hidden_per_layer[layer].resize(embd * ntok);
            if (data) {
                memcpy(hidden_per_layer[layer].data(), data, embd * ntok * sizeof(float));
            }
        }
    }

    // called after each verification decode — append only the accepted tokens' hidden states
    void append_target_hiddens(int n_accepted) {
        int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
        if (n_slots == 0 || n_accepted <= 0) {
            return;
        }

        for (int layer = 0; layer < n_target_layers && layer < n_slots; ++layer) {
            float * data = llama_get_layer_hidden(ctx_tgt, layer);
            int64_t embd = llama_get_layer_hidden_n_embd(ctx_tgt, layer);
            int64_t ntok = llama_get_layer_hidden_n_tokens(ctx_tgt, layer);

            if (!data || ntok <= 0) continue;

            // only keep the first n_accepted tokens from this batch
            int n_to_append = std::min(n_accepted, (int) ntok);

            size_t old_size = hidden_per_layer[layer].size();
            hidden_per_layer[layer].resize(old_size + embd * n_to_append);
            memcpy(hidden_per_layer[layer].data() + old_size, data, embd * n_to_append * sizeof(float));
        }

        committed_len += n_accepted;
    }
};

struct common_speculative {
    std::vector<std::unique_ptr<common_speculative_state>> impls;
    common_speculative_state * curr_impl = nullptr;
};

static common_ngram_map get_common_ngram_map(const common_speculative_config & config) {
    uint16_t size_key   = config.params.ngram_size_n;
    uint16_t size_value = config.params.ngram_size_m;
    bool     key_only   = (config.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
    uint16_t min_hits   = config.params.ngram_min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_state_ngram_cache create_state_ngram_cache(
        const std::string & path_static, const std::string & path_dynamic,
        const common_speculative_config & config) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_state_ngram_cache state(config.type, path_static, path_dynamic, n_draft, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str() {
    std::string result;
    for (size_t i = 0; i < common_speculative_types.size(); i++) {
        if (i > 0) {
            result += ", ";
        }
        result += common_speculative_type_to_str(common_speculative_types[i]);
    }
    return result;
}

std::string common_speculative_type_to_str(enum common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT:         return "draft";
        case COMMON_SPECULATIVE_TYPE_EAGLE3:        return "eagle3";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram_simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram_map_k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram_map_k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram_mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram_cache";
        case COMMON_SPECULATIVE_TYPE_SUFFIX:        return "suffix";
        case COMMON_SPECULATIVE_TYPE_COPYSPEC:      return "copyspec";
        case COMMON_SPECULATIVE_TYPE_RECYCLE:       return "recycle";
        default:                                    return "unknown";
    }
}

enum common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

bool common_speculative_is_compat(llama_context * ctx_tgt) {
    auto * mem = llama_get_memory(ctx_tgt);
    if (mem == nullptr) {
        return false;
    }

    bool res = true;

    llama_memory_clear(mem, true);

    // eval 2 tokens to check if the context is compatible
    std::vector<llama_token> tmp;
    tmp.push_back(0);
    tmp.push_back(0);

    int ret = llama_decode(ctx_tgt, llama_batch_get_one(tmp.data(), tmp.size()));
    if (ret != 0) {
        LOG_ERR("%s: llama_decode() failed: %d\n", __func__, ret);
        res = false;
        goto done;
    }

    // try to remove the last tokens
    if (!llama_memory_seq_rm(mem, 0, 1, -1)) {
        LOG_WRN("%s: the target context does not support partial sequence removal\n", __func__);
        res = false;
        goto done;
    }

done:
    llama_memory_clear(mem, true);
    llama_synchronize(ctx_tgt);

    return res;
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt) {
    llama_context * ctx_dft = nullptr;
    if (params.model_dft) {
        ctx_dft = llama_init_from_model(params.model_dft, params.cparams_dft);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s", "failed to create draft context\n");
            return nullptr;
        }
    }

    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        bool has_draft = !params.mparams_dft.path.empty();
        bool has_draft_eagle3 = false; // TODO PR-18039: if params.speculative.eagle3

        bool has_ngram_cache   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);
        bool has_ngram_simple  = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        bool has_ngram_map_k   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        bool has_ngram_map_k4v = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        bool has_ngram_mod     = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
        bool has_suffix        = (params.type == COMMON_SPECULATIVE_TYPE_SUFFIX);
        bool has_copyspec      = (params.type == COMMON_SPECULATIVE_TYPE_COPYSPEC);
        bool has_recycle       = (params.type == COMMON_SPECULATIVE_TYPE_RECYCLE);
        bool has_dflash        = (params.type == COMMON_SPECULATIVE_TYPE_DFLASH);

        // DFlash uses --model-draft but is NOT a standard draft model
        if (has_dflash) {
            has_draft = false;
        }

        // In a more complex implementation we could use the same implementation but with different parameters.
        // This was initially used in PR-18471 but removed to simplify the code.
        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
        if (has_ngram_mod) {
            // shared instance for all speculative decoding contexts
            if (!params.ngram_mod) {
                params.ngram_mod = std::make_shared<common_ngram_mod>(params.ngram_size_n, 4*1024*1024);

                LOG_INF("%s: initialized ngram_mod with n=%d, size=%zu (%.3f MB)\n", __func__,
                        params.ngram_size_n, params.ngram_mod->size(),
                        (float)(params.ngram_mod->size_bytes())/1024/1024);

                if (params.ngram_size_n < 16) {
                    LOG_WRN("%s: ngram_mod n=%d is too small - poor quality is possible, see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, params.ngram_size_n);
                }
            }

            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_copyspec) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_COPYSPEC, params));
        }
        if (has_recycle) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_RECYCLE, params));
        }
        if (has_suffix) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_SUFFIX, params));
        }
        if (has_dflash) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DFLASH, params));
        }
        if (has_draft) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT, params));
        }
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_EAGLE3, params));
        }
    }

    std::vector<std::unique_ptr<common_speculative_state>> impls = {};

    for (const common_speculative_config & config : configs) {
        LOG_DBG("%s: adding implementation %s\n", __func__, common_speculative_type_to_str(config.type).c_str());
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DFLASH: {
                GGML_ASSERT(ctx_dft != nullptr);
                impls.push_back(std::make_unique<common_speculative_state_dflash>(
                    ctx_tgt,
                    ctx_dft,
                    params.model_dft
                ));
                ctx_dft = nullptr; // ownership transferred
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT: {
                impls.push_back(std::make_unique<common_speculative_state_draft>(config.type,
                    /* .ctx_tgt      = */ ctx_tgt,
                    /* .ctx_dft      = */ ctx_dft,
                    /* .replacements = */ params.replacements
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_state_eagle3>(config.type));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram      = */ ngram_size_key,
                    /* .size_mgram      = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_state_ngram_simple>(
                    /* .type            = */ config.type,
                    /* .state           = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(std::make_unique<common_speculative_state_ngram_map_k>(
                    (config.type),
                    get_common_ngram_map(config)
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                GGML_ASSERT(config.params.ngram_mod);
                impls.push_back(std::make_unique<common_speculative_state_ngram_mod>(config.type, *config.params.ngram_mod));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        params.lookup_cache_static, params.lookup_cache_dynamic, config);
                impls.push_back(std::make_unique<common_speculative_state_ngram_cache>(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_SUFFIX: {
                impls.push_back(std::make_unique<common_speculative_state_suffix>(
                    config.type,
                    config.params.suffix_max_depth,
                    config.params.n_max,
                    config.params.suffix_spec_factor,
                    config.params.suffix_spec_offset,
                    config.params.suffix_min_prob
                ));
                LOG_INF("%s: suffix tree speculative decoding (max_depth=%d, factor=%.1f, min_prob=%.2f)\n",
                    __func__, config.params.suffix_max_depth,
                    config.params.suffix_spec_factor, config.params.suffix_min_prob);
                break;
            }
            case COMMON_SPECULATIVE_TYPE_COPYSPEC: {
                impls.push_back(std::make_unique<common_speculative_state_copyspec>(
                    config.type,
                    config.params.copyspec_gamma
                ));
                LOG_INF("%s: copyspec speculative decoding (gamma=%d)\n",
                    __func__, config.params.copyspec_gamma);
                break;
            }
            case COMMON_SPECULATIVE_TYPE_RECYCLE: {
                impls.push_back(std::make_unique<common_speculative_state_recycle>(
                    config.type,
                    config.params.recycle_k
                ));
                LOG_INF("%s: token recycling speculative decoding (k=%d)\n",
                    __func__, config.params.recycle_k);
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .impls = */ std::move(impls)
    };

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(prompt);
        impl->n_call_begin++;
    }
}

llama_tokens common_speculative_draft(
        common_speculative * spec,
        const common_params_speculative & params,
        const llama_tokens & prompt_tgt, // specified in target model vocab
        llama_token id_last) {
    llama_tokens result;

    spec->curr_impl = nullptr; // reset current implementation

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(params, prompt_tgt, id_last, result);
            impl->n_call_draft++;
        }

        if (!result.empty()) {
            LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                    common_speculative_type_to_str(impl.get()->type).c_str(), prompt_tgt.size(),
                    impl.get()->n_call_draft, result.size());

            spec->curr_impl = impl.get(); // set current implementation for stats
            impl->n_gen_drafts++;
            impl->n_gen_tokens += result.size();

            break; // We have a draft, so break out of the loop and return it.
        }
    }

    return result;
}

common_speculative_tree common_speculative_draft_tree(
        common_speculative * spec,
        const common_params_speculative & params,
        const llama_tokens & prompt_tgt,
        llama_token id_last,
        int tree_budget) {
    common_speculative_tree tree;

    spec->curr_impl = nullptr;

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft_tree(params, prompt_tgt, id_last, tree_budget, tree);
            impl->n_call_draft++;
        }

        if (tree.n_nodes > 0) {
            spec->curr_impl = impl.get();
            impl->n_gen_drafts++;
            impl->n_gen_tokens += tree.n_nodes;
            break;
        }
    }

    return tree;
}

void common_speculative_accept(common_speculative * spec, uint16_t n_accepted) {
    if (n_accepted == 0) {
        return;
    }

    common_speculative_state * impl = spec->curr_impl;

    GGML_ASSERT(impl);

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(n_accepted);
        impl->n_call_accept++;
    }
}

void common_speculative_update_logits(common_speculative * spec, llama_context * ctx, const llama_tokens & batch_tokens, int n_accepted) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        impl->update_logits(ctx, batch_tokens, n_accepted);
    }
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        LOG_INF("statistics %s: #calls(b,g,a) = %zu %zu %zu, #gen drafts = %zu, #acc drafts = %zu, #gen tokens = %zu, #acc tokens = %zu%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_perf.c_str());
    }
}
