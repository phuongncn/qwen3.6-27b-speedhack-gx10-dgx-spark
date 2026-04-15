#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    if (params.speculative.mparams_dft.path.empty() &&
            params.speculative.type != COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE &&
            params.speculative.type != COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K &&
            params.speculative.type != COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V &&
            params.speculative.type != COMMON_SPECULATIVE_TYPE_NGRAM_MOD &&
            params.speculative.type != COMMON_SPECULATIVE_TYPE_NGRAM_CACHE &&
            params.speculative.type != COMMON_SPECULATIVE_TYPE_SUFFIX) {
        LOG_ERR("%s: --model-draft is required (unless using a model-free --spec-type)\n", __func__);
        return 1;
    }

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;

    llama_context * ctx_tgt = NULL;

    // load the target model
    auto llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt->model();
    ctx_tgt   = llama_init_tgt->context();

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // load the draft model (skip for model-free spec types)
    llama_model_ptr model_dft;

    if (!params.speculative.mparams_dft.path.empty()) {
        const auto & params_spec = params.speculative;

        auto params_dft = params;

        params_dft.n_parallel   = 1;
        params_dft.n_ctx        = params_spec.n_ctx;
        params_dft.n_batch      = llama_n_ctx_seq(ctx_tgt);
        params_dft.devices      = params_spec.devices;
        params_dft.model        = params_spec.mparams_dft;
        params_dft.n_gpu_layers = params_spec.n_gpu_layers;

        if (params_spec.cpuparams.n_threads > 0) {
            params_dft.cpuparams.n_threads       = params.speculative.cpuparams.n_threads;
            params_dft.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;
        }

        params_dft.tensor_buft_overrides = params.speculative.tensor_buft_overrides;

        auto mparams_dft = common_model_params_to_llama(params_dft);

        model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
        if (model_dft == nullptr) {
            LOG_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
            return 1;
        }

        params.speculative.model_dft = model_dft.get();
        params.speculative.cparams_dft = common_context_params_to_llama(params_dft);
    }

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool has_eos = false;

    // hybrid models with recurrent layers need re-evaluation of accepted tokens
    // after rejecting draft tokens, because the recurrent state cannot be rolled back
    const bool needs_reeval = llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt);
    // backup sequence ID for saving recurrent state before speculative batches
    const llama_seq_id seq_backup = 1;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    // eval the prompt
    llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size() - 1));

    // note: keep the last token separate!
    llama_token id_last = inp.back();

    // all tokens currently in the target context
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = inp.size() - 1;

    // init the speculator
    const auto & params_spec = params.speculative;

    struct common_speculative * spec = common_speculative_init(params.speculative, ctx_tgt);

    common_speculative_begin(spec, prompt_tgt);

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    const auto t_enc_end = ggml_time_us();

    const auto t_dec_start = ggml_time_us();

    // timing accumulators (microseconds)
    int64_t t_draft_total   = 0;  // suffix tree draft generation
    int64_t t_backup_total  = 0;  // copy_cell backup
    int64_t t_decode1_total = 0;  // main verification decode
    int64_t t_sample_total  = 0;  // sampling
    int64_t t_restore_total = 0;  // seq_rm + seq_cp restore
    int64_t t_decode2_total = 0;  // re-evaluation decode
    int64_t t_other_total   = 0;  // everything else (token output, bookkeeping)
    int     n_iters         = 0;
    int     n_reeval_tokens = 0;  // total tokens re-evaluated
    int     n_reeval_calls  = 0;  // number of re-eval decode calls
    int     n_reeval_skipped = 0; // iterations where all drafts accepted (no re-eval needed)

    while (true) {
        int64_t t0, t1;
        n_iters++;

        t0 = ggml_time_us();
        llama_tokens draft = common_speculative_draft(spec, params_spec, prompt_tgt, id_last);
        t1 = ggml_time_us();
        t_draft_total += (t1 - t0);

        // always have a token to evaluate from before - id_last
        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, id_last, n_past++, { 0 }, true);

        bool has_backup = false;

        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        {
            // do not waste time on small drafts
            if (draft.size() < (size_t) params_spec.n_min) {
                draft.clear();
            }

            // For hybrid models: save recurrent state before adding draft tokens
            // so we can restore it after rejection
            if (needs_reeval && !draft.empty()) {
                t0 = ggml_time_us();
                auto * mem = llama_get_memory(ctx_tgt);
                llama_memory_seq_rm(mem, seq_backup, -1, -1);
                llama_memory_seq_cp(mem, 0, seq_backup, -1, -1);
                has_backup = true;
                t1 = ggml_time_us();
                t_backup_total += (t1 - t0);
            }

            for (size_t i = 0; i < draft.size(); ++i) {
                common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
            }

            t0 = ggml_time_us();
            llama_decode(ctx_tgt, batch_tgt);
            t1 = ggml_time_us();
            t_decode1_total += (t1 - t0);
        }

        t0 = ggml_time_us();
        const auto ids = common_sampler_sample_and_accept_n(smpl, ctx_tgt, draft);
        t1 = ggml_time_us();
        t_sample_total += (t1 - t0);

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token

        n_past    += ids.size() - 1;
        n_drafted += draft.size(); // note: we ignore the discarded small drafts
        n_accept  += ids.size() - 1;
        n_predict += ids.size();

        t0 = ggml_time_us();
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);

            id_last = ids[i];

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);

            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
        }
        t1 = ggml_time_us();
        t_other_total += (t1 - t0);

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, (int) draft.size(), id_last);

        if (has_backup && !has_eos) {
            const bool all_accepted = (ids.size() == draft.size() + 1);

            if (all_accepted) {
                // All draft tokens accepted — recurrent state is already correct
                // from the verification decode. Skip expensive restore+re-decode.
                auto * mem = llama_get_memory(ctx_tgt);
                llama_memory_seq_rm(mem, seq_backup, -1, -1);
                llama_memory_seq_rm(mem, 0, n_past, -1);
                n_reeval_skipped++;
            } else {
                // Partial rejection — must restore backup and re-decode accepted tokens.
                t0 = ggml_time_us();
                auto * mem = llama_get_memory(ctx_tgt);
                const int n_past_before = n_past - (int)ids.size();

                llama_memory_seq_rm(mem, 0, n_past_before, -1);
                llama_memory_seq_cp(mem, seq_backup, 0, -1, -1);
                llama_memory_seq_rm(mem, seq_backup, -1, -1);
                t1 = ggml_time_us();
                t_restore_total += (t1 - t0);

                common_batch_clear(batch_tgt);
                for (int i = n_past_before; i < (int)prompt_tgt.size(); ++i) {
                    common_batch_add(batch_tgt, prompt_tgt[i], i, { 0 }, false);
                }

                if (batch_tgt.n_tokens > 0) {
                    t0 = ggml_time_us();
                    llama_decode(ctx_tgt, batch_tgt);
                    t1 = ggml_time_us();
                    t_decode2_total += (t1 - t0);
                    n_reeval_tokens += batch_tgt.n_tokens;
                    n_reeval_calls++;
                }
                n_past = (int)prompt_tgt.size();
            }
        } else {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past, -1);
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    const int64_t t_dec_total = t_dec_end - t_dec_start;
    const int64_t t_accounted = t_draft_total + t_backup_total + t_decode1_total + t_sample_total + t_restore_total + t_decode2_total + t_other_total;

    LOG_INF("\n");
    LOG_INF("=== SPECULATIVE LOOP TIMING BREAKDOWN ===\n");
    LOG_INF("iterations:     %d\n", n_iters);
    LOG_INF("total decode:   %8.2f ms (100%%)\n", t_dec_total / 1e3);
    LOG_INF("  draft gen:    %8.2f ms (%5.1f%%)\n", t_draft_total / 1e3,   100.0 * t_draft_total / t_dec_total);
    LOG_INF("  backup:       %8.2f ms (%5.1f%%)\n", t_backup_total / 1e3,  100.0 * t_backup_total / t_dec_total);
    LOG_INF("  decode1 (verify): %8.2f ms (%5.1f%%)  [%d tok/call avg]\n", t_decode1_total / 1e3, 100.0 * t_decode1_total / t_dec_total, n_iters > 0 ? (int)(n_predict + n_drafted) / n_iters : 0);
    LOG_INF("  sampling:     %8.2f ms (%5.1f%%)\n", t_sample_total / 1e3,  100.0 * t_sample_total / t_dec_total);
    LOG_INF("  restore:      %8.2f ms (%5.1f%%)\n", t_restore_total / 1e3, 100.0 * t_restore_total / t_dec_total);
    LOG_INF("  decode2 (reeval): %8.2f ms (%5.1f%%)  [%d calls, %d tok total, %.1f tok/call avg, %d skipped (all-accept)]\n", t_decode2_total / 1e3, 100.0 * t_decode2_total / t_dec_total, n_reeval_calls, n_reeval_tokens, n_reeval_calls > 0 ? (float)n_reeval_tokens / n_reeval_calls : 0.0f, n_reeval_skipped);
    LOG_INF("  other:        %8.2f ms (%5.1f%%)\n", t_other_total / 1e3,   100.0 * t_other_total / t_dec_total);
    LOG_INF("  unaccounted:  %8.2f ms (%5.1f%%)\n", (t_dec_total - t_accounted) / 1e3, 100.0 * (t_dec_total - t_accounted) / t_dec_total);
    LOG_INF("=========================================\n");

    const int n_input = inp.size();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", params_spec.n_max);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_INF("\n");
    LOG_INF("draft:\n\n");

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl);

    llama_batch_free(batch_tgt);

    common_sampler_free(smpl);
    common_speculative_free(spec);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
