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
            params.speculative.type != COMMON_SPECULATIVE_TYPE_SUFFIX &&
            params.speculative.type != COMMON_SPECULATIVE_TYPE_COPYSPEC &&
            params.speculative.type != COMMON_SPECULATIVE_TYPE_RECYCLE) {
        LOG_ERR("%s: --model-draft is required (unless using a model-free --spec-type)\n", __func__);
        return 1;
    }

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;

    llama_context * ctx_tgt = NULL;

    // speculative decoding with recurrent/hybrid models needs seq_backup=1 for state rollback
    if (params.n_parallel < 2) {
        params.n_parallel = 2;
    }

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
        // drafter only processes block_size tokens per call — keep batch small to save VRAM
        params_dft.n_batch      = std::min((int32_t)64, params_spec.n_ctx > 0 ? params_spec.n_ctx : (int32_t)llama_n_ctx_seq(ctx_tgt));
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

        // DFlash: share tok_embd/output from target BEFORE creating drafter context
        // This avoids allocating huge placeholder tensors during graph reservation
        if (params.speculative.type == COMMON_SPECULATIVE_TYPE_DFLASH) {
            llama_model_share_tensors(model_dft.get(), model_tgt);
        }
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

    // per-position rejection histogram: reject_pos[i] = how many times position i caused rejection
    // position 0 = first draft token, position N-1 = last. "all_accepted" counted separately.
    std::vector<int> reject_pos(16, 0);
    int n_all_accepted = 0;

    // used to determine end of generation
    bool has_eos = false;

    // hybrid models with recurrent layers need state management after partial rejection
    const bool needs_reeval = llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt);
    // GPU tape replay: runs GDN ops on GPU with state views (zero-copy), uploads only tape data (~24MB).
    // Bit-identical to forward pass. Falls back to CPU replay if no GPU backend.
    const bool use_tape_replay = true;
    // backup sequence ID for saving recurrent state before speculative batches
    const llama_seq_id seq_backup = 1;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    // init the speculator BEFORE prefill so DFlash can configure hidden state capture
    const auto & params_spec = params.speculative;

    struct common_speculative * spec = common_speculative_init(params.speculative, ctx_tgt);

    // eval the prompt (with hidden state capture if DFlash is active)
    llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size() - 1));

    // note: keep the last token separate!
    llama_token id_last = inp.back();

    // all tokens currently in the target context
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = inp.size() - 1;

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

    const int tree_budget = params_spec.tree_budget;

    while (true) {
        n_iters++;

        llama_tokens ids;
        int n_draft_this_iter = 0;
        bool has_backup = false;
        bool accepted_on_main_path = true;

        if (tree_budget > 0) {
            // === DDTree path: tree-structured speculative decoding ===
            common_speculative_tree tree;
            {
                common_time_meas tm(t_draft_total);
                tree = common_speculative_draft_tree(spec, params_spec, prompt_tgt, id_last, tree_budget);
            }

            common_batch_clear(batch_tgt);

            if (tree.n_nodes == 0) {
                // no tree nodes — single token decode (same as empty draft)
                common_batch_add(batch_tgt, id_last, n_past++, {0}, true);
                {
                    common_time_meas tm(t_decode1_total);
                    llama_decode(ctx_tgt, batch_tgt);
                }
                // update DFlash drafter hidden states from this single-token decode
                {
                    llama_tokens batch_tokens = { id_last };
                    common_speculative_update_logits(spec, ctx_tgt, batch_tokens, 1);
                }
                {
                    common_time_meas tm(t_sample_total);
                    llama_token t = common_sampler_sample(smpl, ctx_tgt, 0);
                    common_sampler_accept(smpl, t, true);
                    ids.push_back(t);
                }
            } else {
                // build tree verification batch: root + tree nodes
                common_batch_add(batch_tgt, id_last, n_past, {0}, true);
                for (int i = 0; i < tree.n_nodes; ++i) {
                    common_batch_add(batch_tgt, tree.tokens[i], n_past + tree.depths[i], {0}, true);
                }
                n_past++; // root consumed

                n_draft_this_iter = tree.n_nodes;

                // backup state before tree decode
                if (needs_reeval) {
                    common_time_meas tm(t_backup_total);
                    auto * mem = llama_get_memory(ctx_tgt);
                    llama_memory_seq_rm(mem, seq_backup, -1, -1);
                    llama_memory_seq_cp(mem, 0, seq_backup, -1, -1);
                    has_backup = true;
                }

                // main path is first in batch → DeltaNet processes it before branches
                // tape recording captures clean main-path state for fast rollback
                llama_set_tree_mask(ctx_tgt, tree.visibility.data(), tree.n_nodes + 1);
                if (use_tape_replay && tree.n_nodes > 0) {
                    llama_set_tape_recording(ctx_tgt, true);
                }
                {
                    common_time_meas tm(t_decode1_total);
                    llama_decode(ctx_tgt, batch_tgt);
                }
                if (use_tape_replay && tree.n_nodes > 0) {
                    llama_set_tape_recording(ctx_tgt, false);
                }
                llama_clear_tree_mask(ctx_tgt);

                // tree walk: follow target's greedy choices through child_maps
                // track whether accepted path stays on main path (nodes 1..main_path_len)
                {
                    common_time_meas tm(t_sample_total);
                    int current = 0; // root (batch index 0)
                    while (true) {
                        llama_token target_token = common_sampler_sample(smpl, ctx_tgt, current);
                        common_sampler_accept(smpl, target_token, true);
                        ids.push_back(target_token);

                        auto it = tree.child_maps[current].find(target_token);
                        if (it != tree.child_maps[current].end()) {
                            int next = it->second;
                            if (next > tree.main_path_len) {
                                accepted_on_main_path = false;
                            }
                            current = next;
                        } else {
                            break;
                        }
                    }
                }

                // update drafter hidden states from verification decode
                // main path is first in batch, so hidden states at 0..n_accepted-1 are clean
                if (accepted_on_main_path) {
                    llama_tokens batch_tokens;
                    batch_tokens.push_back(id_last);
                    batch_tokens.insert(batch_tokens.end(), tree.tokens.begin(), tree.tokens.end());
                    common_speculative_update_logits(spec, ctx_tgt, batch_tokens, (int)ids.size());
                }
            }
        } else {
            // === Flat path: linear speculative decoding ===
            llama_tokens draft;
            {
                common_time_meas tm(t_draft_total);
                draft = common_speculative_draft(spec, params_spec, prompt_tgt, id_last);
            }

            common_batch_clear(batch_tgt);
            common_batch_add  (batch_tgt, id_last, n_past++, { 0 }, true);

            {
                if (draft.size() < (size_t) params_spec.n_min) {
                    draft.clear();
                }

                if (needs_reeval && !draft.empty()) {
                    common_time_meas tm(t_backup_total);
                    auto * mem = llama_get_memory(ctx_tgt);
                    llama_memory_seq_rm(mem, seq_backup, -1, -1);
                    llama_memory_seq_cp(mem, 0, seq_backup, -1, -1);
                    has_backup = true;
                }

                for (size_t i = 0; i < draft.size(); ++i) {
                    common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
                }

                if (use_tape_replay && !draft.empty()) {
                    llama_set_tape_recording(ctx_tgt, true);
                }

                {
                    common_time_meas tm(t_decode1_total);
                    llama_decode(ctx_tgt, batch_tgt);
                }

                if (use_tape_replay && !draft.empty()) {
                    llama_set_tape_recording(ctx_tgt, false);
                }
            }

            {
                common_time_meas tm(t_sample_total);
                ids = common_sampler_sample_and_accept_n(smpl, ctx_tgt, draft);
            }

            n_draft_this_iter = (int)draft.size();

            // update draft strategies with logits (e.g. token recycling adjacency matrix)
            {
                llama_tokens batch_tokens;
                batch_tokens.push_back(id_last);
                batch_tokens.insert(batch_tokens.end(), draft.begin(), draft.end());
                common_speculative_update_logits(spec, ctx_tgt, batch_tokens, (int)ids.size());
            }
        }

        GGML_ASSERT(ids.size() > 0);

        n_past    += ids.size() - 1;
        n_drafted += n_draft_this_iter;
        n_accept  += ids.size() - 1;
        n_predict += ids.size();

        // track rejection position
        if (n_draft_this_iter > 0) {
            if ((int)ids.size() == n_draft_this_iter + 1) {
                n_all_accepted++;
            } else {
                int rej_pos = (int)ids.size() - 1;
                if (rej_pos < (int)reject_pos.size()) {
                    reject_pos[rej_pos]++;
                }
            }
        }

        {
            common_time_meas tm(t_other_total);
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
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, n_draft_this_iter, id_last);

        if (has_backup && !has_eos) {
            if (tree_budget > 0) {
                // DDTree rollback
                const bool all_accepted = ((int)ids.size() == n_draft_this_iter + 1);
                const int n_past_before = n_past - (int)ids.size();

                if (all_accepted) {
                    // all drafted tokens accepted — just clean up backup and trim KV
                    auto * mem = llama_get_memory(ctx_tgt);
                    llama_memory_seq_rm(mem, seq_backup, -1, -1);
                    llama_memory_seq_rm(mem, 0, n_past, -1);
                    n_reeval_skipped++;
                } else if (accepted_on_main_path && use_tape_replay) {
                    // accepted path is on main path — tape replay for DeltaNet, KV trim for attention
                    // branch KV entries at accepted depths stay (minor pollution, 16/64 attn layers)
                    {
                        common_time_meas tm(t_decode2_total);
                        llama_dflash_rollback(ctx_tgt, seq_backup, n_past_before, (int)ids.size());
                        n_reeval_tokens += (int)ids.size();
                        n_reeval_calls++;
                    }
                    n_past = (int)prompt_tgt.size();
                } else {
                    // branch acceptance or no tape — full re-eval
                    {
                        common_time_meas tm(t_restore_total);
                        auto * mem = llama_get_memory(ctx_tgt);
                        llama_memory_seq_rm(mem, 0, n_past_before, -1);
                        llama_memory_seq_cp(mem, seq_backup, 0, -1, -1);
                        llama_memory_seq_rm(mem, seq_backup, -1, -1);
                    }
                    common_batch_clear(batch_tgt);
                    for (int i = n_past_before; i < (int)prompt_tgt.size(); ++i) {
                        common_batch_add(batch_tgt, prompt_tgt[i], i, { 0 }, false);
                    }
                    if (batch_tgt.n_tokens > 0) {
                        common_time_meas tm(t_decode2_total);
                        llama_decode(ctx_tgt, batch_tgt);
                        n_reeval_tokens += batch_tgt.n_tokens;
                        n_reeval_calls++;
                    }
                    // update DFlash drafter hidden states from re-eval
                    {
                        llama_tokens reeval_tokens(prompt_tgt.begin() + n_past_before, prompt_tgt.end());
                        common_speculative_update_logits(spec, ctx_tgt, reeval_tokens, (int)reeval_tokens.size());
                    }
                    n_past = (int)prompt_tgt.size();
                }
            } else {
                // Flat path rollback
                const bool all_accepted = ((int)ids.size() == n_draft_this_iter + 1);

                if (all_accepted) {
                    auto * mem = llama_get_memory(ctx_tgt);
                    llama_memory_seq_rm(mem, seq_backup, -1, -1);
                    llama_memory_seq_rm(mem, 0, n_past, -1);
                    n_reeval_skipped++;
                } else {
                    auto * mem = llama_get_memory(ctx_tgt);
                    const int n_past_before = n_past - (int)ids.size();

                    if (use_tape_replay) {
                        common_time_meas tm(t_decode2_total);
                        llama_dflash_rollback(ctx_tgt, seq_backup, n_past_before, (int)ids.size());
                        n_reeval_tokens += (int)ids.size();
                        n_reeval_calls++;
                    } else {
                        {
                            common_time_meas tm(t_restore_total);
                            llama_memory_seq_rm(mem, 0, n_past_before, -1);
                            llama_memory_seq_cp(mem, seq_backup, 0, -1, -1);
                            llama_memory_seq_rm(mem, seq_backup, -1, -1);
                        }
                        common_batch_clear(batch_tgt);
                        for (int i = n_past_before; i < (int)prompt_tgt.size(); ++i) {
                            common_batch_add(batch_tgt, prompt_tgt[i], i, { 0 }, false);
                        }

                        if (batch_tgt.n_tokens > 0) {
                            common_time_meas tm(t_decode2_total);
                            llama_decode(ctx_tgt, batch_tgt);
                            n_reeval_tokens += batch_tgt.n_tokens;
                            n_reeval_calls++;
                        }
                    }
                    n_past = (int)prompt_tgt.size();
                }
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

    // per-position rejection histogram
    {
        int n_rounds_with_draft = n_all_accepted;
        for (int i = 0; i < (int)reject_pos.size(); ++i) {
            n_rounds_with_draft += reject_pos[i];
        }
        if (n_rounds_with_draft > 0) {
            LOG_INF("\nrejection histogram (position → count [%%]):\n");
            for (int i = 0; i < (int)reject_pos.size() && i < params_spec.n_max; ++i) {
                if (reject_pos[i] > 0) {
                    LOG_INF("  pos %2d: %4d (%5.1f%%)\n", i, reject_pos[i], 100.0f * reject_pos[i] / n_rounds_with_draft);
                }
            }
            LOG_INF("  all ok: %4d (%5.1f%%)\n", n_all_accepted, 100.0f * n_all_accepted / n_rounds_with_draft);
        }
    }

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
