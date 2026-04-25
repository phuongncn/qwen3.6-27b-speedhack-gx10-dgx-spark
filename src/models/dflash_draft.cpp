#include "models.h"

#include <algorithm>
#include <atomic>

// DFlash drafter custom graph input
// Holds the target hidden states, context positions, and asymmetric non-causal attention mask
class llm_graph_input_dflash : public llm_graph_input_i {
public:
    llm_graph_input_dflash(const llama_cross * cross, int64_t ctx_len, int64_t n_block, uint32_t n_swa)
        : cross(cross), ctx_len(ctx_len), n_block(n_block), n_swa(n_swa) {}

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * target_hidden     = nullptr; // [n_target_features, ctx_len]
    ggml_tensor * pos_ctx           = nullptr; // [ctx_len]
    ggml_tensor * kq_mask           = nullptr; // [ctx_len + n_block, n_block, 1, 1]
    ggml_tensor * kq_mask_cnv       = nullptr;
    // Only allocated when hparams.is_swa_any(); same shape as kq_mask
    ggml_tensor * kq_mask_swa       = nullptr;
    ggml_tensor * kq_mask_swa_cnv   = nullptr;

    const llama_cross * cross;
    int64_t ctx_len;
    int64_t n_block;
    uint32_t n_swa;
};

void llm_graph_input_dflash::set_input(const llama_ubatch * ubatch) {
    // copy target hidden states from cross->v_embd to the input tensor.
    // v_embd may be narrower than target_hidden when fewer than MAX_SLOTS are active
    // (single-slot path: v_embd sized for PER_SLOT_CTX, target_hidden sized for
    // MAX_SLOTS * PER_SLOT_CTX). Copy what we have, zero-fill the rest — padding
    // regions are masked to -INF in kq_mask so zero content is safe.
    if (target_hidden && cross && !cross->v_embd.empty()) {
        const size_t vembd_bytes  = cross->v_embd.size() * sizeof(float);
        const size_t tensor_bytes = ggml_nbytes(target_hidden);
        const size_t copy_bytes   = std::min(vembd_bytes, tensor_bytes);
        ggml_backend_tensor_set(target_hidden, cross->v_embd.data(), 0, copy_bytes);
        if (copy_bytes < tensor_bytes) {
            ggml_backend_tensor_memset(target_hidden, 0, copy_bytes, tensor_bytes - copy_bytes);
        }
    }

    const int64_t n_real = cross ? cross->n_enc_real : ctx_len;

    // context positions: [0, 1, ..., n_real-1, 0, 0, ..., 0] (padding gets position 0)
    if (pos_ctx && pos_ctx->buffer) {
        GGML_ASSERT(ggml_backend_buffer_is_host(pos_ctx->buffer));
        int32_t * data = (int32_t *) pos_ctx->data;
        for (int64_t i = 0; i < ctx_len; ++i) {
            data[i] = (i < n_real) ? (int32_t) i : 0;
        }
    }

    // attention mask: real positions visible (0.0f), padding masked (-inf)
    // mask shape: [ctx_len + n_block, n_block, 1, 1]
    // K ordering: [ctx_0, ..., ctx_{n-1}, ctx_pad, ..., block_0, ..., block_{b-1}]
    if (kq_mask && kq_mask->buffer) {
        GGML_ASSERT(ggml_backend_buffer_is_host(kq_mask->buffer));
        float * data = (float *) kq_mask->data;
        const int64_t n_kv = ctx_len + n_block;
        for (int64_t q = 0; q < n_block; ++q) {
            for (int64_t k = 0; k < n_kv; ++k) {
                // mask out padded context positions (between n_real and ctx_len)
                if (k >= n_real && k < ctx_len) {
                    data[q * n_kv + k] = -INFINITY;
                } else {
                    data[q * n_kv + k] = 0.0f;
                }
            }
        }
    }

    // Causal SWA: block query q attends to block keys <= q and to real ctx keys
    // in window [q_pos - n_swa, q_pos]; padded ctx slots are masked.
    // q_pos comes from ubatch->pos when available, else inferred as n_real + q.
    if (kq_mask_swa && kq_mask_swa->buffer && n_swa > 0) {
        GGML_ASSERT(ggml_backend_buffer_is_host(kq_mask_swa->buffer));
        float * data = (float *) kq_mask_swa->data;
        const int64_t n_kv   = ctx_len + n_block;
        const int32_t window = (int32_t) n_swa;
        const bool    have_pos = (ubatch != nullptr) && (ubatch->pos != nullptr)
                               && ((int64_t) ubatch->n_tokens >= n_block);
        for (int64_t q = 0; q < n_block; ++q) {
            const int32_t q_pos = have_pos ? ubatch->pos[q] : (int32_t) (n_real + q);
            for (int64_t k = 0; k < n_kv; ++k) {
                float v = 0.0f;
                if (k < n_real) {
                    if (q_pos - (int32_t) k > window) v = -INFINITY;
                } else if (k < ctx_len) {
                    v = -INFINITY;
                } else {
                    const int64_t b_k = k - ctx_len;
                    if (b_k > q) v = -INFINITY;
                }
                data[q * n_kv + k] = v;
            }
        }
    }
}

llm_build_dflash_draft::llm_build_dflash_draft(
        const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    const int64_t n_target_features = hparams.dflash_n_target_features;

    // Drafter graph shape:
    //   n_slots == 1: ctx_len = cross->n_enc (power-of-2 bucket of actual data length).
    //                 set_cross_data triggers sched_need_reserve when the bucket changes,
    //                 so the graph re-reserves at each bucket boundary. This matches the
    //                 pre-B2.0 path and keeps single-slot at the original throughput.
    //   n_slots >= 2: ctx_len = n_slots × PER_SLOT_CTX (fixed). The shared drafter ctx
    //                 services multiple slots whose bucket-of-n_enc would otherwise
    //                 thrash sched_need_reserve as different slots write data of
    //                 different lengths. Fixed width avoids that thrash; multi-slot
    //                 users pay flat n_slots × PER_SLOT_CTX attention cost.
    const int n_slots = std::clamp((int) cparams.dflash_n_slots, 1, (int) LLAMA_DFLASH_MAX_SLOTS);
    int64_t ctx_len;
    if (n_slots == 1) {
        ctx_len = (cross && cross->n_enc > 0) ? cross->n_enc : (int64_t) LLAMA_DFLASH_PER_SLOT_CTX;
    } else {
        ctx_len = (int64_t) n_slots * LLAMA_DFLASH_PER_SLOT_CTX;
    }

    const int64_t n_kv_total = ctx_len + n_tokens;

    // --- DFlash-specific inputs ---
    const bool have_swa = hparams.is_swa_any();
    auto inp_dflash = std::make_unique<llm_graph_input_dflash>(cross, ctx_len, n_tokens, hparams.n_swa);

    // concatenated target hidden states [n_target_features, ctx_len]
    inp_dflash->target_hidden = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_target_features, ctx_len);
    ggml_set_input(inp_dflash->target_hidden);
    cb(inp_dflash->target_hidden, "dflash_target_hidden", -1);

    // context positions for K RoPE [ctx_len]
    inp_dflash->pos_ctx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ctx_len);
    ggml_set_input(inp_dflash->pos_ctx);
    cb(inp_dflash->pos_ctx, "dflash_pos_ctx", -1);

    // asymmetric non-causal mask [n_kv_total, n_tokens, 1, 1] — full-attention layers
    inp_dflash->kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv_total, n_tokens, 1, 1);
    ggml_set_input(inp_dflash->kq_mask);
    inp_dflash->kq_mask_cnv = cparams.flash_attn
        ? ggml_cast(ctx0, inp_dflash->kq_mask, GGML_TYPE_F16)
        : inp_dflash->kq_mask;

    if (have_swa) {
        inp_dflash->kq_mask_swa = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv_total, n_tokens, 1, 1);
        ggml_set_input(inp_dflash->kq_mask_swa);
        cb(inp_dflash->kq_mask_swa, "dflash_kq_mask_swa", -1);
        inp_dflash->kq_mask_swa_cnv = cparams.flash_attn
            ? ggml_cast(ctx0, inp_dflash->kq_mask_swa, GGML_TYPE_F16)
            : inp_dflash->kq_mask_swa;
    }

    ggml_tensor * kq_mask_full  = inp_dflash->kq_mask_cnv;
    ggml_tensor * kq_mask_swa   = inp_dflash->kq_mask_swa_cnv; // may be null if no SWA
    ggml_tensor * pos_ctx       = inp_dflash->pos_ctx;
    ggml_tensor * target_hidden = inp_dflash->target_hidden;

    res->add_input(std::move(inp_dflash));

    // --- Embedding ---
    // tok_embd/output may be nullptr during graph reservation (shared from target at runtime)
    // Use Q4_0 placeholder to avoid 4.8 GB F32 allocation during reservation
    ggml_tensor * tok_embd_use = model.tok_embd;
    if (!tok_embd_use) {
        tok_embd_use = ggml_new_tensor_2d(ctx0, GGML_TYPE_Q4_0, n_embd, model.vocab.n_tokens());
    }
    ggml_tensor * inpL = build_inp_embd(tok_embd_use);

    // block positions from ubatch.pos
    ggml_tensor * inp_pos = build_inp_pos();

    // --- Fusion layer: project concatenated target hidden states ---
    ggml_tensor * fused_target = build_lora_mm(model.dflash_fc, target_hidden);
    fused_target = build_norm(fused_target, model.dflash_hidden_norm, nullptr, LLM_NORM_RMS, -1);
    cb(fused_target, "fused_target", -1);

    // --- Transformer layers ---
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        ggml_tensor * kq_mask = (hparams.is_swa(il) && kq_mask_swa) ? kq_mask_swa : kq_mask_full;

        ggml_tensor * cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // --- KV-injection attention ---
        {
            // Q from drafter hidden only
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                                 n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur", il);

            // K from drafter (noise tokens)
            ggml_tensor * Kcur_noise = build_lora_mm(model.layers[il].wk, cur);
            Kcur_noise = ggml_reshape_3d(ctx0, Kcur_noise, n_embd_head, n_head_kv, n_tokens);
            Kcur_noise = build_norm(Kcur_noise, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            Kcur_noise = ggml_rope_ext(ctx0, Kcur_noise, inp_pos, nullptr,
                                       n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                       ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur_noise, "Kcur_noise", il);

            // K from target (context features)
            ggml_tensor * Kcur_ctx = build_lora_mm(model.layers[il].wk, fused_target);
            Kcur_ctx = ggml_reshape_3d(ctx0, Kcur_ctx, n_embd_head, n_head_kv, ctx_len);
            Kcur_ctx = build_norm(Kcur_ctx, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            Kcur_ctx = ggml_rope_ext(ctx0, Kcur_ctx, pos_ctx, nullptr,
                                     n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur_ctx, "Kcur_ctx", il);

            // V from drafter (noise tokens)
            ggml_tensor * Vcur_noise = build_lora_mm(model.layers[il].wv, cur);
            Vcur_noise = ggml_reshape_3d(ctx0, Vcur_noise, n_embd_head, n_head_kv, n_tokens);
            cb(Vcur_noise, "Vcur_noise", il);

            // V from target (context features)
            ggml_tensor * Vcur_ctx = build_lora_mm(model.layers[il].wv, fused_target);
            Vcur_ctx = ggml_reshape_3d(ctx0, Vcur_ctx, n_embd_head, n_head_kv, ctx_len);
            cb(Vcur_ctx, "Vcur_ctx", il);

            // concatenate K: [ctx, noise] along sequence dim (dim 2)
            ggml_tensor * Kcur = ggml_concat(ctx0, Kcur_ctx, Kcur_noise, 2);
            cb(Kcur, "Kcur", il);

            // concatenate V: [ctx, noise] along sequence dim (dim 2)
            ggml_tensor * Vcur = ggml_concat(ctx0, Vcur_ctx, Vcur_noise, 2);
            cb(Vcur, "Vcur", il);

            // prevent reordering
            ggml_build_forward_expand(gf, Qcur);
            ggml_build_forward_expand(gf, Kcur);
            ggml_build_forward_expand(gf, Vcur);

            // asymmetric attention: Q [head_dim, n_head, n_tokens]
            //                       K [head_dim, n_head_kv, n_kv_total]
            //                    mask [n_kv_total, n_tokens, 1, 1]
            cur = build_attn_mha(Qcur, Kcur, Vcur, nullptr, kq_mask, nullptr, nullptr,
                                 1.0f / sqrtf(float(n_embd_head)), il);
            cb(cur, "kqv_out", il);

            // output projection
            cur = build_lora_mm(model.layers[il].wo, cur);
        }

        // residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        ggml_tensor * ffn_residual = cur;

        // post-attention RMSNorm
        cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        // SwiGLU FFN
        cur = build_ffn(cur,
            model.layers[il].ffn_up,   nullptr, nullptr,
            model.layers[il].ffn_gate, nullptr, nullptr,
            model.layers[il].ffn_down, nullptr, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        // FFN residual
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    // final RMSNorm
    ggml_tensor * cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head — may be nullptr during reservation (shared from target at runtime)
    // Use Q4_0 placeholder to avoid 4.8 GB F32 allocation during reservation
    ggml_tensor * output_use = model.output;
    if (!output_use) {
        output_use = ggml_new_tensor_2d(ctx0, GGML_TYPE_Q4_0, n_embd, model.vocab.n_tokens());
    }
    cur = build_lora_mm(output_use, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    // GPU top-K or argmax — avoids 15.9MB logits transfer + CPU scan for DFlash draft
    const float sample_temp = cparams.dflash_sample_temp;
    static std::atomic<uint64_t> gumbel_counter{1};
    const uint64_t seed = (sample_temp > 0.0f) ? gumbel_counter.fetch_add(1) : 0;
    const int topk = cparams.dflash_topk;
    if (topk > 1) {
        res->t_logits_argmax = ggml_topk_ext(ctx0, cur, topk, sample_temp, seed);
    } else {
        res->t_logits_argmax = ggml_argmax_ext(ctx0, cur, sample_temp, seed);
    }

    ggml_build_forward_expand(gf, res->t_logits_argmax);
}
