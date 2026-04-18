#include "models.h"

// DFlash drafter custom graph input
// Holds the target hidden states, context positions, and asymmetric non-causal attention mask
class llm_graph_input_dflash : public llm_graph_input_i {
public:
    llm_graph_input_dflash(const llama_cross * cross, int64_t ctx_len, int64_t n_block)
        : cross(cross), ctx_len(ctx_len), n_block(n_block) {}

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * target_hidden = nullptr; // [n_target_features, ctx_len]
    ggml_tensor * pos_ctx       = nullptr; // [ctx_len]
    ggml_tensor * kq_mask       = nullptr; // [ctx_len + n_block, n_block, 1, 1]
    ggml_tensor * kq_mask_cnv   = nullptr;

    const llama_cross * cross;
    int64_t ctx_len;
    int64_t n_block;
};

void llm_graph_input_dflash::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    // copy target hidden states from cross->v_embd to the input tensor
    // v_embd may be padded (zero-filled beyond n_enc_real)
    if (target_hidden && cross && !cross->v_embd.empty()) {
        ggml_backend_tensor_set(target_hidden, cross->v_embd.data(), 0, ggml_nbytes(target_hidden));
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
}

llm_build_dflash_draft::llm_build_dflash_draft(
        const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    const int64_t n_target_features = hparams.dflash_n_target_features;

    // n_tokens comes from ubatch — equals block_size during inference, may differ during reservation
    // ctx_len: from cross data at runtime, or cparams.n_ctx during graph reservation
    const bool have_cross = cross && !cross->v_embd.empty();
    const int64_t ctx_len = have_cross ? cross->n_enc : n_ctx;

    const int64_t n_kv_total = ctx_len + n_tokens;

    // --- DFlash-specific inputs ---
    auto inp_dflash = std::make_unique<llm_graph_input_dflash>(cross, ctx_len, n_tokens);

    // concatenated target hidden states [n_target_features, ctx_len]
    inp_dflash->target_hidden = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_target_features, ctx_len);
    ggml_set_input(inp_dflash->target_hidden);
    cb(inp_dflash->target_hidden, "dflash_target_hidden", -1);

    // context positions for K RoPE [ctx_len]
    inp_dflash->pos_ctx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ctx_len);
    ggml_set_input(inp_dflash->pos_ctx);
    cb(inp_dflash->pos_ctx, "dflash_pos_ctx", -1);

    // asymmetric non-causal mask [n_kv_total, n_tokens, 1, 1]
    inp_dflash->kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv_total, n_tokens, 1, 1);
    ggml_set_input(inp_dflash->kq_mask);
    inp_dflash->kq_mask_cnv = cparams.flash_attn
        ? ggml_cast(ctx0, inp_dflash->kq_mask, GGML_TYPE_F16)
        : inp_dflash->kq_mask;

    ggml_tensor * kq_mask       = inp_dflash->kq_mask_cnv;
    ggml_tensor * pos_ctx       = inp_dflash->pos_ctx;
    ggml_tensor * target_hidden = inp_dflash->target_hidden;

    res->add_input(std::move(inp_dflash));

    // --- Embedding ---
    // tok_embd/output may be nullptr during graph reservation (shared from target at runtime)
    // use F16 placeholder to reduce compute buffer (actual tensors are quantized, even smaller)
    ggml_tensor * tok_embd_use = model.tok_embd;
    if (!tok_embd_use) {
        tok_embd_use = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, model.vocab.n_tokens());
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
    ggml_tensor * output_use = model.output;
    if (!output_use) {
        output_use = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, model.vocab.n_tokens());
    }
    cur = build_lora_mm(output_use, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
