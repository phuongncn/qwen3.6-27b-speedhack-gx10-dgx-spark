#pragma once

#include "llama.h"
#include "common.h"

#include <unordered_map>
#include <vector>

struct common_speculative;

// DDTree: tree of likely continuations built from draft logits
struct common_speculative_tree {
    std::vector<llama_token> tokens;   // [n_nodes] tree node tokens (topological order)
    std::vector<int32_t>     parents;  // [n_nodes+1] parent index (-1 for root, 0-based for nodes)
    std::vector<int32_t>     depths;   // [n_nodes] depth (1-based: root's children = 1)
    std::vector<std::unordered_map<llama_token, int>> child_maps; // [n_nodes+1] token → child node index (1-based)
    std::vector<uint8_t>     visibility; // [(n_nodes+1)²] row-major: visibility[i*(n+1)+j] = node i can attend to node j
    int n_nodes = 0;
    int main_path_len = 0; // number of main-path nodes (indices 1..main_path_len in batch)
};

// comma separated list of all types
std::string common_speculative_type_name_str();

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// check if the llama_context is compatible for speculative decoding
// note: clears the memory of the context
bool common_speculative_is_compat(llama_context * ctx_tgt);

// Create a drafter context that can be shared across multiple common_speculative
// instances (DFlash multi-slot). The caller owns the returned context and must
// release it with llama_free after all dependent common_speculative instances
// have been freed. Returns nullptr if the speculative params have no draft model.
// topk / sample_temp / other per-ctx_dft config is applied here so the shared
// context is fully configured before it is wired into any common_speculative.
llama_context * common_speculative_create_ctx_dft(common_params_speculative & params);

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt,
        llama_context             * ctx_dft_shared = nullptr);

void common_speculative_free(common_speculative * spec);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt);

// sample up to n_draft tokens and add them to the batch using the draft model
llama_tokens common_speculative_draft(
                     common_speculative * spec,
        const common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last,
                     std::vector<float> * draft_log_probs = nullptr);

// informs the speculative decoder that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted);

// update implementations with logits from the verification decode
void common_speculative_update_logits(common_speculative * spec, llama_context * ctx, const llama_tokens & batch_tokens, int n_accepted);

// DDTree: build a tree of likely continuations from draft logits
// tree_budget: max tree nodes (0 = flat DFlash, >0 = DDTree)
common_speculative_tree common_speculative_draft_tree(
                     common_speculative * spec,
        const common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last,
                                    int   tree_budget);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);
