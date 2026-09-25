// C surface over XGrammar (https://github.com/mlc-ai/xgrammar, Apache-2.0).
//
// Qwen3Chat drives grammar-constrained decoding through these handles so that no Swift module
// has to enable C++ interoperability. Every function is safe to call with a NULL handle and
// reports failure instead of throwing across the C boundary.
#ifndef CXGRAMMAR_BRIDGE_H
#define CXGRAMMAR_BRIDGE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct xgb_compiler xgb_compiler;
typedef struct xgb_grammar xgb_grammar;
typedef struct xgb_matcher xgb_matcher;

/// A compiler over one tokenizer vocabulary. `tokens[i]`/`lengths[i]` are the raw decoded bytes of
/// token `i` (length 0 for a token the grammar must never admit: special and control tokens).
/// `vocab_size` is the logits width, which may exceed `count`. Returns NULL on failure and writes
/// a NUL-terminated reason into `error` (capacity `error_capacity`).
xgb_compiler *xgb_compiler_create(const char *const *tokens, const int32_t *lengths, int32_t count,
                                  int32_t vocab_size, const int32_t *stop_ids, int32_t stop_count,
                                  int32_t max_threads, char *error, size_t error_capacity);
void xgb_compiler_free(xgb_compiler *compiler);

/// Compile a JSON Schema. Properties keep their declared order, additional properties are
/// rejected unless the schema allows them, and at most `max_whitespace` whitespace characters may
/// appear between tokens (negative for XGrammar's default). Compiled grammars are cached by the
/// compiler. Returns NULL and writes the reason on an invalid or unsupported schema.
xgb_grammar *xgb_compile_json_schema(xgb_compiler *compiler, const char *schema,
                                     int32_t max_whitespace, char *error, size_t error_capacity);
void xgb_grammar_free(xgb_grammar *grammar);

xgb_matcher *xgb_matcher_create(const xgb_grammar *grammar);
void xgb_matcher_free(xgb_matcher *matcher);

/// Words in one packed next-token bitmask: ceil(vocab_size / 32).
int32_t xgb_bitmask_words(int32_t vocab_size);

/// Fill `bitmask` (`words` int32 values, bit `i % 32` of word `i / 32` set when token `i` is
/// admissible). Returns false when the matcher cannot be queried.
bool xgb_matcher_fill_bitmask(xgb_matcher *matcher, int32_t *bitmask, int32_t words);

/// Advance by one sampled token. False when the token was not admissible (state unchanged).
bool xgb_matcher_accept(xgb_matcher *matcher, int32_t token);

/// The root rule has been fully matched; only a stop token may follow.
bool xgb_matcher_is_completed(const xgb_matcher *matcher);

/// A stop token has been accepted.
bool xgb_matcher_is_terminated(const xgb_matcher *matcher);

#ifdef __cplusplus
}
#endif

#endif
