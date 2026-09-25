#include "CXGrammarBridge.h"

#include <xgrammar/xgrammar.h>

#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct xgb_compiler {
  xgrammar::TokenizerInfo info;
  xgrammar::GrammarCompiler compiler;
  xgb_compiler(xgrammar::TokenizerInfo info, int max_threads)
      : info(info), compiler(info, max_threads) {}
};

struct xgb_grammar {
  xgrammar::CompiledGrammar grammar;
};

struct xgb_matcher {
  xgrammar::GrammarMatcher matcher;
};

namespace {

void write_error(char *error, size_t capacity, const char *message) {
  if (error == nullptr || capacity == 0) return;
  std::strncpy(error, message, capacity - 1);
  error[capacity - 1] = '\0';
}

}  // namespace

extern "C" {

xgb_compiler *xgb_compiler_create(const char *const *tokens, const int32_t *lengths, int32_t count,
                                  int32_t vocab_size, const int32_t *stop_ids, int32_t stop_count,
                                  int32_t max_threads, char *error, size_t error_capacity) {
  try {
    std::vector<std::string> vocab;
    vocab.reserve(static_cast<size_t>(count));
    for (int32_t i = 0; i < count; ++i) {
      vocab.emplace_back(tokens[i] == nullptr ? "" : std::string(tokens[i], lengths[i]));
    }
    std::vector<int32_t> stops(stop_ids, stop_ids + stop_count);
    xgrammar::TokenizerInfo info(vocab, xgrammar::VocabType::RAW, vocab_size, stops, false);
    return new xgb_compiler(info, max_threads > 0 ? max_threads : 8);
  } catch (const std::exception &e) {
    write_error(error, error_capacity, e.what());
  } catch (...) {
    write_error(error, error_capacity, "unknown XGrammar failure creating the compiler");
  }
  return nullptr;
}

void xgb_compiler_free(xgb_compiler *compiler) { delete compiler; }

xgb_grammar *xgb_compile_json_schema(xgb_compiler *compiler, const char *schema,
                                     int32_t max_whitespace, char *error, size_t error_capacity) {
  if (compiler == nullptr || schema == nullptr) {
    write_error(error, error_capacity, "no compiler or schema");
    return nullptr;
  }
  try {
    std::optional<int> whitespace;
    if (max_whitespace >= 0) whitespace = max_whitespace;
    auto grammar = compiler->compiler.CompileJSONSchema(
        schema, /*any_whitespace=*/true, /*indent=*/std::nullopt, /*separators=*/std::nullopt,
        /*strict_mode=*/true, whitespace, /*any_order=*/false);
    return new xgb_grammar{grammar};
  } catch (const std::exception &e) {
    write_error(error, error_capacity, e.what());
  } catch (...) {
    write_error(error, error_capacity, "unknown XGrammar failure compiling the schema");
  }
  return nullptr;
}

void xgb_grammar_free(xgb_grammar *grammar) { delete grammar; }

xgb_matcher *xgb_matcher_create(const xgb_grammar *grammar) {
  if (grammar == nullptr) return nullptr;
  try {
    return new xgb_matcher{xgrammar::GrammarMatcher(grammar->grammar)};
  } catch (...) {
    return nullptr;
  }
}

void xgb_matcher_free(xgb_matcher *matcher) { delete matcher; }

int32_t xgb_bitmask_words(int32_t vocab_size) { return xgrammar::GetBitmaskSize(vocab_size); }

bool xgb_matcher_fill_bitmask(xgb_matcher *matcher, int32_t *bitmask, int32_t words) {
  if (matcher == nullptr || bitmask == nullptr || words <= 0) return false;
  try {
    int64_t shape[1] = {words};
    DLTensor tensor;
    tensor.data = bitmask;
    tensor.device = DLDevice{kDLCPU, 0};
    tensor.ndim = 1;
    tensor.dtype = DLDataType{kDLInt, 32, 1};
    tensor.shape = shape;
    tensor.strides = nullptr;
    tensor.byte_offset = 0;
    matcher->matcher.FillNextTokenBitmask(&tensor);
    return true;
  } catch (...) {
    return false;
  }
}

bool xgb_matcher_accept(xgb_matcher *matcher, int32_t token) {
  if (matcher == nullptr) return false;
  try {
    return matcher->matcher.AcceptToken(token);
  } catch (...) {
    return false;
  }
}

bool xgb_matcher_is_completed(const xgb_matcher *matcher) {
  return matcher != nullptr && matcher->matcher.IsCompleted();
}

bool xgb_matcher_is_terminated(const xgb_matcher *matcher) {
  return matcher != nullptr && matcher->matcher.IsTerminated();
}

}  // extern "C"
