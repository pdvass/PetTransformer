#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "tokenizers/tokenizer.hpp"

namespace metrics {

const std::string TRAIN_FILE{"data\\TinyStoriesV2-GPT4-train.txt"};
const std::string VALID_FILE{"data\\TinyStoriesV2-GPT4-valid.txt"};
// Lines of the train file to compare against. Validation uses the whole file.
constexpr int64_t TRAIN_SAMPLE_LINES{100'000};

struct Stats {
  int64_t lines{0};
  int64_t words{0};
  int64_t tokens{0};
  int64_t unks{0};
  int64_t lines_with_unk{0};
  // encode(decode(ids)) != ids
  int64_t unstable_lines{0};
  // decoded text != original text, ignoring whitespace (ASCII lines only)
  int64_t text_mismatches{0};
};

// Encodes up to max_lines lines of the file at path and collects statistics.
// Marks every ID seen in used_ids (sized to the vocab).
Stats evaluate(Tokenizer &tok, const std::string &path, int64_t max_lines,
               int64_t unk_id, std::vector<bool> &used_ids);

void print_stats(const std::string &name, const Stats &stats);
void print_vocab_usage(const std::vector<bool> &used_ids);
double percent(int64_t part, int64_t whole);

// One function per tokenizer. Each loads its saved vocab and prints its
// metrics.
void run_wordpiece_metrics();

} // namespace metrics
