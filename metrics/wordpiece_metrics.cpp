#include "metrics.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "tokenizers/wordpiece.hpp"

namespace metrics {

namespace {

const std::string VOCAB_FILE{"wordpiece_vocab.txt"};
constexpr int64_t VOCAB_SIZE{4096};
constexpr int64_t MIN_FREQ{10};

} // namespace

void run_wordpiece_metrics() {
  std::cout << "######## WordPiece (" << VOCAB_FILE << ") ########\n";

  WordPiece tok{{.vocab_size = VOCAB_SIZE, .min_freq = MIN_FREQ}};
  tok.load_vocab(VOCAB_FILE);

  // Special tokens get the first IDs, in SPECIAL_TOKENS order (see
  // populate_vocab).
  const auto unk_id{std::ranges::find(SPECIAL_TOKENS, "[UNK]") -
                    SPECIAL_TOKENS.begin()};
  std::vector<bool> used_ids(VOCAB_SIZE, false);

  const auto start{std::chrono::steady_clock::now()};
  print_stats("validation: " + VALID_FILE,
              evaluate(tok, VALID_FILE, std::numeric_limits<int64_t>::max(),
                       unk_id, used_ids));
  print_stats("train sample: first " + std::to_string(TRAIN_SAMPLE_LINES) +
                  " lines of " + TRAIN_FILE,
              evaluate(tok, TRAIN_FILE, TRAIN_SAMPLE_LINES, unk_id, used_ids));
  const auto end{std::chrono::steady_clock::now()};

  print_vocab_usage(used_ids);
  std::cout << "Took " << std::chrono::duration<double>(end - start).count()
            << " s\n";
}

} // namespace metrics
