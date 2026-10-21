#include "metrics.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

namespace metrics {

namespace {

constexpr int64_t MAX_EXAMPLES{3};

// Tokenizer-independent word count, using the same rules as WordPiece::split:
// whitespace separates words, every punctuation character is a word of its own.
// Approximate for non-ASCII text.
int64_t count_words(const std::string &line) {
  int64_t words{0};
  bool in_word{false};
  for (const char ch : line) {
    const auto c{static_cast<unsigned char>(ch)};
    if (std::isspace(c)) {
      in_word = false;
    } else if (std::ispunct(c)) {
      words++;
      in_word = false;
    } else if (!in_word) {
      words++;
      in_word = true;
    }
  }
  return words;
}

std::string without_whitespace(const std::string &str) {
  std::string out{};
  for (const char ch : str)
    if (!std::isspace(static_cast<unsigned char>(ch)))
      out += ch;
  return out;
}

bool is_ascii(const std::string &str) {
  return std::ranges::all_of(
      str, [](char ch) { return static_cast<unsigned char>(ch) < 128; });
}

} // namespace

Stats evaluate(Tokenizer &tok, const std::string &path, int64_t max_lines,
               int64_t unk_id, std::vector<bool> &used_ids) {
  std::ifstream file{path};
  if (!file)
    throw std::runtime_error("Cannot open file " + path);

  Stats stats{};
  int64_t examples_shown{0};
  std::string line;

  while (stats.lines < max_lines && std::getline(file, line)) {
    stats.lines++;
    const auto ids{tok.encode(line)};
    const auto unks{std::ranges::count(ids, unk_id)};

    stats.words += count_words(line);
    stats.tokens += std::ssize(ids);
    stats.unks += unks;
    for (const auto id : ids)
      if (id >= 0 && id < std::ssize(used_ids))
        used_ids[id] = true;

    // [UNK] can't be decoded back to the original text, so skip the round-trip
    // checks.
    if (unks > 0) {
      stats.lines_with_unk++;
      continue;
    }

    const auto decoded{tok.decode(ids)};
    const bool unstable{tok.encode(decoded) != ids};
    const bool text_differs{is_ascii(line) && without_whitespace(decoded) !=
                                                  without_whitespace(line)};
    stats.unstable_lines += unstable ? 1 : 0;
    stats.text_mismatches += text_differs ? 1 : 0;

    if ((unstable || text_differs) && examples_shown < MAX_EXAMPLES) {
      examples_shown++;
      std::cout << "  mismatch:\n    original: " << line
                << "\n    decoded:  " << decoded << "\n";
    }
  }
  return stats;
}

double percent(int64_t part, int64_t whole) {
  return whole == 0
             ? 0.0
             : 100.0 * static_cast<double>(part) / static_cast<double>(whole);
}

void print_stats(const std::string &name, const Stats &stats) {
  const double tokens_per_word{stats.words == 0
                                   ? 0.0
                                   : static_cast<double>(stats.tokens) /
                                         static_cast<double>(stats.words)};

  std::cout << std::fixed << std::setprecision(3) << "== " << name << " ==\n"
            << "  lines:        " << stats.lines << "\n"
            << "  words:        " << stats.words << "\n"
            << "  tokens:       " << stats.tokens << "\n"
            << "  tokens/word:  " << tokens_per_word << "\n"
            << "  [UNK] tokens: " << stats.unks << " ("
            << percent(stats.unks, stats.tokens) << "%)\n"
            << "  lines with [UNK]:          " << stats.lines_with_unk << "\n"
            << "  encode(decode) != encode:  " << stats.unstable_lines << "\n"
            << "  text differs (ASCII only): " << stats.text_mismatches << "\n";
}

void print_vocab_usage(const std::vector<bool> &used_ids) {
  const auto used{std::ranges::count(used_ids, true)};
  const auto size{std::ssize(used_ids)};
  std::cout << "== vocab ==\n"
            << "  tokens used at least once: " << used << " / " << size << " ("
            << percent(used, size) << "%)\n";
}

} // namespace metrics
