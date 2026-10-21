#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <exception>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#define NOMINMAX
#include <windows.h>

#include "tokenizers/bpe.hpp"
#include "tokenizers/unigram.hpp"
#include "tokenizers/wordpiece.hpp"

constexpr int DEBUG = 1;

namespace {

const std::string TRAIN_FILE{DEBUG ? "data\\testfile.txt"
                                   : "data\\TinyStoriesV2-GPT4-train.txt"};

const std::string VALID_FILE{DEBUG ? "data\\testfile.txt"
                                   : "data\\TinyStoriesV2-GPT4-valid.txt"};

constexpr int64_t MIN_FREQ{DEBUG ? 0 : 10};

// DEBUG uses the HF course's sizes: 100 entries, plus Unigram's 2 special
// tokens so that its model still has 100. BPE's base alone is 257.
template <typename Tok> constexpr TrainConfig train_config() {
  int64_t vocab_size{4096};
  if constexpr (DEBUG != 0) {
    if constexpr (std::is_same_v<Tok, BPE>)
      vocab_size = 300;
    else if constexpr (std::is_same_v<Tok, Unigram>)
      vocab_size = 100 + UNIGRAM_SPECIAL_COUNT;
    else
      vocab_size = 100;
  }
  return {.vocab_size = vocab_size, .min_freq = MIN_FREQ};
}
// Lines of the train file to compare against. Validation uses the whole file.
constexpr int64_t TRAIN_SAMPLE_LINES{100'000};
constexpr int64_t MAX_EXAMPLES{3};

std::ifstream open_or_throw(const std::string &path) {
  std::ifstream file{path};
  if (!file)
    throw std::runtime_error("Cannot open file " + path);
  return file;
}

// Adds the file and line to a tokenizer's error (e.g. invalid UTF-8), so a bad
// line can be found in a corpus of millions.
[[noreturn]] void rethrow_at(const std::string &path, int64_t line_number,
                             const std::exception &e) {
  throw std::runtime_error(
      std::format("{}:{}: {}", path, line_number, e.what()));
}

enum class TokenizerKind { WordPiece, Bpe, Unigram };
enum class Action { Train, Validate };

std::optional<TokenizerKind> parse_tokenizer(std::string_view arg) {
  if (arg == "wp")
    return TokenizerKind::WordPiece;
  if (arg == "bpe")
    return TokenizerKind::Bpe;
  if (arg == "uni")
    return TokenizerKind::Unigram;
  return std::nullopt;
}

std::optional<Action> parse_action(std::string_view arg) {
  if (arg == "--train" || arg == "-t")
    return Action::Train;
  if (arg == "--validate" || arg == "-v")
    return Action::Validate;
  return std::nullopt;
}

template <typename Tok> void train_model() {
  std::string tinyStoriesText;
  std::ifstream tinyStoriesTrain{open_or_throw(TRAIN_FILE)};
  Tok tok{train_config<Tok>()};

  auto start{std::chrono::steady_clock::now()};
  int64_t line_number{0};
  while (std::getline(tinyStoriesTrain, tinyStoriesText)) {
    line_number++;
    try {
      tok.feed(tinyStoriesText);
    } catch (const std::exception &e) {
      rethrow_at(TRAIN_FILE, line_number, e);
    }
  }
  auto end{std::chrono::steady_clock::now()};
  std::cout << "Took " << (end - start) << " to feed" << "\n";
  start = std::chrono::steady_clock::now();
  tok.train();
  end = std::chrono::steady_clock::now();

  std::cout << "Took " << (end - start) << " to train" << "\n";

  tok.save_vocab();
  // tok.print_info();
}

// Counts words the way WordPiece::split does: whitespace separates words,
// every punctuation character is a word of its own. Approximate for
// non-ASCII text, since normalize() runs before split() in the tokenizer.
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

// What a tokenizer promises, so the report only shows numbers that mean
// something for it.
struct Traits {
  // ID of [UNK], -1 when the tokenizer has none (it matches no ID).
  int64_t unk_id;
  // decode(encode(s)) == s for every s, not only up to whitespace and ASCII.
  bool exact;
};

struct Stats {
  int64_t lines{0};
  int64_t words{0};
  int64_t tokens{0};
  int64_t unks{0};
  int64_t lines_with_unk{0};
  // encode(decode(ids)) != ids
  int64_t unstable_lines{0};
  // decoded text != original text. For tokenizers that aren't exact, this
  // ignores whitespace and only looks at ASCII lines.
  int64_t text_mismatches{0};
};

Stats evaluate(Tokenizer &tok, const std::string &path, int64_t max_lines,
               const Traits &traits, std::vector<bool> &used_ids) {
  std::ifstream file{open_or_throw(path)};
  Stats stats{};
  int64_t examples_shown{0};
  std::string line;

  while (stats.lines < max_lines && std::getline(file, line)) {
    stats.lines++;
    const auto ids{[&] {
      try {
        return tok.encode(line);
      } catch (const std::exception &e) {
        rethrow_at(path, stats.lines, e);
      }
    }()};
    const auto unks{std::ranges::count(ids, traits.unk_id)};

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
    const bool text_differs{traits.exact ? decoded != line
                                         : is_ascii(line) &&
                                               without_whitespace(decoded) !=
                                                   without_whitespace(line)};
    stats.unstable_lines += unstable;
    stats.text_mismatches += text_differs;

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

void print_stats(const std::string &name, const Stats &stats,
                 const Traits &traits) {
  std::cout << std::fixed << std::setprecision(3) << "== " << name << " ==\n"
            << "  lines:        " << stats.lines << "\n"
            << "  words:        " << stats.words << "\n"
            << "  tokens:       " << stats.tokens << "\n"
            << "  tokens/word:  "
            << (stats.words == 0 ? 0.0
                                 : static_cast<double>(stats.tokens) /
                                       static_cast<double>(stats.words))
            << "\n"
            << "  encode(decode) != encode:  " << stats.unstable_lines << "\n";
  if (traits.unk_id >= 0)
    std::cout << "  [UNK] tokens: " << stats.unks << " ("
              << percent(stats.unks, stats.tokens) << "%)\n"
              << "  lines with [UNK]:          " << stats.lines_with_unk
              << "\n";
  std::cout << (traits.exact ? "  text differs:              "
                             : "  text differs (ASCII only): ")
            << stats.text_mismatches << "\n";
}

template <typename Tok> void validate_model() {
  constexpr auto config{train_config<Tok>()};
  Tok tok{config};
  tok.load_vocab();

  // Only WordPiece has an [UNK] token, and only it can lose text: its special
  // tokens get the first IDs, in SPECIAL_TOKENS order (see populate_vocab).
  // Byte-level BPE can encode any text and gives it back unchanged.
  Traits traits{.unk_id = -1, .exact = true};
  if constexpr (std::is_same_v<Tok, WordPiece>) {
    traits.unk_id =
        std::ranges::find(SPECIAL_TOKENS, "[UNK]") - SPECIAL_TOKENS.begin();
    traits.exact = false;
  } else if constexpr (std::is_same_v<Tok, Unigram>) {
    // Characters it has never seen become <unk> and can't be decoded back.
    traits.unk_id = UNIGRAM_UNK_ID;
  }
  std::vector<bool> used_ids(config.vocab_size, false);

  const auto start{std::chrono::steady_clock::now()};
  print_stats("validation: " + VALID_FILE,
              evaluate(tok, VALID_FILE, std::numeric_limits<int64_t>::max(),
                       traits, used_ids),
              traits);
  print_stats("train sample: first " + std::to_string(TRAIN_SAMPLE_LINES) +
                  " lines of " + TRAIN_FILE,
              evaluate(tok, TRAIN_FILE, TRAIN_SAMPLE_LINES, traits, used_ids),
              traits);
  const auto end{std::chrono::steady_clock::now()};

  const auto used{std::ranges::count(used_ids, true)};
  std::cout << "== vocab ==\n"
            << "  tokens used at least once: " << used << " / "
            << config.vocab_size << " (" << percent(used, config.vocab_size)
            << "%)\n"
            << "Took " << std::chrono::duration<double>(end - start).count()
            << " s to validate\n";
}

template <typename Tok> void run(Action action) {
  if (action == Action::Train) {
    train_model<Tok>();
  } else {
    validate_model<Tok>();
  }
}

} // namespace

int main(int argc, char *argv[]) {
  SetConsoleOutputCP(CP_UTF8);
  const std::string usage{"Usage: PetTransformer [wp | bpe | uni] "
                          "(--train | -t) | (--validate | -v)\n"
                          "  wp = wordpiece (default), bpe, uni = unigram\n"};
  if (argc < 2 || argc > 3) {
    std::cerr << usage;
    return 1;
  }

  // The tokenizer, when given, is always the first argument.
  auto kind{TokenizerKind::WordPiece};
  int action_index{1};
  if (argc == 3) {
    const auto parsed{parse_tokenizer(argv[1])};
    if (!parsed) {
      std::cerr << "Unknown tokenizer: " << argv[1] << "\n" << usage;
      return 1;
    }
    kind = *parsed;
    action_index = 2;
  }

  const auto action{parse_action(argv[action_index])};
  if (!action) {
    std::cerr << usage;
    return 1;
  }

  try {
    switch (kind) {
    case TokenizerKind::WordPiece:
      run<WordPiece>(*action);
      break;
    case TokenizerKind::Bpe:
      run<BPE>(*action);
      break;
    case TokenizerKind::Unigram:
      run<Unigram>(*action);
      break;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
