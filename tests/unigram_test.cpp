#include "unigram.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "doctest.h"
#include "utf32.hpp"

using CodePoints = UTF32::CodePoints;
using Model = std::unordered_map<CodePoints, double, UTF32::Hasher>;

// encode_word starts the first position at 1.0, like the HF course does, so
// every score it returns is -log-probability + this offset.
constexpr double START_SCORE{1.0};

struct UnigramTestAccess {
  static std::pair<std::vector<CodePoints>, std::optional<double>>
  encode_word(Unigram &tok, const CodePoints &word) {
    return tok.encode_word(word);
  }
  static Model &model(Unigram &tok) { return tok.model; }
  static auto &word_counts(Unigram &tok) { return tok.word_counts; }
  static void populate_freqs(Unigram &tok) { tok.populate_freqs(); }
  static double unk_score(const Unigram &tok) { return tok.unk_score; }
  static const auto &id_to_token(const Unigram &tok) { return tok.id_to_token; }
  static const auto &token_to_id(const Unigram &tok) { return tok.token_to_id; }
  static double compute_loss(const Unigram &tok) { return tok.compute_loss(); }
  static std::vector<CodePoints> split(const Unigram &tok,
                                       const CodePoints &text) {
    return tok.split(text);
  }
};

namespace {

CodePoints cps(const std::string &str) { return UTF32{}.decode(str); }

std::string str(const CodePoints &code_points) {
  return UTF32{}.encode(code_points);
}

std::vector<std::string> strs(const std::vector<CodePoints> &words) {
  std::vector<std::string> out;
  for (const auto &word : words)
    out.push_back(str(word));
  return out;
}

// The model of the HF course's toy example (section 6.7): the frequencies it
// lists, divided by their total of 210.
Model course_toy_model() {
  const std::vector<std::pair<std::string, double>> freqs{
      {"h", 15}, {"u", 36}, {"g", 20}, {"hu", 15}, {"ug", 20},
      {"p", 17}, {"pu", 17}, {"n", 16}, {"un", 16}, {"b", 4},
      {"bu", 4}, {"s", 5},  {"hug", 15}, {"gs", 5}, {"ugs", 5}};
  Model model;
  for (const auto &[token, freq] : freqs)
    model[cps(token)] = -std::log(freq / 210.0);
  return model;
}

// Lowest total score over every way of cutting word[from:] into model tokens.
// Infinity when there is none.
double brute_force(const CodePoints &word, size_t from, const Model &model) {
  if (from == word.size())
    return 0.0;
  double best{std::numeric_limits<double>::infinity()};
  for (size_t end = from + 1; end <= word.size(); end++) {
    const auto it{model.find(
        CodePoints(word.begin() + static_cast<std::ptrdiff_t>(from),
                   word.begin() + static_cast<std::ptrdiff_t>(end)))};
    if (it == model.end())
      continue;
    best = std::min(best, it->second + brute_force(word, end, model));
  }
  return best;
}

} // namespace

TEST_CASE("encode_word: the course's toy example") {
  Unigram tok{{.vocab_size = 300}};
  UnigramTestAccess::model(tok) = course_toy_model();

  const auto [tokens, score]{UnigramTestAccess::encode_word(tok, cps("unhug"))};

  CHECK(strs(tokens) == std::vector<std::string>{"un", "hug"});
  REQUIRE(score.has_value());
  // The course: un + hug has probability 0.005442.
  CHECK(*score - START_SCORE ==
        doctest::Approx(-std::log((16.0 / 210.0) * (15.0 / 210.0))));
  CHECK(std::exp(-(*score - START_SCORE)) == doctest::Approx(0.005442).epsilon(
                                                 1e-3));
}

TEST_CASE("encode_word: a whole word in the model is one token") {
  Unigram tok{{.vocab_size = 300}};
  UnigramTestAccess::model(tok) = course_toy_model();

  const auto [tokens, score]{UnigramTestAccess::encode_word(tok, cps("hug"))};

  CHECK(strs(tokens) == std::vector<std::string>{"hug"});
  REQUIRE(score.has_value());
  CHECK(*score - START_SCORE == doctest::Approx(-std::log(15.0 / 210.0)));
}

TEST_CASE("encode_word: a single character word") {
  Unigram tok{{.vocab_size = 300}};
  UnigramTestAccess::model(tok) = course_toy_model();

  const auto [tokens, score]{UnigramTestAccess::encode_word(tok, cps("s"))};

  CHECK(strs(tokens) == std::vector<std::string>{"s"});
  REQUIRE(score.has_value());
  CHECK(*score - START_SCORE == doctest::Approx(-std::log(5.0 / 210.0)));
}

TEST_CASE("encode_word: a word that cannot be covered has no score") {
  Unigram tok{{.vocab_size = 300}};
  UnigramTestAccess::model(tok) = course_toy_model();

  for (const auto *word : {"x", "hugx", "xhug", "huxg"}) {
    const auto [tokens, score]{UnigramTestAccess::encode_word(tok, cps(word))};
    CHECK_FALSE(score.has_value());
  }
}

TEST_CASE("encode_word agrees with brute force on random models") {
  std::mt19937_64 rng{12345};
  const std::string alphabet{"abc"};

  for (int round = 0; round < 300; round++) {
    // All single characters, so every word can be covered, plus random longer
    // pieces.
    Model model;
    std::uniform_real_distribution<double> score_dist{0.1, 6.0};
    std::bernoulli_distribution included{0.4};
    for (const char ch : alphabet)
      model[cps(std::string(1, ch))] = score_dist(rng);
    for (const char a : alphabet)
      for (const char b : alphabet) {
        if (included(rng))
          model[cps(std::string{a, b})] = score_dist(rng);
        for (const char c : alphabet) {
          if (included(rng))
            model[cps(std::string{a, b, c})] = score_dist(rng);
        }
      }

    Unigram tok{{.vocab_size = 300}};
    UnigramTestAccess::model(tok) = model;

    std::uniform_int_distribution<size_t> len_dist{1, 9};
    std::uniform_int_distribution<size_t> ch_dist{0, alphabet.size() - 1};
    std::string text;
    for (size_t i = 0, n = len_dist(rng); i < n; i++)
      text += alphabet[ch_dist(rng)];
    const auto word{cps(text)};

    const auto [tokens, score]{UnigramTestAccess::encode_word(tok, word)};

    REQUIRE(score.has_value());
    CHECK(*score - START_SCORE ==
          doctest::Approx(brute_force(word, 0, model)).epsilon(1e-9));

    // The tokens tile the word, are all in the model, and add up to the score.
    CodePoints joined;
    double sum{START_SCORE};
    for (const auto &token : tokens) {
      REQUIRE(model.contains(token));
      sum += model.at(token);
      joined.insert(joined.end(), token.begin(), token.end());
    }
    CHECK(joined == word);
    CHECK(sum == doctest::Approx(*score));
  }
}

TEST_CASE("split: spaces become the word-start marker") {
  Unigram tok{{.vocab_size = 300}};
  const auto words{
      [&](const std::string &text) {
        return strs(UnigramTestAccess::split(tok, cps(text)));
      }};

  CHECK(words("the cat sat") ==
        std::vector<std::string>{"the", "\xE2\x96\x81" "cat",
                                 "\xE2\x96\x81" "sat"});
  CHECK(words("Hello, world") ==
        std::vector<std::string>{"Hello", ",", "\xE2\x96\x81" "world"});
  CHECK(words(" a") == std::vector<std::string>{"\xE2\x96\x81" "a"});
  CHECK(words("a ") == std::vector<std::string>{"a", "\xE2\x96\x81"});
  CHECK(words("a  b") == std::vector<std::string>{"a", "\xE2\x96\x81",
                                                  "\xE2\x96\x81" "b"});
  CHECK(words("\"Hi\"") == std::vector<std::string>{"\"", "Hi", "\""});
  CHECK(words("wow!!") == std::vector<std::string>{"wow", "!", "!"});
  CHECK(words("").empty());
}

TEST_CASE("split never returns an empty word") {
  Unigram tok{{.vocab_size = 300}};
  for (const auto *text :
       {"", " ", "  ", "!", "!!", ", ,", "a, b. c!", "  x  ", "...", "\"\"",
        "end."}) {
    for (const auto &word : UnigramTestAccess::split(tok, cps(text)))
      CHECK_FALSE(word.empty());
  }
}

TEST_CASE("populate_freqs: the toy corpus counts every substring") {
  Unigram tok{{.vocab_size = 300}};
  auto &counts{UnigramTestAccess::word_counts(tok)};
  counts[cps("hug")] = 10;
  counts[cps("pug")] = 5;
  counts[cps("pun")] = 12;
  counts[cps("bun")] = 4;
  counts[cps("hugs")] = 5;

  UnigramTestAccess::populate_freqs(tok);
  const auto &model{UnigramTestAccess::model(tok)};

  // 7 characters + 12 distinct substrings of length >= 2.
  CHECK(model.size() == 19);

  // Characters 113 + substrings 123. The course's list leaves out the
  // whole-word entries pug, pun, bun and hugs (26), which is how it gets 210.
  constexpr double TOTAL{236.0};
  CHECK(model.at(cps("ug")) == doctest::Approx(-std::log(20.0 / TOTAL)));
  CHECK(model.at(cps("u")) == doctest::Approx(-std::log(36.0 / TOTAL)));
  CHECK(model.at(cps("hug")) == doctest::Approx(-std::log(15.0 / TOTAL)));
  CHECK(model.at(cps("hugs")) == doctest::Approx(-std::log(5.0 / TOTAL)));

  double probability_sum{0.0};
  for (const auto &[token, score] : model)
    probability_sum += std::exp(-score);
  CHECK(probability_sum == doctest::Approx(1.0));
}

TEST_CASE("populate_freqs: a small seed keeps the best substrings") {
  // A model size of 4 gives a seed of 12: 7 characters and the 5 best
  // substrings, ug (20), pu (17), un (16), hu (15) and hug (15).
  Unigram tok{{.vocab_size = 4 + UNIGRAM_SPECIAL_COUNT}};
  auto &counts{UnigramTestAccess::word_counts(tok)};
  counts[cps("hug")] = 10;
  counts[cps("pug")] = 5;
  counts[cps("pun")] = 12;
  counts[cps("bun")] = 4;
  counts[cps("hugs")] = 5;

  UnigramTestAccess::populate_freqs(tok);
  const auto &model{UnigramTestAccess::model(tok)};

  CHECK(model.size() == 12);
  for (const auto *token : {"ug", "pu", "un", "hu", "hug"})
    CHECK(model.contains(cps(token)));
  CHECK_FALSE(model.contains(cps("pug")));
}

TEST_CASE("populate_freqs: vocab size larger than the candidates is fine") {
  Unigram tok{{.vocab_size = 100000}};
  auto &counts{UnigramTestAccess::word_counts(tok)};
  counts[cps("ab")] = 3;

  UnigramTestAccess::populate_freqs(tok);

  CHECK(UnigramTestAccess::model(tok).size() == 3);
}

TEST_CASE("populate_freqs: a seed smaller than the characters is fine") {
  Unigram tok{{.vocab_size = 1 + UNIGRAM_SPECIAL_COUNT}};
  auto &counts{UnigramTestAccess::word_counts(tok)};
  counts[cps("abcd")] = 3;

  UnigramTestAccess::populate_freqs(tok);

  // All four characters stay: single characters are never dropped.
  CHECK(UnigramTestAccess::model(tok).size() == 4);
}

TEST_CASE("feed, populate_freqs and encode_word work together") {
  Unigram tok{{.vocab_size = 200}};
  for (const auto *line :
       {"the cat sat on the mat.", "the dog sat on the log.",
        "a cat, a dog, and a log!"})
    tok.feed(line);
  UnigramTestAccess::populate_freqs(tok);

  for (const auto *line :
       {"the cat sat on the mat.", "a cat, a dog, and a log!"}) {
    for (const auto &word : UnigramTestAccess::split(tok, cps(line))) {
      const auto [tokens, score]{UnigramTestAccess::encode_word(tok, word)};
      REQUIRE(score.has_value());
      CodePoints joined;
      for (const auto &token : tokens)
        joined.insert(joined.end(), token.begin(), token.end());
      CHECK(joined == word);
    }
  }
}

TEST_CASE("unk_score is worse than every token and <unk> is not in the model") {
  Unigram tok{{.vocab_size = 300}};
  auto &counts{UnigramTestAccess::word_counts(tok)};
  counts[cps("hug")] = 10;
  counts[cps("pug")] = 5;
  counts[cps("pun")] = 12;
  counts[cps("bun")] = 4;
  counts[cps("hugs")] = 5;

  UnigramTestAccess::populate_freqs(tok);
  const auto &model{UnigramTestAccess::model(tok)};

  double worst{0.0};
  for (const auto &[token, score] : model)
    worst = std::max(worst, score);

  CHECK(UnigramTestAccess::unk_score(tok) == doctest::Approx(worst + 10.0));
  CHECK_FALSE(model.contains(cps(UNIGRAM_UNK_TOKEN)));
}

TEST_CASE("compute_loss: covered words cost freq * score, uncovered ones unk") {
  Unigram tok{{.vocab_size = 300}};
  auto &counts{UnigramTestAccess::word_counts(tok)};
  counts[cps("hug")] = 10;
  counts[cps("pug")] = 5;

  UnigramTestAccess::populate_freqs(tok);
  const double covered{UnigramTestAccess::compute_loss(tok)};

  double expected{0.0};
  for (const auto &[word, freq] : counts) {
    const auto [tokens, score]{UnigramTestAccess::encode_word(tok, word)};
    REQUIRE(score.has_value());
    expected += static_cast<double>(freq) * *score;
  }
  CHECK(covered == doctest::Approx(expected));

  // A word with a character the model has never seen has no segmentation.
  counts[cps("x")] = 3;
  CHECK(UnigramTestAccess::compute_loss(tok) ==
        doctest::Approx(covered + 3.0 * UnigramTestAccess::unk_score(tok)));
}

namespace {

// The HF course's corpus, also the content of data/testfile.txt.
const std::vector<std::string> COURSE_CORPUS{
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and "
    "generate tokens."};

// Sizes here are model sizes, as in the HF course (100): the vocabulary also
// holds the special tokens.
Unigram train_on_course_corpus(int64_t model_size) {
  Unigram tok{{.vocab_size = model_size + UNIGRAM_SPECIAL_COUNT}};
  for (const auto &line : COURSE_CORPUS)
    tok.feed(line);
  tok.train();
  return tok;
}

// Training takes seconds, so each model size is trained once and every test
// gets its own copy.
Unigram trained_on_course_corpus(int64_t model_size) {
  static std::map<int64_t, Unigram> trained;
  auto it{trained.find(model_size)};
  if (it == trained.end())
    it = trained.emplace(model_size, train_on_course_corpus(model_size)).first;
  return it->second;
}

// Characters of the course corpus (the word-start marker included): with the
// special tokens, Unigram's base. Training never removes a character, so they
// are the single-character tokens of any trained model.
int64_t course_char_count() {
  auto tok{trained_on_course_corpus(100)};
  return std::ranges::count_if(UnigramTestAccess::model(tok),
                               [](const auto &item) {
                                 return item.first.size() == 1;
                               });
}

} // namespace

TEST_CASE("train prunes the seed down to exactly vocab_size") {
  auto tok{trained_on_course_corpus(100)};
  const auto &model{UnigramTestAccess::model(tok)};

  CHECK(model.size() == 100);
}

TEST_CASE("train never removes a single character") {
  auto tok{trained_on_course_corpus(100)};
  const auto &model{UnigramTestAccess::model(tok)};

  for (const auto &[word, count] : UnigramTestAccess::word_counts(tok))
    for (const auto ch : word)
      CHECK(model.contains(CodePoints{ch}));
}

TEST_CASE("train: the pruned model is a probability distribution") {
  auto tok{trained_on_course_corpus(100)};

  double sum{0.0};
  for (const auto &[token, score] : UnigramTestAccess::model(tok))
    sum += std::exp(-score);

  CHECK(sum == doctest::Approx(1.0));
}

TEST_CASE("train is deterministic: the order of the lines doesn't matter") {
  // Another feed order fills the hash maps in another order, and that is what
  // their unspecified iteration order depends on. The model must not.
  auto forward{trained_on_course_corpus(100)};

  std::vector<std::string> reversed{COURSE_CORPUS.rbegin(),
                                    COURSE_CORPUS.rend()};
  std::vector<std::string> rotated{COURSE_CORPUS};
  std::ranges::rotate(rotated, rotated.begin() + 1);

  for (const auto &lines : {reversed, rotated}) {
    Unigram tok{{.vocab_size = 100 + UNIGRAM_SPECIAL_COUNT}};
    for (const auto &line : lines)
      tok.feed(line);
    tok.train();

    CHECK(UnigramTestAccess::model(tok) == UnigramTestAccess::model(forward));
  }
}

TEST_CASE("train: every training word is still covered without <unk>") {
  auto tok{trained_on_course_corpus(100)};

  for (const auto &[word, count] : UnigramTestAccess::word_counts(tok)) {
    const auto [tokens, score]{UnigramTestAccess::encode_word(tok, word)};
    CHECK(score.has_value());
  }
}

TEST_CASE("train: the vocabulary has exactly vocab_size entries") {
  auto tok{trained_on_course_corpus(100)};

  // The 100 model tokens plus <|endoftext|> and <unk>.
  CHECK(std::ssize(UnigramTestAccess::id_to_token(tok)) ==
        100 + UNIGRAM_SPECIAL_COUNT);
}

TEST_CASE("train: a vocab size below the characters plus specials is refused") {
  const auto base{course_char_count() + UNIGRAM_SPECIAL_COUNT};
  Unigram tok{{.vocab_size = base - 1}};
  for (const auto &line : COURSE_CORPUS)
    tok.feed(line);

  CHECK_THROWS_AS(tok.train(), std::invalid_argument);
}

TEST_CASE("train: a vocab size equal to the base keeps only characters") {
  const auto base{course_char_count() + UNIGRAM_SPECIAL_COUNT};
  Unigram tok{{.vocab_size = base}};
  for (const auto &line : COURSE_CORPUS)
    tok.feed(line);

  REQUIRE_NOTHROW(tok.train());
  for (const auto &[token, score] : UnigramTestAccess::model(tok))
    CHECK(token.size() == 1);
  CHECK(std::ssize(UnigramTestAccess::id_to_token(tok)) == base);
}

TEST_CASE("populate_freqs: min_freq leaves rare words out") {
  Unigram tok{{.vocab_size = 300, .min_freq = 2}};
  auto &counts{UnigramTestAccess::word_counts(tok)};
  counts[cps("hug")] = 10;
  counts[cps("zq")] = 1;

  UnigramTestAccess::populate_freqs(tok);
  const auto &model{UnigramTestAccess::model(tok)};

  CHECK(model.contains(cps("hug")));
  CHECK_FALSE(model.contains(cps("z")));
  CHECK_FALSE(model.contains(cps("zq")));
}

TEST_CASE("train: a larger target keeps more tokens") {
  auto small{trained_on_course_corpus(60)};
  auto large{trained_on_course_corpus(120)};

  CHECK(UnigramTestAccess::model(small).size() == 60);
  CHECK(UnigramTestAccess::model(large).size() == 120);
}

namespace {

// A vocab file in the temp directory that is removed when the test is done.
struct TempVocab {
  std::filesystem::path path;
  explicit TempVocab(const std::string &name)
      : path{std::filesystem::temp_directory_path() / name} {}
  ~TempVocab() {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
  }
  TempVocab(const TempVocab &) = delete;
  TempVocab &operator=(const TempVocab &) = delete;
};

std::string read_file(const std::filesystem::path &path) {
  std::ifstream file{path, std::ios::binary};
  return {std::istreambuf_iterator<char>{file},
          std::istreambuf_iterator<char>{}};
}

} // namespace

TEST_CASE("train gives <|endoftext|> id 0, <unk> id 1, then the model") {
  auto tok{trained_on_course_corpus(100)};
  const auto &ids{UnigramTestAccess::id_to_token(tok)};

  REQUIRE(ids.size() == 102);
  CHECK(str(ids[UNIGRAM_EOT_ID]) == UNIGRAM_EOT_TOKEN);
  CHECK(str(ids[UNIGRAM_UNK_ID]) == UNIGRAM_UNK_TOKEN);

  // The ids are a bijection with the tokens, and the best token comes first.
  const auto &by_token{UnigramTestAccess::token_to_id(tok)};
  CHECK(by_token.size() == ids.size());
  for (size_t id = 0; id < ids.size(); id++)
    CHECK(by_token.at(ids[id]) == static_cast<int64_t>(id));

  const auto &model{UnigramTestAccess::model(tok)};
  for (size_t id = 3; id < ids.size(); id++)
    CHECK(model.at(ids[id - 1]) <= model.at(ids[id]));
}

TEST_CASE("save_vocab then load_vocab restores model, unk score and ids") {
  TempVocab file{"unigram_test_roundtrip.txt"};
  auto trained{trained_on_course_corpus(100)};
  trained.save_vocab(file.path.string());

  Unigram loaded{{.vocab_size = 100}};
  loaded.load_vocab(file.path.string());

  CHECK(UnigramTestAccess::model(loaded) == UnigramTestAccess::model(trained));
  CHECK(UnigramTestAccess::unk_score(loaded) ==
        UnigramTestAccess::unk_score(trained));
  CHECK(UnigramTestAccess::id_to_token(loaded) ==
        UnigramTestAccess::id_to_token(trained));
  CHECK(UnigramTestAccess::token_to_id(loaded) ==
        UnigramTestAccess::token_to_id(trained));
}

TEST_CASE("a loaded tokenizer encodes words like the trained one") {
  TempVocab file{"unigram_test_encode.txt"};
  auto trained{trained_on_course_corpus(100)};
  trained.save_vocab(file.path.string());
  Unigram loaded{{.vocab_size = 100}};
  loaded.load_vocab(file.path.string());

  for (const auto &line : COURSE_CORPUS)
    for (const auto &word : UnigramTestAccess::split(trained, cps(line))) {
      const auto [expected, expected_score]{
          UnigramTestAccess::encode_word(trained, word)};
      const auto [actual, actual_score]{
          UnigramTestAccess::encode_word(loaded, word)};
      CHECK(actual == expected);
      CHECK(actual_score == expected_score);
    }
}

TEST_CASE("save_vocab writes one line per token, ids in order") {
  TempVocab file{"unigram_test_format.txt"};
  auto tok{trained_on_course_corpus(100)};
  tok.save_vocab(file.path.string());

  std::ifstream in{file.path};
  std::vector<std::string> lines;
  for (std::string line; std::getline(in, line);)
    lines.push_back(line);

  REQUIRE(lines.size() == 102);
  CHECK(lines[0].rfind("<|endoftext|> ", 0) == 0);
  CHECK(lines[1].rfind("<unk> ", 0) == 0);
}

TEST_CASE("save, load, save gives the same file") {
  TempVocab first{"unigram_test_first.txt"};
  TempVocab second{"unigram_test_second.txt"};
  auto trained{trained_on_course_corpus(100)};
  trained.save_vocab(first.path.string());

  Unigram loaded{{.vocab_size = 100}};
  loaded.load_vocab(first.path.string());
  loaded.save_vocab(second.path.string());

  CHECK(read_file(first.path) == read_file(second.path));
}

TEST_CASE("load_vocab rejects a missing or malformed file") {
  Unigram tok{{.vocab_size = 100}};
  CHECK_THROWS_AS(tok.load_vocab("this_file_does_not_exist.txt"),
                  std::runtime_error);

  TempVocab file{"unigram_test_bad.txt"};
  {
    std::ofstream out{file.path};
    out << "hug 1.5\nug 2.5\n"; // does not start with the special tokens
  }
  CHECK_THROWS_AS(tok.load_vocab(file.path.string()), std::runtime_error);
}

TEST_CASE("tokens with non-ASCII characters survive save and load") {
  TempVocab file{"unigram_test_utf8.txt"};
  Unigram tok{{.vocab_size = 100}};
  // "cafe cafe naive" and a Chinese greeting, as UTF-8 bytes.
  for (const auto *line : {"caf\xC3\xA9 caf\xC3\xA9 na\xC3\xAFve",
                           "\xE4\xBD\xA0\xE5\xA5\xBD \xE4\xBD\xA0\xE5\xA5\xBD"})
    tok.feed(line);
  tok.train();
  tok.save_vocab(file.path.string());

  Unigram loaded{{.vocab_size = 100}};
  loaded.load_vocab(file.path.string());

  CHECK(UnigramTestAccess::model(loaded) == UnigramTestAccess::model(tok));
  CHECK(UnigramTestAccess::model(loaded).contains(cps("caf\xC3\xA9")));
}

namespace {

// Everything the round-trip tests are allowed to use. The training corpus
// contains each of them, so none is an unknown character.
const std::vector<std::string> ATOMS{
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "A", "B", "C", "H", "T", "0", "1", "2", "9", ".", ",", "!", "?",
    ";", ":", "'", "\"", "(", ")", "-", " ", " ", " ",
    "\xC3\xA9",         // e acute
    "\xC3\xAF",         // i diaeresis
    "\xE4\xBD\xA0",     // a Chinese character
    "\xE5\xA5\xBD",     // another one
    "\xF0\x9F\x98\x80", // an emoji
};

// A model size, as for train_on_course_corpus.
Unigram train_on_atoms(int64_t model_size) {
  Unigram tok{{.vocab_size = model_size + UNIGRAM_SPECIAL_COUNT}};
  for (const auto *line :
       {"The cat sat on the mat, and the dog sat on the log.",
        "\"Hello!\" said Tom (and he was happy). Is it 2019? Yes: 1990-2009.",
        "caf\xC3\xA9 na\xC3\xAFve; a bad cab, a bag, a gab and a jam."})
    tok.feed(line);

  // One line with every atom, so that no atom is an unknown character.
  std::string all_atoms;
  for (const auto &atom : ATOMS)
    all_atoms += atom + " ";
  tok.feed(all_atoms);

  tok.train();
  return tok;
}

// Training is the slow part, so each model size is trained once and every
// test gets its own copy.
Unigram trained_on_atoms(int64_t model_size) {
  static std::map<int64_t, Unigram> trained;
  auto it{trained.find(model_size)};
  if (it == trained.end())
    it = trained.emplace(model_size, train_on_atoms(model_size)).first;
  return it->second;
}

std::string random_text(std::mt19937_64 &rng, size_t max_atoms) {
  std::uniform_int_distribution<size_t> len_dist{0, max_atoms};
  std::uniform_int_distribution<size_t> atom_dist{0, ATOMS.size() - 1};
  std::string text;
  for (size_t i = 0, n = len_dist(rng); i < n; i++)
    text += ATOMS[atom_dist(rng)];
  return text;
}

} // namespace

TEST_CASE("decode(encode(text)) gives the text back, for any known characters") {
  // 4000 keeps every candidate. 60 prunes hard; the characters stay.
  for (const int64_t vocab_size : {4000, 60}) {
    auto tok{trained_on_atoms(vocab_size)};
    std::mt19937_64 rng{2024};

    for (int round = 0; round < 500; round++) {
      const auto text{random_text(rng, 40)};
      CHECK(tok.decode(tok.encode(text)) == text);
    }
  }
}

TEST_CASE("round trip: the course corpus, line by line") {
  auto tok{trained_on_course_corpus(100)};

  for (const auto &line : COURSE_CORPUS)
    CHECK(tok.decode(tok.encode(line)) == line);
}

TEST_CASE("round trip: spaces at the start, end and in a row") {
  auto tok{trained_on_atoms(4000)};

  for (const auto *text : {"", " ", "   ", " a", "a ", "  a  b  ", "a  b",
                           "!", " !", "! ", "a.b", "..", ". .", "(a)", "- -"})
    CHECK(tok.decode(tok.encode(text)) == text);
}

TEST_CASE("encode: the empty string has no tokens") {
  auto tok{trained_on_atoms(4000)};

  CHECK(tok.encode("").empty());
  CHECK(tok.decode({}).empty());
}

TEST_CASE("encode: ids are valid, and neither <unk> nor <|endoftext|>") {
  auto tok{trained_on_atoms(60)};
  const auto size{static_cast<int64_t>(UnigramTestAccess::id_to_token(tok).size())};
  std::mt19937_64 rng{7};

  for (int round = 0; round < 200; round++)
    for (const auto id : tok.encode(random_text(rng, 40))) {
      CHECK(id >= 2);
      CHECK(id < size);
    }
}

TEST_CASE("encode: the same text always gives the same ids") {
  auto tok{trained_on_atoms(60)};
  std::mt19937_64 rng{11};

  for (int round = 0; round < 100; round++) {
    const auto text{random_text(rng, 40)};
    CHECK(tok.encode(text) == tok.encode(text));
  }
}

TEST_CASE("encode matches the tokens of encode_word, word by word") {
  auto tok{trained_on_atoms(60)};
  const std::string text{"the cat, a bad cab (and 2019)!"};

  std::vector<int64_t> expected;
  for (const auto &word : UnigramTestAccess::split(tok, cps(text)))
    for (const auto &token : UnigramTestAccess::encode_word(tok, word).first)
      expected.push_back(UnigramTestAccess::token_to_id(tok).at(token));

  CHECK(tok.encode(text) == expected);
}

TEST_CASE("encode: a trained, a saved and a loaded tokenizer agree") {
  TempVocab file{"unigram_test_encode_agree.txt"};
  auto trained{trained_on_atoms(60)};
  trained.save_vocab(file.path.string());
  Unigram loaded{{.vocab_size = 60}};
  loaded.load_vocab(file.path.string());

  std::mt19937_64 rng{99};
  for (int round = 0; round < 100; round++) {
    const auto text{random_text(rng, 40)};
    const auto ids{trained.encode(text)};
    CHECK(loaded.encode(text) == ids);
    CHECK(loaded.decode(ids) == text);
  }
}

TEST_CASE("encode: text is normalized to NFC first") {
  auto tok{trained_on_atoms(4000)};
  const std::string decomposed{"cafe\xCC\x81"}; // e + combining acute
  const std::string composed{"caf\xC3\xA9"};

  CHECK(tok.encode(decomposed) == tok.encode(composed));
  CHECK(tok.decode(tok.encode(decomposed)) == composed);
}

TEST_CASE("encode: a word with an unseen character becomes one <unk>") {
  auto tok{trained_on_atoms(4000)};
  const std::string sharp_s{"\xC3\x9F"}; // never in the corpus

  const auto ids{tok.encode("the " + sharp_s + "ab cat")};
  const auto plain{tok.encode("the cat")};

  // Today the whole word is lost, not only the unseen character.
  CHECK(std::ranges::count(ids, UNIGRAM_UNK_ID) == 1);
  CHECK(ids.size() == plain.size() + 1);
  CHECK(tok.decode(tok.encode(sharp_s)) == "<unk>");
  // The words around it are untouched.
  CHECK(tok.decode(tok.encode("the " + sharp_s + "ab cat")) == "the<unk> cat");
}

TEST_CASE("encode: nothing is unknown when every character was seen") {
  auto tok{trained_on_atoms(60)};
  std::mt19937_64 rng{5};

  for (int round = 0; round < 200; round++)
    CHECK(std::ranges::count(tok.encode(random_text(rng, 40)),
                             UNIGRAM_UNK_ID) == 0);
}

TEST_CASE("decode: the special tokens and out-of-range ids") {
  auto tok{trained_on_atoms(60)};
  const auto size{static_cast<int64_t>(UnigramTestAccess::id_to_token(tok).size())};

  CHECK(tok.decode({UNIGRAM_EOT_ID}) == UNIGRAM_EOT_TOKEN);
  CHECK(tok.decode({UNIGRAM_UNK_ID}) == UNIGRAM_UNK_TOKEN);
  CHECK_THROWS_AS(tok.decode({-1}), std::out_of_range);
  CHECK_THROWS_AS(tok.decode({size}), std::out_of_range);
  CHECK_THROWS_AS(tok.decode({0, 5, size + 10}), std::out_of_range);
}

TEST_CASE("decode: word-start markers become spaces") {
  auto tok{trained_on_atoms(4000)};
  const auto &by_token{UnigramTestAccess::token_to_id(tok)};

  const auto marker{by_token.at(CodePoints{U'▁'})};
  const auto a{by_token.at(cps("a"))};

  CHECK(tok.decode({a, marker, a}) == "a a");
  CHECK(tok.decode({marker}) == " ");
}

TEST_CASE("encode: tabs and newlines come back as spaces") {
  auto tok{trained_on_atoms(4000)};

  // Every ASCII whitespace character is the word-start marker.
  CHECK(tok.decode(tok.encode("a\tb\nc")) == "a b c");
}

TEST_CASE("encode needs a vocabulary") {
  Unigram tok{{.vocab_size = 100}};
  CHECK_THROWS_AS(tok.encode("anything"), std::logic_error);
}
