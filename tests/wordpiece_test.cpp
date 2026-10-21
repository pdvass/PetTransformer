#include "wordpiece.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <string>
#include <cstdint>
#include <iterator>
#include <map>
#include <stdexcept>
#include <vector>

#include "doctest.h"
#include "frozen_pair.hpp"

const std::array<std::string, 6> CORPUS = {
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and "
    "generate tokens.",
    // "  Hello,   WORLD! How's   everything?\nI'm fine.  ",
    // "Wait... WHAT?! Really?! — yes, really. (Maybe.)",
};

struct WordPieceTestAccess {
  static const auto &splits(const WordPiece &w) { return w.splits; }
  static const auto &vocab(const WordPiece &w) { return *w.vocab; }
  static const auto &alphabet(const WordPiece &w) { return w.alphabet; }
  static const auto &freqs(const WordPiece &w) { return w.counts; }
};

struct TrainedWordPiece {
  WordPiece tok{{.vocab_size = 70, .min_freq = 1}};

  TrainedWordPiece() {
    for (auto &sentence : CORPUS)
      tok.feed(sentence);
    tok.train();
  }
};

// The special tokens plus the alphabet, from a trained tokenizer.
int64_t base_size() {
  const TrainedWordPiece trained;
  return std::ssize(SPECIAL_TOKENS) +
         std::ssize(WordPieceTestAccess::alphabet(trained.tok));
}

TEST_CASE_FIXTURE(TrainedWordPiece, "the vocabulary has exactly vocab_size "
                                    "entries") {
  CHECK(WordPieceTestAccess::vocab(tok).len() == 70);
}

TEST_CASE("train refuses a vocab size below the base, accepts the base") {
  const auto base{base_size()};

  WordPiece below{{.vocab_size = base - 1, .min_freq = 1}};
  for (auto &sentence : CORPUS)
    below.feed(sentence);
  CHECK_THROWS_AS(below.train(), std::invalid_argument);

  WordPiece exact{{.vocab_size = base, .min_freq = 1}};
  for (auto &sentence : CORPUS)
    exact.feed(sentence);
  CHECK_NOTHROW(exact.train());
  CHECK(static_cast<int64_t>(WordPieceTestAccess::vocab(exact).len()) == base);
}

// Reference values from HF course, chapter 6.6.
// https://huggingface.co/docs/course/chapter6/6

TEST_CASE_FIXTURE(TrainedWordPiece, "word frequencies match HF reference") {
  std::map<std::string, int64_t> expected = {
      {",", 1},         {".", 4},          {"Course", 1},   {"Face", 1},
      {"Hopefully", 1}, {"Hugging", 1},    {"This", 3},     {"able", 1},
      {"about", 1},     {"algorithms", 1}, {"and", 1},      {"are", 1},
      {"be", 1},        {"chapter", 1},    {"generate", 1}, {"how", 1},
      {"is", 2},        {"section", 1},    {"several", 1},  {"shows", 1},
      {"the", 1},       {"they", 1},       {"to", 1},       {"tokenization", 1},
      {"tokenizer", 1}, {"tokens", 1},     {"trained", 1},  {"understand", 1},
      {"will", 1},      {"you", 1},
  };
  CHECK(WordPieceTestAccess::freqs(tok) == expected);
}

TEST_CASE_FIXTURE(TrainedWordPiece, "alphabet matches HF reference") {
  std::vector<std::string> expected = {
      "##a", "##b", "##c", "##d", "##e", "##f", "##g", "##h", "##i", "##k",
      "##l", "##m", "##n", "##o", "##p", "##r", "##s", "##t", "##u", "##v",
      "##w", "##y", "##z", ",",   ".",   "C",   "F",   "H",   "T",   "a",
      "b",   "c",   "g",   "h",   "i",   "s",   "t",   "u",   "w",   "y",
  };
  auto actual{WordPieceTestAccess::alphabet(tok)};
  std::sort(actual.begin(), actual.end());
  CHECK(actual == expected);
}

TEST_CASE_FIXTURE(TrainedWordPiece, "vocab reaches requested size") {
  CHECK(WordPieceTestAccess::vocab(tok).len() == 70);
}

TEST_CASE_FIXTURE(TrainedWordPiece,
                  "decode(encode(s)) round-trips the corpus") {
  for (auto &sentence : CORPUS)
    CHECK(tok.decode(tok.encode(sentence)) == sentence);
}

TEST_CASE_FIXTURE(TrainedWordPiece, "Hugging encodes like HF reference") {
  // No ties are involved in the merges that build "Hugg", so this matches HF
  // exactly.
  std::vector<std::string> expected = {"Hugg", "##i", "##n", "##g"};

  std::vector<std::string> actual{};
  for (auto id : tok.encode("Hugging"))
    actual.push_back(WordPieceTestAccess::vocab(tok).get(id).value());

  CHECK(actual == expected);
}

TEST_CASE_FIXTURE(TrainedWordPiece,
                  "save_vocab/load_vocab keep every word - id pair") {
  auto path{
      (std::filesystem::temp_directory_path() / "wordpiece_test_vocab.txt")
          .string()};
  tok.save_vocab(path);

  WordPiece loaded{{.vocab_size = 70, .min_freq = 1}};
  loaded.load_vocab(path);
  std::filesystem::remove(path);

  auto &original{WordPieceTestAccess::vocab(tok)};
  auto &restored{WordPieceTestAccess::vocab(loaded)};

  CHECK(restored.len() == original.len());
  for (auto &[word, id] : original) {
    CHECK(restored.get(word) == id);
    CHECK(restored.get(id) == word);
  }

  for (auto &sentence : CORPUS)
    CHECK(loaded.encode(sentence) == tok.encode(sentence));
}

// Snapshot of this implementation's own output, NOT the HF reference.
// Ties between equal scores are broken differently: HF takes the first pair in
// corpus order, std::map iterates alphabetically. E.g. HF gives
// generate -> g ##e ##n ##e ##r ##a ##t ##e and algorithms -> ... ##i ##thms.
TEST_CASE_FIXTURE(TrainedWordPiece, "splits snapshot") {
  std::map<std::string, std::vector<std::string>> expected = {
      {"This", {"Th", "##i", "##s"}},
      {"is", {"i", "##s"}},
      {"the", {"th", "##e"}},
      {"Hugging", {"Hugg", "##i", "##n", "##g"}},
      {"Face", {"Fac", "##e"}},
      {"Course", {"C", "##o", "##u", "##r", "##s", "##e"}},
      {".", {"."}},
      {"chapter", {"chapt", "##e", "##r"}},
      {"about", {"ab", "##o", "##ut"}},
      {"tokenization",
       {"t", "##o", "##k", "##e", "##n", "##i", "##zat", "##i", "##o", "##n"}},
      {"section", {"s", "##e", "##ct", "##i", "##o", "##n"}},
      {"shows", {"sh", "##o", "##w", "##s"}},
      {"several", {"s", "##e", "##v", "##e", "##ra", "##l"}},
      {"tokenizer",
       {"t", "##o", "##k", "##e", "##n", "##i", "##z", "##e", "##r"}},
      {"algorithms", {"a", "##l", "##g", "##o", "##r", "##i", "##thm", "##s"}},
      {"Hopefully", {"H", "##o", "##p", "##e", "##fully"}},
      {",", {","}},
      {"you", {"y", "##o", "##u"}},
      {"will", {"w", "##i", "##l", "##l"}},
      {"be", {"b", "##e"}},
      {"able", {"ab", "##l", "##e"}},
      {"to", {"t", "##o"}},
      {"understand",
       {"u", "##n", "##d", "##e", "##r", "##s", "##ta", "##n", "##d"}},
      {"how", {"h", "##o", "##w"}},
      {"they", {"th", "##e", "##y"}},
      {"are", {"a", "##r", "##e"}},
      {"trained", {"t", "##ra", "##i", "##n", "##e", "##d"}},
      {"and", {"a", "##n", "##d"}},
      {"generate", {"g", "##e", "##n", "##e", "##r", "##at", "##e"}},
      {"tokens", {"t", "##o", "##k", "##e", "##n", "##s"}},
  };
  CHECK(WordPieceTestAccess::splits(tok) == expected);
}
