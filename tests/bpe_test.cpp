#include "bpe.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "byte_level.hpp"
#include "doctest.h"
#include "frozen_pair.hpp"
#include "unicode_norm.hpp"

struct BPETestAccess {
  static const auto &vocab(const BPE &b) { return *b.vocab; }
  static const auto &alphabet(const BPE &b) { return b.alphabet; }
  static const auto &counts(const BPE &b) { return b.counts; }
  static const auto &splits(const BPE &b) { return b.splits; }
  static const auto &ranks(const BPE &b) { return b.merge_rank; }
};

namespace {

const UTF32 utf{};

using Merge = std::pair<std::string, std::string>;
using Symbols = std::vector<UTF32::CodePoints>;

struct Corpus {
  std::string name;
  std::vector<std::string> lines;
};

// The GPT-2 space marker, U+0120. Written as bytes so the test doesn't depend
// on the source file's encoding.
std::string g(const std::string &word = "") { return "\xC4\xA0" + word; }

// The text as the tokenizer sees it: the same UTF-8 in NFC. A round trip gives
// this back, which is the input itself whenever the input was already in NFC.
std::string normalized(const std::string &text) {
  return utf.encode(nfc(utf.decode(text)));
}

// <|endoftext|> and one symbol for each of the 256 bytes, whatever the corpus.
int64_t base_size() {
  return static_cast<int64_t>(BPE_SPECIAL_TOKENS.size() +
                              ByteLevel::ALPHABET_SIZE);
}

const std::vector<std::string> HF_CORPUS = {
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and "
    "generate tokens.",
};

// Corpora chosen for their edge cases: nothing, one symbol, one repeated
// symbol, non-ASCII text, whitespace of every kind, and a mix.
const std::vector<Corpus> &corpora() {
  static const std::vector<Corpus> all = {
      {"hf", HF_CORPUS},
      {"empty", {}},
      {"single", {"a"}},
      {"repeated", {"aaaaaaaa aaaa", "aaaa"}},
      {"non-ascii",
       {"caf\xC3\xA9 caf\xC3\xA9 caf\xC3\xA9",
        "\xE2\x80\x9Cquoted\xE2\x80\x9D \xE2\x80\x94 dash",
        "\xE4\xBD\xA0\xE5\xA5\xBD \xE4\xBD\xA0\xE5\xA5\xBD",
        "\xF0\x9F\x98\x80 \xF0\x9F\x98\x80 emoji"}},
      {"whitespace", {"a  b\tc\nd   ", "  lead", "trail  ", "a\r\nb"}},
      {"mixed",
       {"This is the \xC3\xA9t\xC3\xA9.\nThis is the end.",
        "tab\tseparated\tvalues\tvalues", "  double  spaced  text  "}},
      // The same letters spelled both ways: composed and decomposed.
      {"decomposed",
       {"caf\xC3\xA9 cafe\xCC\x81 caf\xC3\xA9", "\xCE\xAC \xCE\xB1\xCC\x81",
        "\xE1\x84\x92\xE1\x85\xA1\xE1\x86\xAB \xED\x95\x9C"}},
      // As in TinyStories: documents separated by the special token, on its
      // own line or inside a line.
      {"special",
       {"Once upon a time.", "<|endoftext|>", "The end<|endoftext|>Next story",
        "<|endoftext|><|endoftext|>"}},
  };
  return all;
}

// Sizes around the interesting boundaries: exactly the base (no merges), one
// merge, and larger than any of the corpora can use. Below the base is refused.
const std::vector<int64_t> VOCAB_SIZES = {257, 258, 300, 1000, 4096};

std::unique_ptr<BPE> train(int64_t vocab_size,
                           const std::vector<std::string> &lines,
                           int64_t min_freq = 0) {
  auto tok{std::make_unique<BPE>(
      TrainConfig{.vocab_size = vocab_size, .min_freq = min_freq})};
  for (const auto &line : lines)
    tok->feed(line);
  tok->train();
  return tok;
}

// Merge rules as strings, ordered by rank.
std::vector<Merge> merges(const BPE &tok) {
  const auto &ranks{BPETestAccess::ranks(tok)};
  std::vector<Merge> ordered(ranks.size());
  for (const auto &[pair, rank] : ranks)
    ordered.at(static_cast<size_t>(rank)) = {utf.encode(pair.first),
                                             utf.encode(pair.second)};
  return ordered;
}

std::string token_of(const BPE &tok, int64_t id) {
  return utf.encode(BPETestAccess::vocab(tok).get(id).value());
}

std::vector<std::string> tokens(BPE &tok, const std::string &text) {
  std::vector<std::string> out{};
  for (const auto id : tok.encode(text))
    out.push_back(token_of(tok, id));
  return out;
}

int64_t encoded_length(BPE &tok, const std::vector<std::string> &lines) {
  int64_t total{0};
  for (const auto &line : lines)
    total += static_cast<int64_t>(tok.encode(line).size());
  return total;
}

// The sorted symbols the alphabet must consist of, as UTF-8 strings.
std::vector<std::string> byte_alphabet() {
  std::vector<char32_t> sorted{ByteLevel::BYTE_TO_CODE_POINT.begin(),
                               ByteLevel::BYTE_TO_CODE_POINT.end()};
  std::ranges::sort(sorted);
  std::vector<std::string> out{};
  for (const auto code_point : sorted)
    out.push_back(utf.encode(UTF32::CodePoints{code_point}));
  return out;
}

// Plain, slow BPE written for clarity only. It takes the words (as symbols) the
// tokenizer counted, and follows the rules in the HF course with this
// project's tie-break: highest count first, then the smaller pair.
std::vector<Merge> reference_merges(const BPE &trained, int64_t vocab_size) {
  std::vector<std::pair<Symbols, int64_t>> words{};
  for (const auto &[word, count] : BPETestAccess::counts(trained)) {
    Symbols symbols{};
    for (const auto code_point : word)
      symbols.push_back(UTF32::CodePoints{code_point});
    words.emplace_back(std::move(symbols), count);
  }

  std::vector<Merge> out{};
  std::set<UTF32::CodePoints> new_tokens{};
  int64_t size{base_size()};

  while (size < vocab_size) {
    std::map<std::pair<UTF32::CodePoints, UTF32::CodePoints>, int64_t> pairs{};
    for (const auto &[symbols, count] : words)
      for (size_t i = 0; i + 1 < symbols.size(); i++)
        pairs[{symbols[i], symbols[i + 1]}] += count;
    if (pairs.empty())
      break;

    // std::map iterates in key order, so keeping only strictly greater counts
    // leaves the smallest pair among equals.
    auto best{pairs.begin()};
    for (auto it = pairs.begin(); it != pairs.end(); ++it)
      if (it->second > best->second)
        best = it;

    const auto &[left, right] = best->first;
    UTF32::CodePoints merged{left};
    merged.insert(merged.end(), right.begin(), right.end());

    for (auto &[symbols, count] : words) {
      Symbols next{};
      for (size_t i = 0; i < symbols.size();) {
        if (i + 1 < symbols.size() && symbols[i] == left &&
            symbols[i + 1] == right) {
          next.push_back(merged);
          i += 2;
        } else {
          next.push_back(symbols[i]);
          i++;
        }
      }
      symbols = std::move(next);
    }

    out.emplace_back(utf.encode(left), utf.encode(right));
    // A merge whose result is already a token doesn't grow the vocabulary.
    if (new_tokens.insert(merged).second)
      size++;
  }
  return out;
}

// Valid UTF-8 built from code points taken from several scripts, so the round
// trip is checked on text the tokenizer has never seen.
std::string random_text(std::mt19937 &rng) {
  static const std::array<std::pair<char32_t, char32_t>, 6> ranges = {{
      {0x20, 0x7E},       // printable ASCII
      {0x09, 0x0D},       // \t \n \v \f \r
      {0xA1, 0xFF},       // Latin-1
      {0x300, 0x36F},     // combining marks
      {0x4E00, 0x9FFF},   // CJK
      {0x1F600, 0x1F64F}, // emoji
  }};

  UTF32::CodePoints code_points{};
  const auto length{rng() % 41};
  for (size_t i = 0; i < length; i++) {
    const auto &[low, high] = ranges[rng() % ranges.size()];
    code_points.push_back(low + static_cast<char32_t>(rng() % (high - low + 1)));
  }
  return utf.encode(code_points);
}

// Everything that has to be true for a trained BPE, for any corpus and any
// requested vocabulary size.
void check_invariants(BPE &tok, int64_t vocab_size,
                      const std::vector<std::string> &lines) {
  const auto &vocab{BPETestAccess::vocab(tok)};
  const auto len{static_cast<int64_t>(vocab.len())};
  const auto base{base_size()};

  // The vocabulary never shrinks below its base, and never goes over the
  // request unless the request was below the base.
  CHECK(len >= base);
  if (vocab_size > base)
    CHECK(len <= vocab_size);
  else
    CHECK(len == base);

  // Stopping short of the request means every word was already merged.
  if (len < vocab_size)
    for (const auto &[word, split] : BPETestAccess::splits(tok))
      CHECK(split.size() == 1);

  // Special token first, then the 256 byte symbols in sorted order.
  CHECK(token_of(tok, 0) == "<|endoftext|>");
  const auto alphabet{byte_alphabet()};
  for (size_t i = 0; i < alphabet.size(); i++)
    CHECK(token_of(tok, static_cast<int64_t>(i) + 1) == alphabet[i]);

  // IDs are dense: 0 .. len-1 resolve, len doesn't.
  for (int64_t id = 0; id < len; id++)
    CHECK(vocab.get(id).has_value());
  CHECK_FALSE(vocab.get(len).has_value());

  // Every learned merge produced a token that is in the vocabulary.
  const auto rules{merges(tok)};
  for (const auto &[left, right] : rules)
    CHECK(vocab.get(utf.decode(left + right)).has_value());

  // Ranks are exactly 0 .. n-1.
  const auto &ranks{BPETestAccess::ranks(tok)};
  std::set<int64_t> seen{};
  for (const auto &[pair, rank] : ranks)
    seen.insert(rank);
  CHECK(seen.size() == ranks.size());
  if (!seen.empty()) {
    CHECK(*seen.begin() == 0);
    CHECK(*seen.rbegin() == static_cast<int64_t>(ranks.size()) - 1);
  }

  // Encoding stays inside the vocabulary, and decoding gives the text back.
  for (const auto &line : lines) {
    const auto ids{tok.encode(line)};
    for (const auto id : ids) {
      CHECK(id >= 0);
      CHECK(id < len);
    }
    CHECK(tok.decode(ids) == normalized(line));
  }
}

} // namespace

// ----------------------------------------------------------------------------
// The HF course corpus, chapter 6.5. The tokenizer is trained with a size far
// above what the corpus can use, so it learns every merge that exists.
// https://huggingface.co/docs/course/chapter6/5
// ----------------------------------------------------------------------------

struct TrainedBPE {
  std::unique_ptr<BPE> owner{train(4096, HF_CORPUS)};
  BPE &tok{*owner};
};

TEST_CASE_FIXTURE(TrainedBPE, "word frequencies match HF reference") {
  const std::map<std::string, int64_t> expected = {
      {"This", 3},            {g("is"), 2},         {g("the"), 1},
      {g("Hugging"), 1},      {g("Face"), 1},       {g("Course"), 1},
      {".", 4},               {g("chapter"), 1},    {g("about"), 1},
      {g("tokenization"), 1}, {g("section"), 1},    {g("shows"), 1},
      {g("several"), 1},      {g("tokenizer"), 1},  {g("algorithms"), 1},
      {"Hopefully", 1},       {",", 1},             {g("you"), 1},
      {g("will"), 1},         {g("be"), 1},         {g("able"), 1},
      {g("to"), 1},           {g("understand"), 1}, {g("how"), 1},
      {g("they"), 1},         {g("are"), 1},        {g("trained"), 1},
      {g("and"), 1},          {g("generate"), 1},   {g("tokens"), 1},
  };

  std::map<std::string, int64_t> actual{};
  for (const auto &[word, count] : BPETestAccess::counts(tok))
    actual[utf.encode(word)] = count;

  CHECK(actual == expected);
}

TEST_CASE_FIXTURE(TrainedBPE, "alphabet is the 256 byte symbols, sorted") {
  std::vector<std::string> actual{};
  for (const auto &symbol : BPETestAccess::alphabet(tok))
    actual.push_back(utf.encode(symbol));

  CHECK(actual.size() == 256);
  CHECK(actual == byte_alphabet());
}

TEST_CASE_FIXTURE(TrainedBPE, "first merge is the same as HF's") {
  // ('Ġ', 't') appears 7 times, the most frequent pair, so there is no tie.
  REQUIRE(!merges(tok).empty());
  CHECK(merges(tok).front() == Merge{g(), "t"});
}

// Snapshot of this implementation's own output, NOT the HF reference.
// Ties between equal pair counts are broken by the smaller pair (code point
// order), HF takes the first pair it saw. E.g. ('e', 'r') and ('i', 's') both
// occur 5 times: HF learns 'is' first, this implementation learns 'er' first.
// Training is greedy, so these are also the first rules of any larger run.
TEST_CASE_FIXTURE(TrainedBPE, "first merge rules snapshot") {
  const std::vector<Merge> expected = {
      {g(), "t"},    {"e", "r"},  {"i", "s"},   {g(), "a"},   {"e", "n"},
      {g("t"), "o"}, {"T", "h"},  {"Th", "is"}, {"k", "en"},  {"n", "d"},
      {"o", "u"},    {"s", "e"},  {g("to"), "ken"}, {"a", "t"}, {"h", "e"},
      {"h", "o"},    {"ho", "w"}, {"i", "n"},   {"i", "o"},
  };

  const auto rules{merges(tok)};
  REQUIRE(rules.size() >= expected.size());
  CHECK(std::equal(expected.begin(), expected.end(), rules.begin()));
}

TEST_CASE_FIXTURE(TrainedBPE, "encode applies merges by rank") {
  // (i, s) has rank 2 and (T, h) rank 6, so 'is' forms first, then (T, h),
  // then (Th, is) gives the whole word.
  CHECK(tokens(tok, "This") == std::vector<std::string>{"This"});
  // 'u' never merges with anything here, so 'Thus' stops at Th.
  CHECK(tokens(tok, "Thus") == std::vector<std::string>{"Th", "u", "s"});
  // Single symbol: no pairs at all.
  CHECK(tokens(tok, "a") == std::vector<std::string>{"a"});
}

TEST_CASE_FIXTURE(TrainedBPE, "encode sentence snapshot") {
  // The corpus's own words are single tokens once training has run out of
  // pairs.
  const std::vector<std::string> expected = {
      "This", g("is"), g("the"), g("Hugging"), g("Face"), g("Course"), ".",
  };
  CHECK(tokens(tok, "This is the Hugging Face Course.") == expected);
}

TEST_CASE_FIXTURE(TrainedBPE, "whitespace keeps its own symbol") {
  CHECK(tokens(tok, " This is") ==
        std::vector<std::string>{g(), "This", g("is")});
  // newline is U+010A, 'Ċ'
  CHECK(tokens(tok, "a\nb") ==
        std::vector<std::string>{"a", "\xC4\x8A", "b"});
  // Spaces are not collapsed.
  CHECK(tokens(tok, "a  b") == std::vector<std::string>{"a", g(), g(), "b"});
}

TEST_CASE_FIXTURE(TrainedBPE, "letters never seen in training still encode") {
  // 'q' and 'Q' are not in the corpus. Every byte is in the alphabet, so
  // nothing is unknown.
  CHECK(tokens(tok, "q") == std::vector<std::string>{"q"});
  CHECK(tokens(tok, "Quiz") == std::vector<std::string>{"Q", "u", "iz"});
}

TEST_CASE_FIXTURE(TrainedBPE, "<|endoftext|> is the single token with ID 0") {
  const std::string eot{"<|endoftext|>"};

  CHECK(tok.encode(eot) == std::vector<int64_t>{0});
  CHECK(tok.decode({0}) == eot);

  // Next to other text it is one ID, and the text around it is encoded as it
  // would be on its own.
  auto expected{tok.encode("This")};
  expected.push_back(0);
  for (const auto id : tok.encode(" is"))
    expected.push_back(id);
  CHECK(tok.encode("This" + eot + " is") == expected);
  CHECK(tok.decode(expected) == "This" + eot + " is");

  // At either end, and repeated.
  CHECK(tok.encode(eot + eot) == std::vector<int64_t>{0, 0});
  CHECK(tok.encode(eot + "a").front() == 0);
  CHECK(tok.encode("a" + eot).back() == 0);
}

TEST_CASE("a vocab size below the base is refused, the base itself is not") {
  CHECK(base_size() == 257);
  CHECK_THROWS_AS(BPE{{.vocab_size = base_size() - 1}}, std::invalid_argument);
  CHECK_THROWS_AS(BPE{{.vocab_size = 1}}, std::invalid_argument);
  CHECK_NOTHROW(BPE{{.vocab_size = base_size()}});
}

TEST_CASE("the vocabulary has exactly vocab_size entries when there is enough "
          "to merge") {
  // HF_CORPUS has more than 43 possible merges.
  auto tok{train(300, HF_CORPUS)};
  CHECK(BPETestAccess::vocab(*tok).len() == 300);
}

TEST_CASE("special tokens are boundaries, not words, when feeding") {
  BPE tok{{.vocab_size = 4096}};
  tok.feed("This is<|endoftext|> the end");

  std::map<std::string, int64_t> actual{};
  for (const auto &[word, count] : BPETestAccess::counts(tok))
    actual[utf.encode(word)] = count;

  // No 'endoftext', no '<' or '|': only the words around it.
  const std::map<std::string, int64_t> expected = {
      {"This", 1}, {g("is"), 1}, {g("the"), 1}, {g("end"), 1}};
  CHECK(actual == expected);
}

TEST_CASE_FIXTURE(TrainedBPE, "a multi-byte character is several symbols") {
  CHECK(tok.encode("\xC3\xA9").size() == 2);   // U+00E9 is C3 A9
  CHECK(tok.encode("\xE2\x80\x9C").size() == 3); // U+201C is E2 80 9C
}

// ----------------------------------------------------------------------------
// NFC: spellings of the same letter are one text for the tokenizer.
// ----------------------------------------------------------------------------

TEST_CASE_FIXTURE(TrainedBPE, "equivalent spellings encode to the same ids") {
  const std::vector<std::pair<std::string, std::string>> equivalents = {
      {"caf\xC3\xA9", "cafe\xCC\x81"},             // e-acute, composed / decomposed
      {"\xCE\xAC", "\xCE\xB1\xCC\x81"},            // Greek alpha with tonos
      {"\xC3\x85", "A\xCC\x8A"},                   // A with ring above
      {"\xE2\x84\xAB", "\xC3\x85"},                // Angstrom sign -> A with ring
      {"\xED\x95\x9C", "\xE1\x84\x92\xE1\x85\xA1\xE1\x86\xAB"}, // Hangul
  };
  for (const auto &[left, right] : equivalents)
    CHECK(tok.encode(left) == tok.encode(right));
}

TEST_CASE_FIXTURE(TrainedBPE, "a decomposed letter decodes as its composed form") {
  CHECK(tok.decode(tok.encode("e\xCC\x81")) == "\xC3\xA9");
  CHECK(tok.decode(tok.encode("\xCE\xB1\xCC\x81")) == "\xCE\xAC");
}

TEST_CASE("both spellings count as the same word when feeding") {
  BPE tok{{.vocab_size = 4096}};
  // Words: "cafe" at the start (no space), then " caf\xC3\xA9" twice, once
  // decomposed and once composed.
  tok.feed("caf\xC3\xA9 cafe\xCC\x81 caf\xC3\xA9");

  std::vector<int64_t> counts{};
  for (const auto &[word, count] : BPETestAccess::counts(tok))
    counts.push_back(count);
  std::ranges::sort(counts);

  // Without NFC the two spellings would be three words seen once each.
  CHECK(counts == std::vector<int64_t>{1, 2});
}

TEST_CASE_FIXTURE(TrainedBPE, "ASCII text is left as it is") {
  // The fast path for ASCII must give what NFC gives.
  for (const auto &sentence : HF_CORPUS)
    CHECK(normalized(sentence) == sentence);
}

TEST_CASE_FIXTURE(TrainedBPE, "encoding what was decoded gives the same ids") {
  for (const std::string text :
       {"cafe\xCC\x81", "\xCE\xB1\xCC\x81 \xCE\xAC", "This is the end"}) {
    const auto ids{tok.encode(text)};
    CHECK(tok.encode(tok.decode(ids)) == ids);
  }
}

// ----------------------------------------------------------------------------
// Round trips: decode(encode(s)) is s, in NFC. For text that is already in NFC
// that is s itself, byte for byte.
// ----------------------------------------------------------------------------

TEST_CASE_FIXTURE(TrainedBPE, "decode(encode(s)) round-trips the corpus") {
  for (const auto &sentence : HF_CORPUS)
    CHECK(tok.decode(tok.encode(sentence)) == sentence);
}

TEST_CASE_FIXTURE(TrainedBPE, "round trip is exact on text it never saw") {
  // All of these are already in NFC, so the round trip is the identity.
  const std::vector<std::string> samples = {
      "",
      " ",
      "   ",
      " leading",
      "trailing ",
      "double  space",
      "tab\there",
      "line\nbreak",
      "carriage\r\nreturn",
      "caf\xC3\xA9",
      "\xE2\x80\x9Cquoted\xE2\x80\x9D \xE2\x80\x94 dash",  // curly quotes, dash
      "\xE2\x80\xA6",                                       // ellipsis
      "\xF0\x9F\x98\x80 emoji",
      "\xE4\xBD\xA0\xE5\xA5\xBD",
      "q Q xyz 0123456789 !?;:()[]{}",
      std::string("embedded\0nul", 12),
  };
  for (const auto &sample : samples) {
    REQUIRE(normalized(sample) == sample);
    CHECK(tok.decode(tok.encode(sample)) == sample);
  }
}

TEST_CASE_FIXTURE(TrainedBPE, "round trip gives NFC on random text") {
  // The generator includes combining marks, so many samples change under NFC.
  std::mt19937 rng{20260101};
  int changed{0};
  for (int i = 0; i < 300; i++) {
    const auto text{random_text(rng)};
    INFO("sample " << i);
    CHECK(tok.decode(tok.encode(text)) == normalized(text));
    if (normalized(text) != text)
      changed++;
  }
  // Guard against the generator going quiet: NFC must have had work to do.
  CHECK(changed > 0);
}

TEST_CASE_FIXTURE(TrainedBPE, "decode rejects ids that are not in the vocab") {
  const auto len{static_cast<int64_t>(BPETestAccess::vocab(tok).len())};
  CHECK(tok.decode({}).empty());
  CHECK_THROWS_AS((void)tok.decode({len}), std::out_of_range);
  CHECK_THROWS_AS((void)tok.decode({-1}), std::out_of_range);
  // The valid ids before it don't leak out as a partial result.
  CHECK_THROWS_AS((void)tok.decode({0, 1, len}), std::out_of_range);
}

// ----------------------------------------------------------------------------
// Properties: any corpus, any requested size.
// ----------------------------------------------------------------------------

TEST_CASE("invariants hold for any corpus and any vocab size") {
  for (const auto &corpus : corpora()) {
    for (const auto vocab_size : VOCAB_SIZES) {
      INFO("corpus=" << corpus.name << " vocab_size=" << vocab_size);
      auto tok{train(vocab_size, corpus.lines)};
      check_invariants(*tok, vocab_size, corpus.lines);
    }
  }
}

TEST_CASE("merges match a plain reference implementation") {
  for (const auto &corpus : corpora()) {
    for (const auto vocab_size : VOCAB_SIZES) {
      INFO("corpus=" << corpus.name << " vocab_size=" << vocab_size);
      auto tok{train(vocab_size, corpus.lines)};
      CHECK(merges(*tok) == reference_merges(*tok, vocab_size));
    }
  }
}

TEST_CASE("min_freq drops rare words, but not their letters") {
  // Only 'This' (3), 'Ġis' (2) and '.' (4) have a count of at least 2.
  auto tok{train(4096, HF_CORPUS, 2)};

  CHECK(BPETestAccess::counts(*tok).size() == 3);
  // ('i','s') = 3 + 2 = 5 is the most frequent pair.
  const auto rules{merges(*tok)};
  REQUIRE(!rules.empty());
  CHECK(rules.front() == Merge{"i", "s"});

  // The alphabet is the 256 bytes, so letters of the dropped words still work.
  CHECK(BPETestAccess::alphabet(*tok).size() == 256);
  const std::string rare{"tokenizer zebra"};
  CHECK(tok->decode(tok->encode(rare)) == rare);

  check_invariants(*tok, 4096, HF_CORPUS);
}

TEST_CASE("training is deterministic") {
  for (const auto &corpus : corpora()) {
    INFO("corpus=" << corpus.name);
    auto first{train(4096, corpus.lines)};
    auto second{train(4096, corpus.lines)};
    CHECK(merges(*first) == merges(*second));
  }
}

TEST_CASE("a larger vocab size only adds merges at the end") {
  const std::vector<int64_t> sizes = {257, 270, 300, 400, 4096};
  for (const auto &corpus : corpora()) {
    INFO("corpus=" << corpus.name);

    std::vector<Merge> previous_merges{};
    int64_t previous_length{std::numeric_limits<int64_t>::max()};

    for (const auto size : sizes) {
      auto tok{train(size, corpus.lines)};
      const auto rules{merges(*tok)};

      // The smaller run's rules are the start of the larger run's.
      REQUIRE(rules.size() >= previous_merges.size());
      CHECK(std::equal(previous_merges.begin(), previous_merges.end(),
                       rules.begin()));

      // More merges never make the encoded corpus longer.
      const auto length{encoded_length(*tok, corpus.lines)};
      CHECK(length <= previous_length);

      previous_merges = rules;
      previous_length = length;
    }
  }
}

// ----------------------------------------------------------------------------
// Saving and loading.
// ----------------------------------------------------------------------------

TEST_CASE("save_vocab/load_vocab keep vocab, ranks and encoding") {
  // The mixed corpus has newlines, tabs and non-ASCII text, so the file has to
  // hold tokens like 'Ċ' and multi-byte symbols.
  const auto &lines{corpora().back().lines};
  const auto path{
      (std::filesystem::temp_directory_path() / "bpe_test_vocab.txt").string()};

  for (const auto vocab_size : {257, 300, 4096}) {
    INFO("vocab_size=" << vocab_size);
    auto tok{train(vocab_size, lines)};
    tok->save_vocab(path);

    BPE loaded{{.vocab_size = vocab_size}};
    loaded.load_vocab(path);
    std::filesystem::remove(path);

    const auto &original{BPETestAccess::vocab(*tok)};
    const auto &restored{BPETestAccess::vocab(loaded)};

    CHECK(restored.len() == original.len());
    for (const auto &[word, id] : original) {
      CHECK(restored.get(word) == id);
      CHECK(restored.get(id) == word);
    }

    CHECK(merges(loaded) == merges(*tok));

    for (const auto &line : lines) {
      CHECK(loaded.encode(line) == tok->encode(line));
      CHECK(loaded.decode(loaded.encode(line)) == line);
    }
  }
}
