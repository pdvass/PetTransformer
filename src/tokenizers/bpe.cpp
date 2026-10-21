#include "tokenizers/bpe.hpp"

#include "../private/frozen_pair.hpp"
#include "byte_level.hpp"
#include "unicode_norm.hpp"

#include "utf32.hpp"
#include <algorithm>
#include <format>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// ============== CTOR - DTOR =======================
BPE::BPE(TrainConfig config)
    : vocab(std::make_unique<
            FrozenPair<UTF32::CodePoints, int64_t, UTF32::Hasher>>()),
      vocab_size(config.vocab_size), min_freq(config.min_freq) {
  // populate_vocab always adds these, so the base is known before training.
  check_vocab_size(vocab_size,
                   std::ssize(BPE_SPECIAL_TOKENS) +
                       static_cast<int64_t>(ByteLevel::ALPHABET_SIZE));
}

BPE::~BPE() = default;
// ============== CTOR - DTOR =======================

// ============== MAIN PIPELINE =======================
void BPE::feed(const std::string &str) {

  for (const auto &[text, special] : split_special(str)) {
    if (special >= 0)
      continue;
    auto utf_str{normalize(text)};
    auto splitted{split(utf_str)};
    auto symbols{to_symbols(splitted)};
    count_words(symbols);
  }
}

void BPE::train() {
  populate_vocab();

  while (vocab->len() < static_cast<size_t>(vocab_size)) {
    pairs.clear();
    compute_pair_scores();
    if (pairs.empty()) {
      break;
    }

    auto it{std::ranges::max_element(pairs, [](const auto &a, const auto &b) {
      return a.second < b.second || (a.second == b.second && a.first > b.first);
    })};
    Pair best_pair{it->first};

    merge_pair(best_pair);
    const auto rank{static_cast<int64_t>(merge_rank.size())};
    merge_rank[best_pair] = rank;

    UTF32::CodePoints new_token{};
    new_token.insert(new_token.end(), best_pair.first.begin(),
                     best_pair.first.end());
    new_token.insert(new_token.end(), best_pair.second.begin(),
                     best_pair.second.end());

    auto _ = vocab->insert(new_token, vocab->len());
  }
}
std::vector<int64_t> BPE::encode(const std::string &str) {
  std::vector<int64_t> encoded{};
  std::vector<UTF32::CodePoints> intermediate{};

  for (const auto &[text, special] : split_special(str)) {
    // A special token's ID is its position in BPE_SPECIAL_TOKENS, which is
    // how populate_vocab numbers them.
    if (special >= 0) {
      encoded.push_back(special);
      continue;
    }

    auto splitted{to_symbols(split(normalize(text)))};
    for (const auto &split : splitted) {
      encode_word(split, intermediate);
      for (const auto &inter : intermediate)
        encoded.push_back(vocab->get(inter).value());

      intermediate.clear();
    }
  }
  return encoded;
}

void BPE::encode_word(const UTF32::CodePoints &word,
                      std::vector<UTF32::CodePoints> &out) {
  constexpr int64_t NO_RANK = std::numeric_limits<int64_t>::max();

  if (word.empty())
    return;

  std::ranges::transform(
      word, std::back_inserter(out),
      [](UTF32::CodePoint cp) { return UTF32::CodePoints{cp}; });

  std::vector<std::pair<UTF32::CodePoints, UTF32::CodePoints>> word_split{};
  for (size_t i = 0; i < out.size() - 1; i++) {
    word_split.emplace_back(out.at(i), out.at(i + 1));
  }

  // We use a flat vector to iterate over again and again
  // until we find that there are no more merges that can be
  // done. Then, we consider it to have reached the final form
  // and can be returned, so that every element it contains
  // can be converted to its ID.
  while (true) {
    // pattern found
    // https://cppreference.com/cpp/ranges/transform_view
    std::vector<int64_t> ranks{};
    std::ranges::copy(std::views::transform(
                          word_split,
                          [this](const auto &pair) {
                            bool pair_exists{merge_rank.contains(pair)};
                            return pair_exists ? merge_rank.at(pair) : NO_RANK;
                          }),
                      std::back_inserter(ranks));

    if (ranks.empty())
      break;
    auto it{std::ranges::min_element(ranks)};
    if (*it == NO_RANK)
      break;
    const auto idx{static_cast<size_t>(it - ranks.begin())};

    UTF32::CodePoints merged{};
    merged.insert(merged.end(), word_split.at(idx).first.begin(),
                  word_split.at(idx).first.end());
    merged.insert(merged.end(), word_split.at(idx).second.begin(),
                  word_split.at(idx).second.end());

    const auto pos{out.begin() + static_cast<std::ptrdiff_t>(idx)};
    const auto after{out.erase(pos, pos + 2)};
    out.insert(after, merged);

    word_split.clear();
    for (size_t i = 0; i < out.size() - 1; i++) {
      word_split.emplace_back(out.at(i), out.at(i + 1));
    }
  }
}

std::string BPE::decode(const std::vector<int64_t> &encoded) const {
  std::string decoded{""};

  for (auto code : encoded) {
    const auto token{vocab->get(code)};
    if (!token)
      throw std::out_of_range(std::format("Unknown token ID: {}", code));
    decoded += ByteLevel::to_bytes(token.value());
  }

  return decoded;
}
// ============== MAIN PIPELINE =======================

// ============== HELPERS =======================

UTF32::CodePoints BPE::normalize(const std::string &str) const {
  auto code_points{utf.decode(str)};

  // Every ASCII string is already in NFC, and that is nearly all of the
  // text, so it skips the normalization.
  if (std::ranges::all_of(code_points,
                          [](UTF32::CodePoint cp) { return cp < 128; }))
    return code_points;

  return nfc(code_points);
}

std::vector<std::pair<std::string, int64_t>>
BPE::split_special(const std::string &str) const {
  std::vector<std::pair<std::string, int64_t>> parts{};
  size_t start{0};

  while (true) {
    // The special token that starts earliest, from `start` on.
    size_t found{std::string::npos};
    int64_t which{-1};
    for (size_t i = 0; i < BPE_SPECIAL_TOKENS.size(); i++) {
      const auto pos{str.find(BPE_SPECIAL_TOKENS[i], start)};
      if (pos < found) {
        found = pos;
        which = static_cast<int64_t>(i);
      }
    }
    if (which < 0)
      break;

    if (found > start)
      parts.emplace_back(str.substr(start, found - start), -1);
    parts.emplace_back("", which);
    start = found + BPE_SPECIAL_TOKENS[static_cast<size_t>(which)].size();
  }

  if (start < str.size())
    parts.emplace_back(str.substr(start), -1);
  return parts;
}

std::vector<UTF32::CodePoints>
BPE::split(const UTF32::CodePoints &code_points) const {
  std::vector<UTF32::CodePoints> words{};
  UTF32::CodePoints word{};

  for (auto cp : code_points) {
    const bool is_ascii = cp < 128;
    if (is_ascii && std::ispunct(static_cast<unsigned char>(cp))) {
      words.push_back(word);
      words.push_back({cp});

      word = UTF32::CodePoints{};
      continue;
    }

    if (is_ascii && std::isspace(static_cast<unsigned char>(cp))) {
      words.push_back(word);
      word = UTF32::CodePoints{};
      word.emplace_back(cp);
      continue;
    }

    word.emplace_back(cp);
  }

  if (!word.empty())
    words.push_back(word);

  std::erase_if(words,
                [](const UTF32::CodePoints &word) { return word.empty(); });

  return words;
}

void BPE::count_words(
    const std::vector<UTF32::CodePoints> &v_cps) const noexcept {
  for (const auto &cps : v_cps) {
    counts[cps]++;
  }
}

void BPE::populate_vocab() {
  int64_t counter{0};

  for (const auto &special_token : BPE_SPECIAL_TOKENS)
    auto _ = vocab->insert(utf.decode(special_token), counter++);

  for (const auto code_point : ByteLevel::BYTE_TO_CODE_POINT)
    alphabet.push_back(UTF32::CodePoints{code_point});

  std::erase_if(counts, [this](const auto &item) {
    auto const &[_, count] = item;
    return count < min_freq;
  });

  for (auto &[word, _] : counts) {
    std::vector<UTF32::CodePoints> splitted{};

    for (size_t i = 0; i < word.size(); i++) {
      UTF32::CodePoints letter{word.at(i)};
      splitted.push_back(letter);
    }
    splits[word] = splitted;
  }
  std::ranges::sort(alphabet);
  for (const auto &value : alphabet)
    auto _ = vocab->insert(value, counter++);
}

void BPE::compute_pair_scores() {
  for (auto &[word, count] : counts) {
    const auto &split{splits[word]};

    if (split.size() == 1) {
      continue;
    }

    for (size_t i = 0; i < split.size() - 1; i++) {
      Pair pair{split[i], split[i + 1]};
      pairs[pair] += count;
    }
  }
}

void BPE::merge_pair(const Pair &pair) {
  std::array<UTF32::CodePoints, 2> target{pair.first, pair.second};
  for (auto &[word, _] : counts) {
    auto &split{splits[word]};
    if (split.size() == 1)
      continue;

    auto it{split.begin()};
    while (true) {
      it = std::search(it, split.end(), target.begin(), target.end());
      if (it == split.end())
        break;

      UTF32::CodePoints merged{};
      merged.insert(merged.end(), pair.first.begin(), pair.first.end());
      merged.insert(merged.end(), pair.second.begin(), pair.second.end());

      it = split.erase(it, it + 2);
      it = split.insert(it, merged);
      it++;
    }
  }
  pairs.erase(pair);
}

std::vector<UTF32::CodePoints>
BPE::to_symbols(const std::vector<UTF32::CodePoints> &words) const {
  std::vector<UTF32::CodePoints> symbols{};
  for (const auto &word : words)
    symbols.emplace_back(ByteLevel::to_code_points(utf.encode(word)));
  return symbols;
}
// ============== HELPERS =======================

// ============== MISC =======================
void BPE::save_vocab(const std::string &filename) const {
  std::ofstream vocab_file(filename);

  std::vector<std::pair<std::string, int64_t>> sorted_vocab{};
  for (auto &[word, id] : *vocab)
    sorted_vocab.emplace_back(utf.encode(word), id);

  std::ranges::sort(sorted_vocab, {}, &std::pair<std::string, int64_t>::second);
  for (auto &[word, count] : sorted_vocab)
    vocab_file << word + " " + std::to_string(count) + "\n";

  vocab_file << "##Ranks\n";
  std::vector<std::pair<Pair, int64_t>> ordered{merge_rank.begin(),
                                                merge_rank.end()};
  std::ranges::sort(ordered, [](const auto &a, const auto &b) {
    return a.second < b.second;
  });
  for (const auto &[pair, rank] : ordered)
    vocab_file << utf.encode(pair.first) << " " << utf.encode(pair.second)
               << "\n";
}

void BPE::load_vocab(const std::string &filename) {
  std::ifstream vocab_file(filename);
  if (!vocab_file)
    throw std::runtime_error("Cannot open file " + filename);

  std::string line;

  bool ranks = false;
  int64_t rank{0};
  while (std::getline(vocab_file, line)) {
    if (line == "##Ranks") {
      ranks = true;
      continue;
    }

    if (!ranks) {
      auto pos{line.find(' ')};
      std::string word{line.substr(0, pos)};
      int64_t count{std::stoll(line.substr(pos + 1))};
      auto _ = vocab->insert(utf.decode(word), count);
      continue;
    }

    auto word_pos{line.find(' ')};

    std::string word1{line.substr(0, word_pos)};
    std::string word2{line.substr(word_pos + 1)};

    Pair pair{utf.decode(word1), utf.decode(word2)};
    merge_rank[pair] = rank++;
  }
}

void BPE::print_info() const {
  std::cout << "\nCounts\n";
  for (const auto &[cps, count] : counts) {
    std::cout << utf.encode(cps) << ' ' << count << '\n';
  }

  std::cout << "\nAlphabet\n";
  for (auto &str : alphabet)
    std::cout << "'" << utf.encode(str) << "'" << ", ";

  std::cout << "\nVocab\n";
  std::vector<std::pair<std::string, int64_t>> sorted_vocab{};
  for (const auto &[word, id] : *vocab)
    sorted_vocab.emplace_back(utf.encode(word), id);

  std::ranges::sort(sorted_vocab, {}, &std::pair<std::string, int64_t>::second);
  for (const auto &[word, id] : sorted_vocab)
    std::cout << "{" << word << ":" << id << "}, ";

  std::cout << "\nSplits\n";
  for (const auto &[word, splitted] : splits) {
    std::cout << utf.encode(word) << ": ";
    for (const auto &c : splitted)
      std::cout << utf.encode(c) << " ";
    std::cout << "\n";
  }

  std::cout << "\nMerge Rank\n";
  std::vector<std::pair<Pair, int64_t>> ordered{merge_rank.begin(),
                                                merge_rank.end()};
  std::ranges::sort(ordered, [](const auto &a, const auto &b) {
    return a.second < b.second;
  });
  for (const auto &[pair, rank] : ordered)
    std::cout << rank << ": (" << utf.encode(pair.first) << ", "
              << utf.encode(pair.second) << ")\n";

  std::cout << "\nPair Scores\n";
  for (const auto &[pair, count] : pairs) {
    std::cout << "(" << utf.encode(pair.first) << ", "
              << utf.encode(pair.second) << ") = " << count << "\n";
  }
}
// ============== MISC =======================