#include "tokenizers/wordpiece.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <ranges>

#include "../private/frozen_pair.hpp"

// ============== CTOR - DTOR =======================
WordPiece::WordPiece(TrainConfig config)
    : vocab(std::make_unique<FrozenPair<std::string, int64_t>>()),
      vocab_size(config.vocab_size), min_freq(config.min_freq) {}

WordPiece::~WordPiece() = default;
// ============== CTOR - DTOR =======================

// ============== MAIN PIPELINE =======================
void WordPiece::feed(const std::string &str) {
  auto normalized{normalize(str)};
  auto splitted{split(normalized)};
  count_words(splitted);
}

void WordPiece::train() {
  populate_vocab();
  // The special tokens and the alphabet are the base: merges only add to it.
  check_vocab_size(vocab_size, static_cast<int64_t>(vocab->len()));

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
    auto new_token{best_pair.first + (best_pair.second.starts_with("##")
                                          ? best_pair.second.substr(2)
                                          : best_pair.second)};
    auto _ = vocab->insert(new_token, vocab->len());
  }
}

std::vector<int64_t> WordPiece::encode(const std::string &str) {
  std::vector<int64_t> encoded{};

  auto splitted{split(normalize(str))};
  std::vector<int64_t> intermediate{};
  for (const auto &split : splitted) {
    encode_word(split, intermediate);
    encoded.insert(encoded.end(), intermediate.begin(), intermediate.end());
    intermediate.clear();
  }
  return encoded;
}

void WordPiece::encode_word(const std::string &word,
                            std::vector<int64_t> &out) {
  size_t i{word.length()};
  // When having '#' as a token, using 0 as the condition leads to
  // infinite recursion.
  const size_t min_len{word.starts_with("##") ? size_t{2} : size_t{0}};
  while (i > min_len && !vocab->get(word.substr(0, i)).has_value()) {
    i--;
  }
  if (i == min_len) {
    out.push_back(vocab->get("[UNK]").value());
    return;
  }
  out.push_back(vocab->get(word.substr(0, i)).value());
  if (word.substr(i, word.length()).empty()) {
    return;
  }

  std::string leftover{"##" + word.substr(i, word.length())};
  // I think this is how tail-recursion is done.
  return encode_word(leftover, out);
}

std::string WordPiece::decode(const std::vector<int64_t> &encoded) const {
  std::string decoded{""};
  for (auto code : encoded) {
    std::string token{vocab->get(code).value()};
    // Too terse
    // Basicaly we first check if the token is a continuation (starts with ##)
    // or not. If it isn't we check if it's punctuation or not. If it's neither
    // a continuation nor punctutation we add a space before. Otherwise we
    // concatenate it to the sentence as is.
    decoded += token.starts_with("##") ? token.substr(2)
               : (token.length() == 1 &&
                  std::ispunct(static_cast<unsigned char>(token[0])))
                   ? token
                   : (" " + token);
  }

  auto not_space{[](unsigned char c) {
    return !std::isspace(static_cast<unsigned char>(c));
  }};
  decoded.erase(decoded.begin(), std::ranges::find_if(decoded, not_space));
  decoded.erase(
      std::ranges::find_if(decoded | std::views::reverse, not_space).base(),
      decoded.end());
  return decoded;
}

// ============== MAIN PIPELINE =======================

// ============== HELPERS =======================

std::string WordPiece::normalize(std::string str) const {
  // UTF-8 bytes of typographic characters -> ASCII equivalents.
  // Written as escapes so the result doesn't depend on the source file's
  // encoding.
  const std::array<std::pair<std::string, std::string>, 7> replacements = {{
      {"\xE2\x80\x9C", "\""},  // “
      {"\xE2\x80\x9D", "\""},  // ”
      {"\xE2\x80\x98", "'"},   // ‘
      {"\xE2\x80\x99", "'"},   // ’
      {"\xE2\x80\x93", "-"},   // –
      {"\xE2\x80\x94", "-"},   // —
      {"\xE2\x80\xA6", "..."}, // …
  }};

  for (auto &[from, to] : replacements) {
    size_t pos{0};
    while ((pos = str.find(from, pos)) != std::string::npos) {
      str.replace(pos, from.length(), to);
      pos += to.length();
    }
  }
  return str;
}

std::vector<std::string> WordPiece::split(const std::string &str) const {
  std::vector<std::string> words{};
  std::string word{""};

  for (char c : str) {
    if (std::ispunct(static_cast<unsigned char>(c))) {
      words.push_back(word);
      // string(size_type count, CharT ch):
      words.push_back(std::string(1, c));
      word = "";
      continue;
    }

    if (std::isspace(static_cast<unsigned char>(c))) {
      words.push_back(word);
      word = "";
      continue;
    }

    word += c;
  }

  if (!word.empty())
    words.push_back(word);

  // Remove spaces
  // eg If you have 'Hello, hi' it will be splitted as 'Hello', ',', ' ', 'hi'.
  std::erase_if(words, [](const std::string &word) { return word.empty(); });

  return words;
}

void WordPiece::count_words(
    const std::vector<std::string> &words) const noexcept {
  for (const auto &word : words) {
    counts[word]++;
  }
}

void WordPiece::populate_vocab() {
  int64_t counter{0};
  for (const auto &special_token : SPECIAL_TOKENS)
    auto _ = vocab->insert(special_token, counter++);

  std::erase_if(counts, [this](const auto &item) {
    auto const &[_, count] = item;
    return count < min_freq;
  });

  for (auto &[word, _] : counts) {
    std::vector<std::string> splitted{};

    for (size_t i = 0; i < word.length(); i++) {
      // first chars, do not need "##".
      std::string letter = (i == 0 ? "" : "##") + std::string(1, word[i]);
      bool inserted = vocab->insert(letter, counter);
      splitted.push_back(letter);

      if (inserted) {
        alphabet.push_back(letter);
        counter++;
      }
    }

    // HuggingFace does it separately. To avoid recomputing, I do it here.
    splits[word] = splitted;
  }
}

void WordPiece::compute_pair_scores() {
  std::map<std::string, int64_t> letter_freqs{};
  std::map<Pair, double> pair_freqs{};
  for (auto &[word, count] : counts) {
    const auto &split{splits[word]};

    if (split.size() == 1) {
      letter_freqs[split.at(0)] += count;
      continue;
    }

    for (size_t i = 0; i < split.size() - 1; i++) {
      Pair pair{split[i], split[i + 1]};
      letter_freqs[split.at(i)] += count;
      pair_freqs[pair] += count;
    }

    letter_freqs[split.at(split.size() - 1)] += count;
  }

  for (auto &[pair, count] : pair_freqs) {
    pairs[pair] =
        count / (letter_freqs[pair.first] * letter_freqs[pair.second]);
  }
}

void WordPiece::merge_pair(const Pair &pair) {
  std::array<std::string, 2> target{pair.first, pair.second};
  for (auto &[word, _] : counts) {
    auto &split{splits[word]};
    if (split.size() == 1)
      continue;

    auto it{split.begin()};
    while (true) {
      it = std::search(it, split.end(), target.begin(), target.end());
      if (it == split.end())
        break;

      std::string merged =
          pair.first +
          (pair.second.starts_with("##") ? pair.second.substr(2) : pair.second);
      it = split.erase(it, it + 2);
      it = split.insert(it, merged);
      it++;
    }
  }
  pairs.erase(pair);
}
// ============== HELPERS =======================

// ============== MISC =======================

void WordPiece::print_info() const {
  std::cout << "Word Counts\n";
  for (auto &[word, count] : counts)
    std::cout << word << ": " << count << ", ";

  std::cout << "\nAlphabet\n";
  for (auto &str : alphabet)
    std::cout << "'" << str << "'" << ", ";

  std::cout << "\nVocab\n";
  for (auto &[word, id] : *vocab)
    std::cout << "{" << word << ":" << id << "}, ";

  std::cout << "\nSplits\n";
  for (auto &[word, splitted] : splits) {
    std::cout << word << ": ";
    for (auto c : splitted)
      std::cout << c << " ";
    std::cout << "\n";
  }

  std::cout << "\nPair Scores\n";
  for (auto &[pair, count] : pairs) {
    std::cout << "(" << pair.first << ", " << pair.second << ") = " << count
              << "\n";
  }
}

void WordPiece::save_vocab(const std::string &filename) const {
  std::ofstream vocab_file(filename);

  for (auto &[word, count] : *vocab) {
    vocab_file << word + " " + std::to_string(count) + "\n";
  }

  vocab_file.close();
}

void WordPiece::load_vocab(const std::string &filename) {
  std::ifstream vocab_file(filename);
  if (!vocab_file)
    throw std::runtime_error("Cannot open file " + filename);

  std::string line;

  while (std::getline(vocab_file, line)) {
    // We know it only has one space.
    auto pos{line.find(' ')};
    std::string word{line.substr(0, pos)};
    int64_t count{std::stoll(line.substr(pos + 1))};
    auto _ = vocab->insert(word, count);
  }

  vocab_file.close();
}
// ============== MISC =======================