#include "tokenizers/unigram.hpp"

#include "unicode_norm.hpp"
#include "utf32.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <numeric>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ============== CTOR - DTOR =======================
Unigram::Unigram(TrainConfig config, double percent_to_remove)
    : vocab_size(config.vocab_size), min_freq(config.min_freq),
      p(percent_to_remove) {}

Unigram::~Unigram() = default;
// ============== CTOR - DTOR =======================

// ============== MAIN PIPELINE =======================
void Unigram::feed(const std::string &text) {
  auto utf_str{normalize(text)};
  auto splitted{split(utf_str)};
  for (const auto &word : splitted)
    word_counts[word]++;
}

void Unigram::train() {
  populate_freqs();
  // Single characters are never pruned, so they and the special tokens are the
  // base.
  check_vocab_size(vocab_size, std::ssize(ch_counts) + UNIGRAM_SPECIAL_COUNT);
  while (std::ssize(model) > model_size()) {
    auto scores{compute_scores()};
    // Only single characters are left, nothing can be removed.
    if (scores.empty())
      break;

    std::ranges::sort(scores, [](const auto &a, const auto &b) {
      return a.second < b.second || (a.second == b.second && a.first < b.first);
    });

    // At least one token (or the loop could never end), at most what can be
    // removed, and never below model_size().
    const auto to_remove{static_cast<size_t>(std::clamp<int64_t>(
        static_cast<int64_t>(static_cast<double>(model.size()) * p), 1,
        std::min<int64_t>(std::ssize(scores),
                          std::ssize(model) - model_size())))};
    for (size_t i = 0; i < to_remove; i++) {
      std::erase_if(token_counts, [&scores, i](const auto &item) {
        return item.first == scores.at(i).first;
      });
    }

    auto total_sum{std::accumulate(
        token_counts.begin(), token_counts.end(), int64_t{0},
        [](auto acc, const auto &item) { return acc + item.second; })};

    model.clear();
    for (const auto &[token, freq] : token_counts) {
      model[token] =
          -std::log(static_cast<double>(freq) / static_cast<double>(total_sum));
    }
    update_unk_score();
  }
  assign_ids(vocab_order());
}

std::vector<int64_t> Unigram::encode(const std::string &text) {
  if (token_to_id.empty())
    throw std::logic_error("Unigram has no vocabulary: train or load one");

  std::vector<int64_t> ids{};
  for (const auto &word : split(normalize(text))) {
    // A word with a character the model has never seen comes back as the
    // single token <unk>, which is in token_to_id like any other.
    const auto segmentation{encode_word(word)};
    for (const auto &token : segmentation.first) {
      const auto it{token_to_id.find(token)};
      ids.push_back(it != token_to_id.end() ? it->second : UNIGRAM_UNK_ID);
    }
  }
  return ids;
}

std::string Unigram::decode(const std::vector<int64_t> &ids) const {
  UTF32::CodePoints text{};
  for (const auto id : ids) {
    if (id < 0 || id >= std::ssize(id_to_token))
      throw std::out_of_range("Unknown token id " + std::to_string(id));
    const auto &token{id_to_token[static_cast<size_t>(id)]};
    text.insert(text.end(), token.begin(), token.end());
  }

  // The word-start marker stands for the space that preceded the word.
  std::ranges::replace(text, U'▁', U' ');
  return utf.encode(text);
}
// ============== MAIN PIPELINE =======================

// ============== HELPERS =======================
UTF32::CodePoints Unigram::normalize(const std::string &str) const {
  auto code_points{utf.decode(str)};

  if (std::ranges::all_of(code_points,
                          [](UTF32::CodePoint cp) { return cp < 128; }))
    return code_points;

  return nfc(code_points);
}

std::vector<UTF32::CodePoints>
Unigram::split(const UTF32::CodePoints &code_points) const {
  std::vector<UTF32::CodePoints> words{};
  UTF32::CodePoints word{};

  for (auto cp : code_points) {
    const bool is_ascii{cp < 128};
    if (is_ascii && std::ispunct(static_cast<unsigned char>(cp))) {
      if (!word.empty())
        words.push_back(word);
      words.push_back({cp});

      word = UTF32::CodePoints{};
      continue;
    }

    if (is_ascii && std::isspace(static_cast<unsigned char>(cp))) {
      if (!word.empty())
        words.push_back(word);

      word = UTF32::CodePoints{U'\u2581'};
      continue;
    }

    word.emplace_back(cp);
  }

  if (!word.empty())
    words.push_back(word);

  return words;
}

std::pair<std::vector<UTF32::CodePoints>, std::optional<double>>
Unigram::encode_word(const UTF32::CodePoints &word) const {
  return encode_word(word, model);
}

std::pair<std::vector<UTF32::CodePoints>, std::optional<double>>
Unigram::encode_word(const UTF32::CodePoints &word, const Model &m) const {
  std::vector<std::pair<std::optional<int64_t>, std::optional<double>>>
      best_segmentations{};
  best_segmentations.push_back(
      {std::make_optional<int64_t>(0), std::make_optional<double>(1.0)});

  for (size_t i = 0; i < word.size(); i++) {
    best_segmentations.push_back({std::nullopt, std::nullopt});
  }

  for (size_t idx = 0; idx < word.size(); idx++) {

    const auto starting_score{best_segmentations.at(idx).second};
    if (!starting_score.has_value())
      continue;

    for (size_t end_idx = idx + 1; end_idx < word.size() + 1; end_idx++) {
      UTF32::CodePoints token(word.begin() + idx, word.begin() + end_idx);

      if (m.contains(token)) {
        const double score{*starting_score + m.at(token)};

        if (!best_segmentations.at(end_idx).second.has_value() ||
            best_segmentations.at(end_idx).second.value() > score) {
          best_segmentations.at(end_idx) = {idx, score};
        }
      }
    }
  }

  auto segmentation{best_segmentations.at(best_segmentations.size() - 1)};
  if (!segmentation.second.has_value()) {
    return {{utf.decode(UNIGRAM_UNK_TOKEN)}, std::nullopt};
  }

  auto score{segmentation.second};
  auto start{segmentation.first.value()};
  auto end{word.size()};
  std::vector<UTF32::CodePoints> tokens{};
  UTF32::CodePoints word_slice{};
  while (start != 0) {
    std::copy(word.begin() + start, word.begin() + end,
              std::back_inserter(word_slice));
    tokens.insert(tokens.begin(), word_slice);
    auto next_start{best_segmentations.at(start).first.value()};
    end = start;
    start = next_start;
    word_slice.clear();
  }
  std::copy(word.begin() + start, word.begin() + end,
            std::back_inserter(word_slice));
  tokens.insert(tokens.begin(), word_slice);
  return {tokens, score};
}

void Unigram::populate_freqs() {
  std::erase_if(word_counts, [this](const auto &item) {
    auto const &[_, count] = item;
    return count < min_freq;
  });

  for (const auto &[word, count] : word_counts) {
    for (size_t i = 0; i < word.size(); i++) {
      ch_counts[{word.at(i)}] += count;
      for (size_t j = i + 2; j <= word.size(); j++) {
        UTF32::CodePoints subword{};
        std::copy(word.begin() + i, word.begin() + j,
                  std::back_inserter(subword));
        subword_counts_map[subword] += count;
      }
    }
  }
  subword_counts.reserve(subword_counts_map.size());
  std::copy(subword_counts_map.begin(), subword_counts_map.end(),
            std::back_inserter(subword_counts));
  std::ranges::sort(subword_counts, [](const auto &a, const auto &b) {
    return a.second > b.second || (a.second == b.second && a.first < b.first);
  });

  std::copy(ch_counts.begin(), ch_counts.end(),
            std::back_inserter(token_counts));

  // Slots left after the characters, and how many substrings exist to fill
  // them.
  const int64_t seed_size{model_size() * SEED_FACTOR};
  const int64_t room{std::max<int64_t>(0, seed_size - std::ssize(ch_counts))};
  const int64_t take{std::min<int64_t>(room, std::ssize(subword_counts))};

  std::copy(subword_counts.begin(), subword_counts.begin() + take,
            std::back_inserter(token_counts));

  int64_t total_sum{std::accumulate(
      token_counts.begin(), token_counts.end(), int64_t{0},
      [](int64_t acc, const auto &item) { return acc + item.second; })};

  for (const auto &item : token_counts)
    model[item.first] = -std::log(static_cast<double>(item.second) /
                                  static_cast<double>(total_sum));

  update_unk_score();
}

void Unigram::update_unk_score() {
  // Worse than every real token, so <unk> is only used when nothing else
  // covers a character.
  constexpr double UNK_PENALTY{10.0};
  double worst_score{0.0};
  for (const auto &[token, score] : model)
    worst_score = std::max(worst_score, score);
  unk_score = worst_score + UNK_PENALTY;
}

double Unigram::compute_loss() const { return compute_loss(model); }

double Unigram::compute_loss(const Model &m) const {
  double loss{0};
  for (const auto &[word, freq] : word_counts) {
    const auto [_, word_loss]{encode_word(word, m)};
    loss += static_cast<double>(freq) * word_loss.value_or(unk_score);
  }
  return loss;
}

std::vector<std::pair<UTF32::CodePoints, double>>
Unigram::compute_scores() const {
  std::vector<std::pair<UTF32::CodePoints, double>> scores{};

  const auto model_loss{compute_loss()};
  for (const auto &[token, score] : model) {
    if (token.size() == 1)
      continue;

    Model model_without_token{model};

    model_without_token.erase(token);
    scores.emplace_back(token, compute_loss(model_without_token) - model_loss);
  }

  return scores;
}
// ============== HELPERS =======================

// ============== MISC =======================
void Unigram::save_vocab(const std::string &filename) const {
  std::ofstream vocab_file(filename);
  if (!vocab_file)
    throw std::runtime_error("Cannot open file " + filename);

  // One line per token, in ID order: "token score". 17 digits round-trip a
  // double exactly. The special tokens carry no model score: <|endoftext|>
  // gets 0, and <unk> gets unk_score so that loading restores it.
  vocab_file << std::setprecision(17);
  const auto order{vocab_order()};
  for (size_t i = 0; i < order.size(); i++) {
    const auto id{static_cast<int64_t>(i)};
    double score{0.0};
    if (id == UNIGRAM_UNK_ID)
      score = unk_score;
    else if (id != UNIGRAM_EOT_ID)
      score = model.at(order[i]);

    vocab_file << utf.encode(order[i]) << ' ' << score << '\n';
  }
}

void Unigram::load_vocab(const std::string &filename) {
  std::ifstream vocab_file(filename);
  if (!vocab_file)
    throw std::runtime_error("Cannot open file " + filename);

  Model loaded{};
  double loaded_unk_score{0.0};
  std::vector<UTF32::CodePoints> order{};

  std::string line;
  while (std::getline(vocab_file, line)) {
    // The score is after the last space, so the token itself may hold one.
    const auto pos{line.rfind(' ')};
    if (pos == std::string::npos)
      throw std::runtime_error("Malformed vocab line: " + line);

    auto token{utf.decode(line.substr(0, pos))};
    const double score{std::stod(line.substr(pos + 1))};

    const auto id{static_cast<int64_t>(order.size())};
    if (id == UNIGRAM_UNK_ID)
      loaded_unk_score = score;
    else if (id != UNIGRAM_EOT_ID)
      loaded[token] = score;
    order.push_back(std::move(token));
  }

  if (order.size() < 2 ||
      order[UNIGRAM_EOT_ID] != utf.decode(UNIGRAM_EOT_TOKEN) ||
      order[UNIGRAM_UNK_ID] != utf.decode(UNIGRAM_UNK_TOKEN))
    throw std::runtime_error("Vocab file " + filename +
                             " must start with <|endoftext|> and <unk>");

  model = std::move(loaded);
  unk_score = loaded_unk_score;
  assign_ids(std::move(order));
}

std::vector<UTF32::CodePoints> Unigram::vocab_order() const {
  std::vector<std::pair<UTF32::CodePoints, double>> tokens{model.begin(),
                                                           model.end()};
  std::ranges::sort(tokens, [](const auto &a, const auto &b) {
    return a.second < b.second || (a.second == b.second && a.first < b.first);
  });

  std::vector<UTF32::CodePoints> order{};
  order.reserve(tokens.size() + 2);
  order.push_back(utf.decode(UNIGRAM_EOT_TOKEN));
  order.push_back(utf.decode(UNIGRAM_UNK_TOKEN));
  for (const auto &[token, score] : tokens)
    order.push_back(token);
  return order;
}

void Unigram::assign_ids(std::vector<UTF32::CodePoints> order) {
  id_to_token = std::move(order);
  token_to_id.clear();
  for (size_t i = 0; i < id_to_token.size(); i++)
    token_to_id[id_to_token[i]] = static_cast<int64_t>(i);
}
// ============== MISC =======================
