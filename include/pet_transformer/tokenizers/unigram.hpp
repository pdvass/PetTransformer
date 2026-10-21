#ifndef UNIGRAM_HPP
#define UNIGRAM_HPP

#include "tokenizer.hpp"
#include "utf32.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

const std::string UNIGRAM_EOT_TOKEN{"<|endoftext|>"};
const std::string UNIGRAM_UNK_TOKEN{"<unk>"};
constexpr int64_t UNIGRAM_EOT_ID{0};
constexpr int64_t UNIGRAM_UNK_ID{1};
// <|endoftext|> and <unk>: in the vocabulary, but not in the model.
constexpr int64_t UNIGRAM_SPECIAL_COUNT{2};

/**
 * Based on: https://huggingface.co/docs/course/chapter6/7
 *
 */
class Unigram : public Tokenizer {
private:
  friend struct UnigramTestAccess;
  using Model = std::unordered_map<UTF32::CodePoints, double, UTF32::Hasher>;

  std::unordered_map<UTF32::CodePoints, int64_t, UTF32::Hasher> word_counts;
  std::unordered_map<UTF32::CodePoints, int64_t, UTF32::Hasher>
      subword_counts_map;
  std::vector<std::pair<UTF32::CodePoints, int64_t>> subword_counts;
  std::unordered_map<UTF32::CodePoints, int64_t, UTF32::Hasher> ch_counts;
  std::vector<std::pair<UTF32::CodePoints, int64_t>> token_counts;
  std::unordered_map<UTF32::CodePoints, int64_t, UTF32::Hasher>
      token_counts_map;

  Model model;
  // Score of <unk>: the largest score in `model` plus a penalty.
  double unk_score{0.0};

  [[nodiscard]] UTF32::CodePoints normalize(const std::string &) const;
  [[nodiscard]] std::vector<UTF32::CodePoints>
  split(const UTF32::CodePoints &) const;

  [[nodiscard]] std::pair<std::vector<UTF32::CodePoints>, std::optional<double>>
  encode_word(const UTF32::CodePoints &) const;
  [[nodiscard]] std::pair<std::vector<UTF32::CodePoints>, std::optional<double>>
  encode_word(const UTF32::CodePoints &, const Model &) const;

  [[nodiscard]] double compute_loss() const;
  [[nodiscard]] double compute_loss(const Model &) const;
  std::vector<std::pair<UTF32::CodePoints, double>> compute_scores() const;

  UTF32 utf;
  const int64_t vocab_size;
  const int64_t min_freq;
  const double p;

  // The model starts with model_size() * SEED_FACTOR tokens (characters plus
  // the most frequent substrings) and is pruned down to model_size(). The HF
  // course starts at 300 and prunes to 100; real data wants a larger factor.
  static constexpr int64_t SEED_FACTOR{3};

  // Tokens the model keeps: the vocabulary without the special tokens.
  [[nodiscard]] int64_t model_size() const {
    return vocab_size - UNIGRAM_SPECIAL_COUNT;
  }

  void populate_freqs();
  void update_unk_score();

  [[nodiscard]] std::vector<UTF32::CodePoints> vocab_order() const;
  void assign_ids(std::vector<UTF32::CodePoints> order);

  std::vector<UTF32::CodePoints> id_to_token;
  std::unordered_map<UTF32::CodePoints, int64_t, UTF32::Hasher> token_to_id;

public:
  void feed(const std::string &) override;
  // Throws std::invalid_argument if vocab_size is below the special tokens
  // plus the characters found in the fed text.
  void train() override;
  std::vector<int64_t> encode(const std::string &) override;
  [[nodiscard]] std::string decode(const std::vector<int64_t> &) const override;

  explicit Unigram(TrainConfig config, double percent_to_remove = 0.1);
  void
  save_vocab(const std::string &filename = "unigram_vocab.txt") const override;
  void load_vocab(const std::string &filename = "unigram_vocab.txt") override;
  ~Unigram() override;
};

#endif // UNIGRAM_HPP
