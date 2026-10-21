#ifndef WORDPIECE_HPP
#define WORDPIECE_HPP

#include "frozen_pair_fwd.hpp"
#include "tokenizer.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

const auto SPECIAL_TOKENS{
    std::array<std::string, 5>{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}};
/**
 * Based on: https://huggingface.co/docs/course/chapter6/6
 *
 * feed -> normalize -> split -> count_words -> populate_vocab.
 *
 * train -> score_pairs -> merge until vocab length is reached.
 */

/**
 * @brief
 *
 */
class WordPiece : public Tokenizer {
private:
  friend struct WordPieceTestAccess;
  using Pair = std::pair<std::string, std::string>;

  std::unique_ptr<FrozenPair<std::string, int64_t>> vocab;
  std::vector<std::string> alphabet;

  mutable std::map<std::string, int64_t> counts;
  std::map<Pair, double> pairs;
  // Split of a word. eg This: [T, h, i, s]
  std::map<std::string, std::vector<std::string>> splits;

  const int64_t vocab_size;
  const int64_t min_freq;

  [[nodiscard]] std::string normalize(std::string) const;
  [[nodiscard]] std::vector<std::string> split(const std::string &) const;

  void count_words(const std::vector<std::string> &) const noexcept;

  void populate_vocab();
  void compute_pair_scores();
  void merge_pair(const Pair &);

  void encode_word(const std::string &, std::vector<int64_t> &);

public:
  void feed(const std::string &) override;
  std::vector<int64_t> encode(const std::string &) override;
  // The way I understand it, I use the nodiscard part on Pure functions.
  [[nodiscard]] std::string decode(const std::vector<int64_t> &) const override;
  // Throws std::invalid_argument if vocab_size is below the special tokens
  // plus the alphabet found in the fed text.
  void train() override;

  explicit WordPiece(TrainConfig config);
  void save_vocab(
      const std::string &filename = "wordpiece_vocab.txt") const override;
  void load_vocab(const std::string &filename = "wordpiece_vocab.txt") override;
  void print_info() const;
  ~WordPiece();
};

#endif // WORDPIECE_HPP