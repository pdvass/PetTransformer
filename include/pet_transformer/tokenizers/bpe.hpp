#ifndef BPE_HPP
#define BPE_HPP

#include "frozen_pair_fwd.hpp"
#include "tokenizer.hpp"
#include "utf32.hpp"

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

const auto BPE_SPECIAL_TOKENS{std::array<std::string, 1>{"<|endoftext|>"}};

class BPE : public Tokenizer {
private:
  friend struct BPETestAccess;
  using Pair = std::pair<UTF32::CodePoints, UTF32::CodePoints>;

  std::unique_ptr<FrozenPair<UTF32::CodePoints, int64_t, UTF32::Hasher>> vocab;
  UTF32 utf;
  mutable std::unordered_map<UTF32::CodePoints, int64_t, UTF32::Hasher> counts;
  std::vector<UTF32::CodePoints> alphabet;
  std::unordered_map<UTF32::CodePoints, std::vector<UTF32::CodePoints>,
                     UTF32::Hasher>
      splits;

  std::unordered_map<Pair, int64_t, UTF32::Hasher> pairs;
  std::unordered_map<Pair, int64_t, UTF32::Hasher> merge_rank; // pair -> index

  const int64_t vocab_size;
  const int64_t min_freq;

  // Text between special tokens, with the index of the special token that
  // follows it. Plain text has index -1 and an empty special token has
  // empty text.
  [[nodiscard]] std::vector<std::pair<std::string, int64_t>>
  split_special(const std::string &) const;

  [[nodiscard]] UTF32::CodePoints normalize(const std::string &) const;
  [[nodiscard]] std::vector<UTF32::CodePoints>
  split(const UTF32::CodePoints &) const;
  std::vector<UTF32::CodePoints>
  to_symbols(const std::vector<UTF32::CodePoints> &) const;

  void populate_vocab();
  void compute_pair_scores();
  void merge_pair(const Pair &);

  void count_words(const std::vector<UTF32::CodePoints> &) const noexcept;

  void encode_word(const UTF32::CodePoints &, std::vector<UTF32::CodePoints> &);

public:
  // Input must be valid UTF-8. Throws std::runtime_error on a malformed
  // sequence.
  void feed(const std::string &) override;
  std::vector<int64_t> encode(const std::string &) override;
  [[nodiscard]] std::string decode(const std::vector<int64_t> &) const override;
  void train() override;

  // Throws std::invalid_argument if vocab_size is below the special token plus
  // the 256 byte symbols.
  explicit BPE(TrainConfig config);
  void save_vocab(const std::string &filename = "bpe_vocab.txt") const override;
  void load_vocab(const std::string &filename = "bpe_vocab.txt") override;
  void print_info() const;
  ~BPE();
};

#endif // BPE_HPP