#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>

struct TrainConfig {
  int64_t vocab_size;
  int64_t min_freq{0};
};

class Tokenizer {

public:
  virtual void feed(const std::string &) = 0;
  virtual void train() = 0;
  virtual std::vector<int64_t> encode(const std::string &) = 0;

  virtual void save_vocab(const std::string &) const = 0;
  virtual void load_vocab(const std::string &) = 0;
  [[nodiscard]] virtual std::string
  decode(const std::vector<int64_t> &) const = 0;

  virtual ~Tokenizer() = default;

protected:
  // Every tokenizer refuses a vocab_size below its base the same way.
  static void check_vocab_size(int64_t vocab_size, int64_t base) {
    if (vocab_size < base)
      throw std::invalid_argument(
          std::format("vocab_size {} is smaller than the base vocabulary ({})",
                      vocab_size, base));
  }
};

#endif // TOKENIZER_HPP
