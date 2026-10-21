#ifndef FROZER_PAIR_HPP
#define FROZER_PAIR_HPP

#include <optional>
#include <stdexcept>
#include <unordered_map>

#include "tokenizers/frozen_pair_fwd.hpp"

template <typename A, typename B, typename HashA, typename HashB>
class FrozenPair {
private:
  std::unordered_map<A, B, HashA> a_to_b;
  std::unordered_map<B, A, HashB> b_to_a;

public:
  FrozenPair() = default;

  [[nodiscard]] bool insert(A a, B b) {
    if (a_to_b.contains(a) || b_to_a.contains(b)) {
      return false;
    }
    a_to_b[a] = b;
    b_to_a[b] = a;
    return true;
  };

  [[nodiscard]] std::optional<B> get(A a) const {
    if (a_to_b.contains(a)) {
      return a_to_b.at(a);
    }
    return std::nullopt;
  };

  [[nodiscard]] std::optional<A> get(B b) const {
    if (b_to_a.contains(b)) {
      return b_to_a.at(b);
    }
    return std::nullopt;
  };

  [[nodiscard]] size_t len() const {
    if (a_to_b.size() != b_to_a.size()) {
      throw std::runtime_error("Maps have different sizes.");
    }
    return a_to_b.size();
  }

  [[nodiscard]] auto begin() const { return a_to_b.begin(); }
  [[nodiscard]] auto end() const { return a_to_b.end(); }
};

#endif // FROZER_PAIR_HPP