#ifndef FROZEN_PAIR_FWD_HPP
#define FROZEN_PAIR_FWD_HPP

#include <functional>

// https://en.cppreference.com/cpp/language/pimpl
template <typename A, typename B, typename HashA = std::hash<A>,
          typename HashB = std::hash<B>>
class FrozenPair;

#endif // FROZEN_PAIR_FWD_HPP
