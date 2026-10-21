#ifndef UNICODE_NORM_HPP
#define UNICODE_NORM_HPP

#include "utf32.hpp"

// Normalizes the code points to NFC: canonical decomposition, canonical
// ordering, then canonical composition.
// The algorithm and its resources are documented in unicode_norm.cpp.
[[nodiscard]] UTF32::CodePoints nfc(const UTF32::CodePoints &code_points);

#endif // UNICODE_NORM_HPP
