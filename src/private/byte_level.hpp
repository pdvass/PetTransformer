#ifndef BYTE_LEVEL_HPP
#define BYTE_LEVEL_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "utf32.hpp"

// Byte-level BPE alphabet, as in GPT-2's `bytes_to_unicode`.as
// Space (0x20) becomes U+0120 'Ġ', newline (0x0A) becomes U+010A 'Ċ'.
//
// Resources:
// https://github.com/openai/gpt-2/blob/master/src/encoder.py
namespace ByteLevel {

// Largest mapped code point is 255 or 256 + 67 = 323.
inline constexpr std::size_t ALPHABET_SIZE = 256;
inline constexpr std::size_t INVERSE_SIZE = 324;

namespace detail {

constexpr bool is_printable(std::size_t byte) {
  return (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
         (byte >= 174 && byte <= 255);
}

constexpr std::array<char32_t, ALPHABET_SIZE> make_table() {
  std::array<char32_t, ALPHABET_SIZE> table{};
  char32_t next{256};
  for (std::size_t byte = 0; byte < ALPHABET_SIZE; byte++) {
    table[byte] = is_printable(byte) ? static_cast<char32_t>(byte) : next++;
  }
  return table;
}

// code point -> byte, or -1 when the code point is not part of the alphabet.
constexpr std::array<std::int16_t, INVERSE_SIZE> make_inverse() {
  std::array<std::int16_t, INVERSE_SIZE> inverse{};
  for (auto &entry : inverse)
    entry = -1;

  const auto table{make_table()};
  for (std::size_t byte = 0; byte < ALPHABET_SIZE; byte++)
    inverse[table[byte]] = static_cast<std::int16_t>(byte);
  return inverse;
}

} // namespace detail

// BYTE_TO_CODE_POINT[b] is the visible character standing for byte b.
inline constexpr std::array<char32_t, ALPHABET_SIZE> BYTE_TO_CODE_POINT =
    detail::make_table();

inline constexpr std::array<std::int16_t, INVERSE_SIZE> CODE_POINT_TO_BYTE =
    detail::make_inverse();

static_assert(BYTE_TO_CODE_POINT[0x41] == U'A');
static_assert(BYTE_TO_CODE_POINT[0x20] == U'Ġ'); // space -> Ġ
static_assert(BYTE_TO_CODE_POINT[0x0A] == U'Ċ'); // newline -> Ċ
static_assert(BYTE_TO_CODE_POINT[0x09] == U'ĉ'); // tab -> ĉ
static_assert(BYTE_TO_CODE_POINT[0xAD] == U'Ń'); // last shifted byte

[[nodiscard]] constexpr std::optional<std::uint8_t>
byte_of(char32_t code_point) noexcept {
  if (code_point >= INVERSE_SIZE)
    return std::nullopt;
  const auto byte{CODE_POINT_TO_BYTE[code_point]};
  if (byte < 0)
    return std::nullopt;
  return static_cast<std::uint8_t>(byte);
}

// Raw bytes of a string (typically UTF-8) -> one visible character per byte.
[[nodiscard]] inline UTF32::CodePoints to_code_points(std::string_view bytes) {
  UTF32::CodePoints out{};
  out.reserve(bytes.size());
  for (const char ch : bytes)
    out.push_back(BYTE_TO_CODE_POINT[static_cast<unsigned char>(ch)]);
  return out;
}

// Inverse of to_code_points. The result holds raw bytes, it is UTF-8 only if
// the sequence of bytes was. Throws for a code point outside the alphabet.
[[nodiscard]] inline std::string
to_bytes(const UTF32::CodePoints &code_points) {
  std::string out{};
  out.reserve(code_points.size());
  for (const auto code_point : code_points) {
    const auto byte{byte_of(code_point)};
    if (!byte)
      throw std::invalid_argument(
          "Code point " +
          std::to_string(static_cast<std::uint32_t>(code_point)) +
          " is not in the byte-level alphabet");
    out.push_back(static_cast<char>(*byte));
  }
  return out;
}

} // namespace ByteLevel

#endif // BYTE_LEVEL_HPP
