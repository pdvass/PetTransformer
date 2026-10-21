#ifndef UTF32_HPP
#define UTF32_HPP

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

class UTF32 {
public:
  // https://cp-algorithms.com/string/string-hashing.html
  constexpr static std::uint64_t BASE = 1299709;
  constexpr static std::uint64_t MOD = 1e9 + 9;

  using CodePoint = char32_t;
  using CodePoints = std::vector<CodePoint>;

  [[nodiscard]]
  CodePoints decode(const std::string &line) const {
    CodePoints decoded{};
    auto bytearray = std::as_bytes(std::span(line));

    auto size = bytearray.size();
    for (size_t i = 0; i < size; i++) {
      // https://en.wikipedia.org/wiki/UTF-8#Description
      // I need to count consecutive one. They'll be from left at most 4.
      std::uint8_t current_byte = std::to_integer<std::uint8_t>(bytearray[i]);
      std::uint8_t code_points_count = std::countl_one(current_byte);

      if (code_points_count == 0) {

        auto ch = static_cast<CodePoint>(current_byte);
        decoded.push_back(ch);

      } else if (code_points_count == 2) {

        // Bounds Checking
        if (i + 1 >= size)
          throw std::runtime_error("Malformed byte sequence on byte " +
                                   std::to_string(i));

        // Then Byte 1: 110xxxyy
        auto masked_byte1 =
            static_cast<std::uint32_t>(current_byte & 0b00011111);
        std::uint8_t next_byte = std::to_integer<std::uint8_t>(bytearray[++i]);
        // And Byte 2: 10yyzzzz

        if ((next_byte & 0b11000000) != 0b10000000)
          throw std::runtime_error("Malformed byte sequence on byte " +
                                   std::to_string(i));

        auto masked_byte2 = static_cast<std::uint32_t>(next_byte & 0b00111111);

        auto ch = static_cast<CodePoint>((masked_byte1 << 6) | masked_byte2);
        decoded.push_back(ch);

      } else if (code_points_count == 3) {

        if (i + 2 >= size)
          throw std::runtime_error("Malformed byte sequence on byte " +
                                   std::to_string(i));

        // Then Byte 1: 1110wwww
        auto masked_byte1 =
            static_cast<std::uint32_t>(current_byte & 0b00001111);
        std::uint8_t next_byte = std::to_integer<std::uint8_t>(bytearray[++i]);
        // And Byte 2: 10xxxxyy

        if ((next_byte & 0b11000000) != 0b10000000)
          throw std::runtime_error("Malformed byte sequence on byte " +
                                   std::to_string(i));

        auto masked_byte2 = static_cast<std::uint32_t>(next_byte & 0b00111111);
        next_byte = std::to_integer<std::uint8_t>(bytearray[++i]);

        // And Byte 3: 10yyzzzz

        if ((next_byte & 0b11000000) != 0b10000000)
          throw std::runtime_error("Malformed byte sequence on byte " +
                                   std::to_string(i));

        auto masked_byte3 = static_cast<std::uint32_t>(next_byte & 0b00111111);
        auto ch = static_cast<CodePoint>((masked_byte1 << 12) |
                                         (masked_byte2 << 6) | masked_byte3);
        decoded.push_back(ch);
      } else if (code_points_count == 4) {

        if (i + 3 >= size)
          throw std::runtime_error("Malformed byte sequence on byte " +
                                   std::to_string(i));

        // Then Byte 1: 11110uvv
        auto masked_byte1 =
            static_cast<std::uint32_t>(current_byte & 0b00000111);
        std::uint8_t next_byte = std::to_integer<std::uint8_t>(bytearray[++i]);

        // And Byte 2: 10vvwwww

        if ((next_byte & 0b11000000) != 0b10000000)
          throw std::runtime_error("Malformed byte sequence on byte " +
                                   std::to_string(i));

        auto masked_byte2 = static_cast<std::uint32_t>(next_byte & 0b00111111);
        next_byte = std::to_integer<std::uint8_t>(bytearray[++i]);

        // And Byte 3: 10xxxxyy

        if ((next_byte & 0b11000000) != 0b10000000)
          throw std::runtime_error("Malformed byte sequence on byte " +
                                   std::to_string(i));

        auto masked_byte3 = static_cast<std::uint32_t>(next_byte & 0b00111111);
        next_byte = std::to_integer<std::uint8_t>(bytearray[++i]);

        // And Byte 4: 10yyzzzz

        if ((next_byte & 0b11000000) != 0b10000000)
          throw std::runtime_error("Malformed byte sequence on byte " +
                                   std::to_string(i));

        auto masked_byte4 = static_cast<std::uint32_t>(next_byte & 0b00111111);
        auto ch =
            static_cast<CodePoint>((masked_byte1 << 18) | (masked_byte2 << 12) |
                                   (masked_byte3 << 6) | masked_byte4);

        if (ch > 0x10FFFF)
          throw std::runtime_error("Malformed byte sequence on byte " +
                                   std::to_string(i));

        decoded.push_back(ch);

      } else {
        throw std::runtime_error("Malformed byte sequence on byte " +
                                 std::to_string(i));
      }
    }

    return decoded;
  }

  [[nodiscard]]
  std::string encode(const CodePoints &line) const {
    std::string str{};

    for (const auto chr : line) {
      if (chr < 0x0080) {

        str += static_cast<char>(chr);

      } else if (chr < 0x0800) {

        // Two bytes
        auto first_char = static_cast<char>(0b11000000 | (chr >> 6));
        auto second_char = static_cast<char>(0b10000000 | (chr & 0b00111111));
        str += first_char;
        str += second_char;

      } else if (chr < 0x010000) {

        // Three bytes
        auto first_char = static_cast<char>(0b11100000 | (chr >> 12));
        auto second_char =
            static_cast<char>(0b10000000 | ((chr >> 6) & 0b00111111));
        auto third_char = static_cast<char>(0b10000000 | (chr & 0b00111111));

        str += first_char;
        str += second_char;
        str += third_char;

      } else if (chr <= 0x10FFFF) {

        // Four bytes
        auto first_char = static_cast<char>(0b11110000 | (chr >> 18));
        auto second_char =
            static_cast<char>(0b10000000 | ((chr >> 12) & 0b00111111));
        auto third_char =
            static_cast<char>(0b10000000 | ((chr >> 6) & 0b00111111));
        auto fourth_char = static_cast<char>(0b10000000 | (chr & 0b00111111));

        str += first_char;
        str += second_char;
        str += third_char;
        str += fourth_char;

      } else {
        throw std::runtime_error(
            "Invalid code point " +
            std::to_string(static_cast<std::uint32_t>(chr)));
      }
    }

    return str;
  }

    struct Hasher {
    std::size_t operator()(const CodePoints &key) const noexcept {
      std::uint64_t hash_value = 0;
      std::uint64_t p_pow = 1;
      std::ranges::for_each(key, [&hash_value, &p_pow](CodePoint code_point) {
        hash_value =
            (hash_value +
             (static_cast<std::uint64_t>(code_point) - 'a' + 1) * p_pow) %
            UTF32::MOD;
        p_pow = (p_pow * UTF32::BASE) % UTF32::MOD;
      });

      return hash_value;
    }

    std::size_t
    operator()(const std::pair<CodePoints, CodePoints> &key) const noexcept {
      const auto h1 = (*this)(key.first);
      const auto h2 = (*this)(key.second);

      // https://github.com/boostorg/container_hash/blob/boost-1.92.0/include/boost/container_hash/hash.hpp#L474
      return h1 ^ (h2 + 0x9e3779b9ULL + (h1 << 6) + (h1 >> 2));
    }
  };
};

#endif // UTF32_HPP