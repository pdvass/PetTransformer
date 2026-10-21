#include "byte_level.hpp"

#include <cstdint>
#include <set>
#include <string>

#include "doctest.h"

TEST_CASE("the table maps the 256 bytes to 256 distinct code points") {
  std::set<char32_t> seen{};
  for (const auto code_point : ByteLevel::BYTE_TO_CODE_POINT)
    seen.insert(code_point);
  CHECK(seen.size() == 256);
}

TEST_CASE("printable bytes map to themselves") {
  for (std::size_t byte = 33; byte <= 126; byte++)
    CHECK(ByteLevel::BYTE_TO_CODE_POINT[byte] == static_cast<char32_t>(byte));
  for (std::size_t byte = 161; byte <= 172; byte++)
    CHECK(ByteLevel::BYTE_TO_CODE_POINT[byte] == static_cast<char32_t>(byte));
  for (std::size_t byte = 174; byte <= 255; byte++)
    CHECK(ByteLevel::BYTE_TO_CODE_POINT[byte] == static_cast<char32_t>(byte));
}

TEST_CASE("the other 68 bytes are shifted to 256 and up, in byte order") {
  // Values as in GPT-2's bytes_to_unicode.
  CHECK(ByteLevel::BYTE_TO_CODE_POINT[0x00] == 256);
  CHECK(ByteLevel::BYTE_TO_CODE_POINT[0x09] == U'ĉ'); // tab     -> ĉ
  CHECK(ByteLevel::BYTE_TO_CODE_POINT[0x0A] == U'Ċ'); // newline -> Ċ
  CHECK(ByteLevel::BYTE_TO_CODE_POINT[0x20] == U'Ġ'); // space   -> Ġ
  CHECK(ByteLevel::BYTE_TO_CODE_POINT[0x7F] == 289);
  CHECK(ByteLevel::BYTE_TO_CODE_POINT[0xA0] == 322);
  CHECK(ByteLevel::BYTE_TO_CODE_POINT[0xAD] == 323);
}

TEST_CASE("every byte survives the inverse table") {
  for (std::size_t byte = 0; byte < 256; byte++) {
    const auto back{ByteLevel::byte_of(ByteLevel::BYTE_TO_CODE_POINT[byte])};
    REQUIRE(back.has_value());
    CHECK(*back == byte);
  }
}

TEST_CASE("code points outside the alphabet have no byte") {
  // The last image is 323 (byte 0xAD).
  CHECK_FALSE(ByteLevel::byte_of(324).has_value());
  CHECK_FALSE(ByteLevel::byte_of(U'Ѐ').has_value());
  CHECK_FALSE(ByteLevel::byte_of(0x1F600).has_value());
  // Control characters below 33 are not images of themselves.
  CHECK_FALSE(ByteLevel::byte_of(U' ').has_value());
  CHECK_FALSE(ByteLevel::byte_of(U'\n').has_value());
}

TEST_CASE("printable ASCII keeps its letters, a space becomes G-dot") {
  const auto mapped{ByteLevel::to_code_points("a b")};
  REQUIRE(mapped.size() == 3);
  CHECK(mapped[0] == U'a');
  CHECK(mapped[1] == U'Ġ');
  CHECK(mapped[2] == U'b');
}

TEST_CASE("a multi-byte character becomes one symbol per byte") {
  // U+201C LEFT DOUBLE QUOTATION MARK is E2 80 9C in UTF-8.
  const auto mapped{ByteLevel::to_code_points("\xE2\x80\x9C")};
  CHECK(mapped.size() == 3);
}

TEST_CASE("to_bytes(to_code_points(s)) returns s") {
  const std::string samples[] = {
      "",
      "plain ascii",
      "tabs\tand\nnewlines  and   runs of spaces",
      "\xE2\x80\x9Cquoted\xE2\x80\x9D \xE2\x80\x94 dash",   // curly quotes, dash
      "\xF0\x9F\x98\x80 emoji",                              // U+1F600
      "caf\xC3\xA9",                                        // é
      std::string("embedded\0nul", 12),
  };
  for (const auto &sample : samples)
    CHECK(ByteLevel::to_bytes(ByteLevel::to_code_points(sample)) == sample);
}

TEST_CASE("to_bytes rejects code points that are not in the alphabet") {
  CHECK_THROWS_AS((void)ByteLevel::to_bytes({U'a', U'Ѐ'}),
                  std::invalid_argument);
  CHECK_THROWS_AS((void)ByteLevel::to_bytes({U' '}), std::invalid_argument);
}
