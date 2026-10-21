#include "utf32.hpp"

#include <stdexcept>
#include <string>

#include "doctest.h"

// "This test must pass ✅" in five scripts. ✅ is U+2705 (3 bytes in UTF-8).
TEST_CASE("UTF32 round trip: same sentence in several languages") {
  const UTF32 utf8;

  const std::string greek = "Αυτό το τεστ πρέπει να περάσει ✅";
  const std::string russian = "Этот тест должен пройти ✅";
  const std::string japanese = "このテストは合格しなければなりません ✅";
  const std::string arabic = "يجب أن ينجح هذا الاختبار ✅";
  const std::string hindi = "यह परीक्षण पास होना चाहिए ✅";

  SUBCASE("greek") { CHECK(utf8.encode(utf8.decode(greek)) == greek); }
  SUBCASE("russian") { CHECK(utf8.encode(utf8.decode(russian)) == russian); }
  SUBCASE("japanese") { CHECK(utf8.encode(utf8.decode(japanese)) == japanese); }
  SUBCASE("arabic") { CHECK(utf8.encode(utf8.decode(arabic)) == arabic); }
  SUBCASE("hindi") { CHECK(utf8.encode(utf8.decode(hindi)) == hindi); }
}

TEST_CASE("UTF32 decodes to the expected code points") {
  const UTF32 utf8;

  SUBCASE("1 byte (ASCII)") {
    CHECK(utf8.decode("A") == UTF32::CodePoints{U'A'});
  }
  SUBCASE("2 bytes (Greek epsilon with tonos, U+03AD)") {
    CHECK(utf8.decode("έ") == UTF32::CodePoints{0x03AD});
  }
  SUBCASE("3 bytes (white heavy check mark, U+2705)") {
    CHECK(utf8.decode("✅") == UTF32::CodePoints{0x2705});
  }
  SUBCASE("4 bytes (grinning face, U+1F600)") {
    CHECK(utf8.decode("😀") == UTF32::CodePoints{0x1F600});
  }
  SUBCASE("byte count differs from code point count") {
    // "έρχομαι" is 7 code points but 14 bytes.
    const std::string word = "έρχομαι";
    CHECK(word.size() == 14);
    CHECK(utf8.decode(word).size() == 7);
  }
}

TEST_CASE("UTF32 round trip covers the 4-byte branch") {
  const UTF32 utf8;
  const std::string text = "Hello 😀 κόσμε ✅";
  CHECK(utf8.encode(utf8.decode(text)) == text);
}

TEST_CASE("UTF32 decode rejects malformed input") {
  const UTF32 utf8;

  SUBCASE("lone continuation byte") {
    CHECK_THROWS_AS(static_cast<void>(utf8.decode("\x80")), std::runtime_error);
  }
  SUBCASE("truncated 2-byte sequence") {
    CHECK_THROWS_AS(static_cast<void>(utf8.decode("\xC3")), std::runtime_error);
  }
  SUBCASE("truncated 3-byte sequence") {
    CHECK_THROWS_AS(static_cast<void>(utf8.decode("\xE2\x9C")),
                    std::runtime_error);
  }
  SUBCASE("truncated 4-byte sequence") {
    CHECK_THROWS_AS(static_cast<void>(utf8.decode("\xF0\x9F\x98")),
                    std::runtime_error);
  }
  SUBCASE("2-byte lead followed by a non-continuation byte") {
    CHECK_THROWS_AS(static_cast<void>(utf8.decode("\xC3\x41")),
                    std::runtime_error);
  }
  SUBCASE("continuation byte with marker 11 instead of 10") {
    CHECK_THROWS_AS(static_cast<void>(utf8.decode("\xC3\xC1")),
                    std::runtime_error);
  }
  SUBCASE("bad byte in the middle of a 3-byte sequence") {
    CHECK_THROWS_AS(static_cast<void>(utf8.decode("\xE2\x41\x85")),
                    std::runtime_error);
  }
  SUBCASE("code point above U+10FFFF") {
    // F4 90 80 80 encodes U+110000.
    CHECK_THROWS_AS(static_cast<void>(utf8.decode("\xF4\x90\x80\x80")),
                    std::runtime_error);
  }
}

TEST_CASE("UTF32 encode rejects code points above U+10FFFF") {
  const UTF32 utf8;
  CHECK_THROWS_AS(static_cast<void>(utf8.encode(UTF32::CodePoints{0x110000})),
                  std::runtime_error);
}
