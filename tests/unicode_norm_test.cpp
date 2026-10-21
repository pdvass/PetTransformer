#include "unicode_norm.hpp"

#include <string>
#include <utility>
#include <vector>

#include "doctest.h"
#include "utf32.hpp"

// Expected values come from Python's unicodedata.normalize("NFC", ...).

namespace {

struct NfcCase {
  const char *name;
  UTF32::CodePoints in;
  UTF32::CodePoints out;

  NfcCase(const char *name_, UTF32::CodePoints in_, UTF32::CodePoints out_)
      : name(name_), in(std::move(in_)), out(std::move(out_)) {}
};

const std::vector<NfcCase> NFC_CASES{
    {"empty", {}, {}},
    {"ascii", {0x0063, 0x0061, 0x006E}, {0x0063, 0x0061, 0x006E}},

    // Composition of the same letter spelled in different ways.
    {"e-acute already composed", {0x00E9}, {0x00E9}},
    {"e + acute", {0x0065, 0x0301}, {0x00E9}},
    {"greek epsilon + tonos", {0x03B5, 0x0301}, {0x03AD}},
    {"u-diaeresis-acute composed", {0x01D8}, {0x01D8}},
    {"u-diaeresis + acute", {0x00FC, 0x0301}, {0x01D8}},
    {"u + diaeresis + acute", {0x0075, 0x0308, 0x0301}, {0x01D8}},
    {"angstrom sign", {0x212B}, {0x00C5}},

    // Canonical ordering: the marks are reordered, then composed.
    {"dot above then below", {0x0061, 0x0307, 0x0323}, {0x1EA1, 0x0307}},
    {"dot below then above", {0x0061, 0x0323, 0x0307}, {0x1EA1, 0x0307}},
    {"same class keeps its order", {0x0061, 0x0308, 0x0307}, {0x00E4, 0x0307}},
    {"mark between letters",
     {0x0063, 0x0323, 0x0307, 0x0061},
     {0x010B, 0x0323, 0x0061}},

    // Blocking.
    {"blocked: b acute dot", {0x0062, 0x0301, 0x0307}, {0x0062, 0x0301, 0x0307}},
    {"not blocked: a cedilla acute", {0x0061, 0x0327, 0x0301}, {0x00E1, 0x0327}},

    // Marks without a starter.
    {"marks at the start", {0x0307, 0x0323}, {0x0323, 0x0307}},
    {"marks then a letter", {0x0307, 0x0323, 0x0061}, {0x0323, 0x0307, 0x0061}},
    {"marks of the same class", {0x0308, 0x0307}, {0x0308, 0x0307}},

    // Composition exclusions.
    {"excluded 0344", {0x0344}, {0x0308, 0x0301}},

    // A class-0 code point only combines when nothing sits between.
    {"oriya E + AA", {0x0B47, 0x0B3E}, {0x0B4B}},
    {"oriya blocked by nukta",
     {0x0B47, 0x0B3C, 0x0B3E},
     {0x0B47, 0x0B3C, 0x0B3E}},

    // Hangul.
    {"hangul L V", {0x1100, 0x1161}, {0xAC00}},
    {"hangul L V T", {0x1100, 0x1161, 0x11A8}, {0xAC01}},
    {"hangul LV + T", {0xAC00, 0x11A8}, {0xAC01}},
    {"hangul LVT + T", {0xAC01, 0x11A8}, {0xAC01, 0x11A8}},
    {"hangul LV + 11A7", {0xAC00, 0x11A7}, {0xAC00, 0x11A7}},
    {"hangul blocked by tone mark",
     {0x1100, 0x302E, 0x1161},
     {0x1100, 0x302E, 0x1161}},

    // NFC leaves compatibility characters alone (that would be NFKC).
    {"ligature fi", {0xFB01}, {0xFB01}},
    {"fullwidth A", {0xFF21}, {0xFF21}},

    // Outside the tables.
    {"emoji", {0x1F600}, {0x1F600}},
    {"last code point", {0x10FFFF}, {0x10FFFF}},
};

} // namespace

TEST_CASE("nfc gives the expected result") {
  for (const auto &c : NFC_CASES) {
    SUBCASE(c.name) { CHECK(nfc(c.in) == c.out); }
  }
}

TEST_CASE("nfc is idempotent") {
  for (const auto &c : NFC_CASES) {
    SUBCASE(c.name) { CHECK(nfc(nfc(c.in)) == nfc(c.in)); }
  }
}

TEST_CASE("nfc keeps every precomposed Hangul syllable and recomposes jamo") {
  // Independent of the code under test: the formulas from Unicode ch. 3.12.
  constexpr char32_t S_BASE{0xAC00}, L_BASE{0x1100}, V_BASE{0x1161},
      T_BASE{0x11A7};
  constexpr char32_t V_COUNT{21}, T_COUNT{28}, S_COUNT{11172};

  int unchanged_failures{0};
  int recomposed_failures{0};
  for (char32_t s = S_BASE; s < S_BASE + S_COUNT; ++s) {
    const char32_t index{s - S_BASE};
    UTF32::CodePoints jamo{L_BASE + index / (V_COUNT * T_COUNT),
                           V_BASE + (index % (V_COUNT * T_COUNT)) / T_COUNT};
    if (index % T_COUNT != 0)
      jamo.push_back(T_BASE + index % T_COUNT);

    if (nfc({s}) != UTF32::CodePoints{s})
      ++unchanged_failures;
    if (nfc(jamo) != UTF32::CodePoints{s})
      ++recomposed_failures;
  }
  CHECK(unchanged_failures == 0);
  CHECK(recomposed_failures == 0);
}

TEST_CASE("nfc on real text: decomposed Greek becomes composed") {
  const UTF32 utf;
  // "Αυτό το" with the omicron + tonos written as two code points...
  const std::string decomposed{"Αυτό το"};
  // ...and as the single composed code point U+03CC.
  const std::string composed{"Αυτό το"};

  CHECK(utf.encode(nfc(utf.decode(decomposed))) == composed);
  CHECK(utf.encode(nfc(utf.decode(composed))) == composed);
}
