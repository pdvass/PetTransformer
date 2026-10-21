add_executable(wordpiece_test wordpiece_test.cpp)

target_link_libraries(wordpiece_test PRIVATE tokenizer doctest::doctest_with_main)

target_include_directories(wordpiece_test PRIVATE
  ${PROJECT_SOURCE_DIR}/include/pet_transformer/tokenizers
  ${PROJECT_SOURCE_DIR}/src/private)

add_test(NAME wordpiece_test COMMAND wordpiece_test)

add_executable(utf32_test utf32_test.cpp)

target_link_libraries(utf32_test PRIVATE doctest::doctest_with_main)

target_include_directories(utf32_test PRIVATE ${PROJECT_SOURCE_DIR}/src/private)

add_test(NAME utf32_test COMMAND utf32_test)

add_executable(unicode_norm_test unicode_norm_test.cpp
  ${PROJECT_SOURCE_DIR}/src/private/unicode_norm.cpp)

target_link_libraries(unicode_norm_test PRIVATE doctest::doctest_with_main)

target_include_directories(unicode_norm_test PRIVATE ${PROJECT_SOURCE_DIR}/src/private)

add_test(NAME unicode_norm_test COMMAND unicode_norm_test)

add_executable(bpe_test bpe_test.cpp)

target_link_libraries(bpe_test PRIVATE tokenizer doctest::doctest_with_main)

target_include_directories(bpe_test PRIVATE
  ${PROJECT_SOURCE_DIR}/include/pet_transformer/tokenizers
  ${PROJECT_SOURCE_DIR}/src/private)

add_test(NAME bpe_test COMMAND bpe_test)

add_executable(unigram_test unigram_test.cpp)

target_link_libraries(unigram_test PRIVATE tokenizer doctest::doctest_with_main)

target_include_directories(unigram_test PRIVATE
  ${PROJECT_SOURCE_DIR}/include/pet_transformer/tokenizers
  ${PROJECT_SOURCE_DIR}/src/private)

add_test(NAME unigram_test COMMAND unigram_test)

add_executable(byte_level_test byte_level_test.cpp)

target_link_libraries(byte_level_test PRIVATE doctest::doctest_with_main)

target_include_directories(byte_level_test PRIVATE ${PROJECT_SOURCE_DIR}/src/private)

add_test(NAME byte_level_test COMMAND byte_level_test)
