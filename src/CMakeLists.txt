add_library(tokenizer tokenizers/wordpiece.cpp tokenizers/bpe.cpp
    tokenizers/unigram.cpp private/unicode_norm.cpp)

target_include_directories(tokenizer PRIVATE ${PROJECT_SOURCE_DIR}/src/private/)
target_include_directories(tokenizer PUBLIC ${PROJECT_SOURCE_DIR}/include/pet_transformer)

target_compile_features(tokenizer PUBLIC cxx_std_20)

add_executable(PetTransformer main.cpp)

target_link_libraries(PetTransformer PRIVATE tokenizer)

# bpe.hpp includes the private utf32.hpp.
target_include_directories(PetTransformer PRIVATE ${PROJECT_SOURCE_DIR}/src/private)
