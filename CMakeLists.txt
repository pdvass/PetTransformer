cmake_minimum_required(VERSION 3.28...4.4)

include(FetchContent)
FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest
  GIT_TAG v2.5.0
)
FetchContent_MakeAvailable(doctest)

project(
  PetTransformer
  VERSION 0.1
  DESCRIPTION "Learning project"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Warnings for our own targets (doctest is added above, so it isn't affected).
add_compile_options(-Wall -Wextra)

# clang-tidy runs on every compiled file when found. Checks are in .clang-tidy.
# Configure with -DENABLE_CLANG_TIDY=OFF to skip it for faster builds.
option(ENABLE_CLANG_TIDY "Run clang-tidy while compiling" ON)
if(ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy
    HINTS "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/x64/bin")
  if(CLANG_TIDY_EXE)
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_EXE})
  else()
    message(WARNING "clang-tidy not found, skipping")
  endif()
endif()

add_subdirectory(src)
add_subdirectory(metrics)

enable_testing()
add_subdirectory(tests)


