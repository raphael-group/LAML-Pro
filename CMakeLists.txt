cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_DEBUG "-g --coverage")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -funsafe-math-optimizations -march=native")

project(fastlaml VERSION 1.0)

add_subdirectory(src/)
add_subdirectory(tests/)

target_include_directories(fastlaml PUBLIC "${fastlaml_SOURCE_DIR}/include")
target_include_directories(fastlaml PUBLIC "${fastlaml_SOURCE_DIR}/third-party/csv-parser/include")
target_include_directories(fastlaml_tests PUBLIC "${fastlaml_SOURCE_DIR}/include")
target_include_directories(fastlaml_tests PUBLIC "${fastlaml_SOURCE_DIR}/include")

# turn off JSON library tests
set(JSON_BuildTests OFF CACHE INTERNAL "")

# compile libraries
add_subdirectory(third-party/spdlog)
add_subdirectory(third-party/csv-parser)
add_subdirectory(third-party/argparse)
add_subdirectory(third-party/json)
add_subdirectory(third-party/gcem)
add_subdirectory(third-party/nanobench)
add_subdirectory(third-party/osqp)
add_subdirectory(third-party/Catch2)