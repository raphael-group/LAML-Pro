#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

#include "strided_read.h"

// The numpy->C++ converters read arrays via per-axis element strides so any
// memory layout is handled correctly. These test that stride math directly with
// hand-built buffers -- no interpreter needed -- for the layouts numpy produces.
// Logical matrix under test:  [[1, 2, 3],
//                              [4, 5, 6]]
TEST_CASE("read_strided_2d handles all memory layouts", "[strided]") {
    const std::vector<std::vector<int32_t>> want{{1, 2, 3}, {4, 5, 6}};

    SECTION("C-order (row-major): s0=n1, s1=1") {
        int32_t buf[] = {1, 2, 3, 4, 5, 6};
        REQUIRE(read_strided_2d<int32_t>(buf, 2, 3, 3, 1) == want);
    }

    SECTION("F-order (column-major): s0=1, s1=n0") {
        int32_t buf[] = {1, 4, 2, 5, 3, 6};
        REQUIRE(read_strided_2d<int32_t>(buf, 2, 3, 1, 2) == want);
    }

    SECTION("strided column slice (e.g. wide[:, ::2]): s1=2") {
        // Every other column of a 2x6 row-major buffer holds the wanted values;
        // the interleaved 9s are the dropped columns.
        int32_t buf[] = {1, 9, 2, 9, 3, 9,
                         4, 9, 5, 9, 6, 9};
        REQUIRE(read_strided_2d<int32_t>(buf, 2, 3, 6, 2) == want);
    }
}

TEST_CASE("read_strided_3d handles C- and F-order", "[strided]") {
    // Logical 2x2x2 with values 0..7 in C-order.
    const std::vector<std::vector<std::vector<double>>> want{
        {{0, 1}, {2, 3}}, {{4, 5}, {6, 7}}};

    SECTION("C-order: s0=4, s1=2, s2=1") {
        double buf[] = {0, 1, 2, 3, 4, 5, 6, 7};
        REQUIRE(read_strided_3d<double>(buf, 2, 2, 2, 4, 2, 1) == want);
    }

    SECTION("F-order: s0=1, s1=2, s2=4") {
        double buf[] = {0, 4, 2, 6, 1, 5, 3, 7};
        REQUIRE(read_strided_3d<double>(buf, 2, 2, 2, 1, 2, 4) == want);
    }
}
