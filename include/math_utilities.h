#ifndef MATH_UTILITIES_H
#define MATH_UTILITIES_H

#include <array>
#include <cmath>
#include <algorithm>

#include "gcem.hpp"

#define TABLE_SIZE 100000
#define TABLE_MIN  -25.0
#define TABLE_MAX  +25.0

constexpr std::array<double, TABLE_SIZE + 1> make_exp_table() {
    std::array<double, TABLE_SIZE + 1> table{};
    for (int i = 0; i <= TABLE_SIZE; ++i) {
        double x = TABLE_MIN + ((TABLE_MAX - TABLE_MIN) * i / TABLE_SIZE);
        double y = gcem::exp(x);
        table[i] = y;
    }
    return table;
}

auto EXP_TABLE = make_exp_table();

inline double fast_exp(double x) {
    if (x <= TABLE_MIN) return 0.0;
    if (x >= TABLE_MAX) return 1e7;
    int index = static_cast<int>((x - TABLE_MIN) * ((double) TABLE_SIZE) / (TABLE_MAX - TABLE_MIN));
    return EXP_TABLE[index];
}

template <typename Iter>
typename std::iterator_traits<Iter>::value_type
log_sum_exp(Iter begin, Iter end)
{
    using VT = typename std::iterator_traits<Iter>::value_type;
    if (begin == end) return VT{};

    auto max_elem = *std::max_element(begin, end);
    VT sum = VT{0};
    for (auto it = begin; it != end; ++it) {
        sum += fast_exp(*it - max_elem);
    }

    return max_elem + std::log(sum);
}

#endif // MATH_UTILITIES_H