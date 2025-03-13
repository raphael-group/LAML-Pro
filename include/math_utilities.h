#ifndef MATH_UTILITIES_H
#define MATH_UTILITIES_H

#include <array>
#include <cmath>
#include <algorithm>

#include "gcem.hpp"

/* 
 * Flags to enable fast math functions, 
 * slightly altering likelihood calculations 
 */
#define FAST_LOG 1
#define FAST_EXP 1

#define TABLE_SIZE 1000000

#define EXP_MIN  -25.0
#define EXP_MAX  +25.0

#define LOG_MIN  1e-7
#define LOG_MAX  5.0

constexpr std::array<double, TABLE_SIZE + 1> make_exp_table() {
    std::array<double, TABLE_SIZE + 1> table{};
    for (int i = 0; i <= TABLE_SIZE; ++i) {
        double x = EXP_MIN + ((EXP_MAX - EXP_MIN) * i / TABLE_SIZE);
        double y = gcem::exp(x);
        table[i] = y;
    }
    return table;
}

constexpr std::array<double, TABLE_SIZE + 1> make_log_table() {
    std::array<double, TABLE_SIZE + 1> table{};
    for (int i = 0; i <= TABLE_SIZE; ++i) {
        double x = LOG_MIN + ((LOG_MAX - LOG_MIN) * i / TABLE_SIZE);
        double y = gcem::log(x);
        table[i] = y;
    }
    return table;
}

inline auto EXP_TABLE = make_exp_table();
inline auto LOG_TABLE = make_log_table();

inline double fast_exp(double x) {
    if (x <= EXP_MIN) return 0.0;
    if (x >= EXP_MAX) return 1e7;
    int index = static_cast<int>((x - EXP_MIN) * ((double) TABLE_SIZE) / (EXP_MAX - EXP_MIN));
    return EXP_TABLE[index];
}

inline double fast_log(double x) {
    if (x <= LOG_MIN) return -16.118;
    if (x >= LOG_MAX) return 0.0;
    int index = static_cast<int>((x - LOG_MIN) * ((double) TABLE_SIZE) / (LOG_MAX - LOG_MIN));
    return LOG_TABLE[index];
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
        #if FAST_EXP
        sum += fast_exp(*it - max_elem);
        #else
        sum += std::exp(*it - max_elem);
        #endif
    }

    #if FAST_LOG
    return max_elem + fast_log(sum);
    #else
    return max_elem + std::log(sum);
    #endif
}

#endif // MATH_UTILITIES_H