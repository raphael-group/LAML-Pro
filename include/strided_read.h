#ifndef STRIDED_READ_H
#define STRIDED_READ_H

#include <cstddef>
#include <vector>

// Copy a logically (n0 x n1) matrix out of a flat buffer using per-axis element
// strides. Works for any layout: C-order (s0=n1, s1=1), F-order (s0=1, s1=n0),
// strided slices, etc. Strides are in elements, not bytes.
template <typename T>
std::vector<std::vector<T>> read_strided_2d(
    const T* p, std::size_t n0, std::size_t n1, std::ptrdiff_t s0, std::ptrdiff_t s1
) {
    std::vector<std::vector<T>> out(n0, std::vector<T>(n1));
    for (std::size_t i = 0; i < n0; ++i)
        for (std::size_t j = 0; j < n1; ++j)
            out[i][j] = p[i * s0 + j * s1];
    return out;
}

// 3D analogue of read_strided_2d.
template <typename T>
std::vector<std::vector<std::vector<T>>> read_strided_3d(
    const T* p, std::size_t n0, std::size_t n1, std::size_t n2,
    std::ptrdiff_t s0, std::ptrdiff_t s1, std::ptrdiff_t s2
) {
    std::vector<std::vector<std::vector<T>>> out(
        n0, std::vector<std::vector<T>>(n1, std::vector<T>(n2)));
    for (std::size_t i = 0; i < n0; ++i)
        for (std::size_t j = 0; j < n1; ++j)
            for (std::size_t k = 0; k < n2; ++k)
                out[i][j][k] = p[i * s0 + j * s1 + k * s2];
    return out;
}

#endif
