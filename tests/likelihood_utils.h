// likelihood_test_utils.h
#pragma once
#include <vector>
#include "phylogeny.h"

inline bool check_inside_outside(
    const likelihood_buffer& inside_ll,
    const likelihood_buffer& outside_ll,
    const std::vector<double>& llh,
    int max_alphabet_size,
    double tolerance
) {
    size_t num_characters = inside_ll.num_characters;
    size_t num_nodes = inside_ll.num_nodes;

    for (size_t character = 0; character < num_characters; ++character) {
        for (size_t node = 0; node < num_nodes; ++node) {
            std::vector<double> tmp_buffer(max_alphabet_size, 0.0);
            for (int j = 0; j < max_alphabet_size; ++j) {
                tmp_buffer[j] = inside_ll(character, node, j) + outside_ll(character, node, j);
            }

            double test_llh = log_sum_exp(tmp_buffer.begin(), tmp_buffer.end());
            if (std::abs(llh[character] - test_llh) > tolerance) {
                return false;
            }
        }
    }

    return true;
}

