#include <vector>
#include <limits>
#include "models/laml.h"

#include <iterator>
#include <algorithm>
#include <numeric>
#include <cmath>

template <typename Iter>
typename std::iterator_traits<Iter>::value_type
log_sum_exp(Iter begin, Iter end)
{
    using VT = typename std::iterator_traits<Iter>::value_type;
    if (begin == end) return VT{};

    auto max_elem = *std::max_element(begin, end);
    VT sum = std::accumulate(begin, end, VT{0}, 
        [max_elem](VT a, VT b) { return a + std::exp(b - max_elem); });

    return max_elem + std::log(sum);
}

std::vector<double> laml_model::compute_log_pmatrix_vector_product(
    size_t character, 
    double branch_length, 
    const std::vector<double>& log_vector
) const {
    double nu = this->parameters[0];

    std::vector<double> result(this->alphabet_sizes[character]);
    std::vector<double> tmp(log_vector.begin(), log_vector.end()); // tmp LSE array

    /* Handle i = 0 which is state = ? case */
    result[0] = log_vector[0];

    /* Handle i = 1 which is state = 0 case */
    tmp[0] += std::log(1 - std::exp(-branch_length * nu));
    tmp[1] += -branch_length * (1 + nu);
    for (size_t j = 2; j < this->alphabet_sizes[character]; j++) {
        tmp[j] += std::log(this->mutation_priors[character][j - 2])
                 - branch_length * nu + std::log(1 - std::exp(-branch_length));
    }
    result[1] = log_sum_exp(tmp.begin(), tmp.end());

    /* Handle remaining i \notin {0, 1}, non-missing cases */
    double tmp2[] = {0.0, 0.0};
    for (size_t i = 2; i < this->alphabet_sizes[character]; i++) {
        tmp2[0] = log_vector[i] -  branch_length * nu;
        tmp2[1] = log_vector[0] + std::log(1 - std::exp(-branch_length * nu));
        result[i] = log_sum_exp(tmp2, tmp2 + 2);
    }

    return result;
}

std::vector<double> laml_model::compute_log_pmatrix_transpose_vector_product(
    size_t character, 
    double branch_length, 
    const std::vector<double>& log_vector
) const {
    std::vector<double> result(this->alphabet_sizes[character], -1.0);
    return result;
}

std::vector<double> laml_model::compute_taxa_log_inside_likelihood(
    size_t character, 
    size_t taxa_id
) const {
    double phi = this->parameters[1];
    int state = this->character_matrix[taxa_id][character];
    
    double log_phi = std::log(phi);
    std::vector<double> result(
        this->alphabet_sizes[character], 
        -1e7//std::numeric_limits<double>::infinity()
    );

    if (state == -1) {
        result[0] = 0.0;
        for (size_t i = 1; i < this->alphabet_sizes[character]; i++) {
            result[i] = log_phi;
        }
    } else {
        result[state + 1] = std::log(1 - phi);
    }

    return result;
};

std::vector<double> laml_model::compute_root_distribution(size_t character) const {
    std::vector<double> result(this->alphabet_sizes[character], -1.0);
    return result;
}
