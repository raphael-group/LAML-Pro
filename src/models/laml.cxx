#include <vector>
#include <limits>
#include "models/laml.h"

#include <iterator>
#include <algorithm>
#include <numeric>
#include <cmath>

#define NEGATIVE_INFINITY (-1e7) //-std::numeric_limits<double>::infinity())

double underflow_exp(double x) {
    if (x <= -100) return 0.0;
    return std::exp(x);
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
        sum += underflow_exp(*it - max_elem);
    }

    return max_elem + std::log(sum);
}

void laml_model::compute_log_pmatrix_vector_product(
    laml_data& d,
    size_t character, 
    double branch_length, 
    const std::vector<double>& log_vector,
    std::vector<double>& result
) const {
    size_t alphabet_size = this->alphabet_sizes[character];
    double nu = this->parameters[0];

    std::vector<double>& tmp = d.buffer;
    std::copy(log_vector.begin(), log_vector.begin() + alphabet_size, tmp.begin());
    
    /* Handle i = 0 which is state = ? case */
    result[0] = log_vector[0];

    /* Handle i = 1 which is state = 0 case */
    tmp[0] += d.v2;
    tmp[1] += -branch_length * (1 + nu);
    for (size_t j = 2; j < alphabet_size; j++) {
        tmp[j] += this->log_mutation_priors[character][j - 2] - branch_length * nu + d.v1;
    }
    result[1] = log_sum_exp(tmp.begin(), tmp.begin() + alphabet_size);

    /* Handle remaining i \notin {0, 1}, non-missing cases */
    double tmp2[] = {0.0, 0.0};
    for (size_t i = 2; i < alphabet_size; i++) {
        tmp2[0] = log_vector[i] - branch_length * nu;
        tmp2[1] = log_vector[0] + d.v2;
        result[i] = log_sum_exp(tmp2, tmp2 + 2);
    }
}

void laml_model::compute_log_pmatrix_transpose_vector_product(
    laml_data& d,
    size_t character, 
    double branch_length, 
    const std::vector<double>& log_vector,
    std::vector<double>& result
) const {
}

void laml_model::compute_taxa_log_inside_likelihood(
    laml_data& d,
    size_t character, 
    size_t taxa_id,
    std::vector<double>& result
) const {
    double phi = this->parameters[1];
    int state = this->character_matrix[taxa_id][character];
    
    std::fill(result.begin(), result.end(), NEGATIVE_INFINITY);

    if (state == -1) {
        result[0] = 0.0;
        for (size_t i = 1; i < this->alphabet_sizes[character]; i++) {
            result[i] = d.log_phi;
        }
    } else {
        result[state + 1] = d.log_one_minus_phi;
    }
};

void laml_model::compute_root_distribution(laml_data& d, size_t character, std::vector<double>& result) const {
    std::fill(result.begin(), result.end(), NEGATIVE_INFINITY);
    result[1] = 0.0; // root must start at unmutated state
};