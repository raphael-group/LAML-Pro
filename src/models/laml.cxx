#include <vector>
#include <limits>
#include "models/laml.h"
#include "math_utilities.h"

#include <iterator>
#include <algorithm>
#include <numeric>
#include <cmath>

#define NEGATIVE_INFINITY (-1e7) //-std::numeric_limits<double>::infinity())

void laml_model::compute_log_pmatrix_vector_product(
    laml_data& d,
    size_t character, 
    double branch_length, 
    const std::vector<double>& log_vector,
    std::vector<double>& result
) const {
    size_t alphabet_size = this->alphabet_sizes[character];
    double nu = this->parameters[0];

    std::vector<double>& tmp = *(d.buffer);
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
    size_t alphabet_size = this->alphabet_sizes[character];
    double nu = this->parameters[0];

    std::vector<double>& tmp = *(d.buffer);
    std::copy(log_vector.begin(), log_vector.begin() + alphabet_size, tmp.begin());

    /* Handle i = 0 which is state = ? case */
    tmp[0] = log_vector[0];
    for (size_t i = 1; i < alphabet_size; i++) {
        tmp[i] = log_vector[i] + d.v2;
    }
    result[0] = log_sum_exp(tmp.begin(), tmp.begin() + alphabet_size);

    /* Handle i = 1 which is state = 0 case */
    result[1] = -branch_length * (1 + nu) + log_vector[1];
    
    /* Handle remaining i \notin {0, 1}, non-missing cases */
    double tmp2[] = {0.0, 0.0};
    for (size_t i = 2; i < alphabet_size; i++) {
        tmp2[0] = log_vector[1] + log_mutation_priors[character][i - 2] - branch_length * nu + d.v1;
        tmp2[1] = log_vector[i] - branch_length * nu;
        result[i] = log_sum_exp(tmp2, tmp2 + 2);
    }
}

void laml_model::compute_taxa_log_inside_likelihood(
    laml_data& d,
    size_t character, 
    size_t taxa_id,
    std::vector<double>& result
) const {

    if (this->data_type == "character-matrix") {

        int state = this->character_matrix[taxa_id][character];
        
        std::fill(result.begin(), result.end(), NEGATIVE_INFINITY);

        if (state == -1) {
            result[0] = 0.0; // 0 corresponds to silenced state? log(1.0) = 0
            for (size_t i = 1; i < this->alphabet_sizes[character]; i++) {
                result[i] = d.log_phi;
            }
        } else {
            result[state + 1] = d.log_one_minus_phi;
        }
    } else {
        const std::vector<double>& probs = this->observation_matrix[taxa_id][character];  // now a tensor

        std::fill(result.begin(), result.end(), NEGATIVE_INFINITY);
        double sum = std::accumulate(probs.begin(), probs.end(), 0.0);

        if (sum == 0.0) { // Missing observed data case
            result[0] = 0.0;
            for (size_t i = 0; i < this->alphabet_sizes[character]; ++i) {
                result[i+1] = d.log_phi;
            }
         } else {
             // observed data is not missing, result[0] corresponds to missing latent state, initialized to NEGATIVE_INFINITY
             for (size_t i = 0; i < probs.size(); ++i) {
                // result[i] = probs[i] + d.log_one_minus_phi; // assumes precomputed
                result[i+1] = std::log(probs[i]) + d.log_one_minus_phi; // fills in 1-4
            }
        }
    }
};

void laml_model::compute_root_distribution(laml_data& d, size_t character, std::vector<double>& result) const {
    std::fill(result.begin(), result.end(), NEGATIVE_INFINITY);
    result[1] = 0.0; // root must start at unmutated state
};
