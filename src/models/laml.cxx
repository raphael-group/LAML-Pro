#include <vector>
#include "models/laml.h"

std::vector<double> laml_model::compute_log_pmatrix_vector_product(
    size_t character, 
    double branch_length, 
    const std::vector<double>& log_vector
) const {
    std::vector<double> result(this->alphabet_sizes[character], -1.0);
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
    std::vector<double> result(this->alphabet_sizes[character], -1.0);
    return result;
};

std::vector<double> laml_model::compute_root_distribution(size_t character) const {
    std::vector<double> result(this->alphabet_sizes[character], -1.0);
    return result;
}
