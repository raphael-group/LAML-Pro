#ifndef LAML_MODEL_H
#define LAML_MODEL_H

#include "../digraph.h"
#include "../phylogenetic_model.h"

/*
* In the LAML model, we order the alphabet as {-1, 0, 1, 2, ...} where -1 is 
* the missing data state, 0 is the unmutated state, and the remaining states
* are the mutated states.
*/
class laml_model : public phylogenetic_model {
    private:
    std::vector<double> tmp_buffer; // tmp LSE array

    public:
    digraph<size_t> tree;
    std::vector<std::vector<int>> character_matrix;   // [leaf_id][character]
    std::vector<std::vector<double>> mutation_priors; // [character][state]
    std::vector<std::vector<double>> log_mutation_priors; // [character][state]

    laml_model(
        const digraph<size_t>& tree,
        const std::vector<std::vector<int>>& character_matrix,
        const std::vector<std::vector<double>>& mutation_priors,
        double nu,
        double phi
    ) : tree(tree), character_matrix(character_matrix), mutation_priors(mutation_priors) {
        parameters = {nu, phi}; // nu, phi
        alphabet_sizes = std::vector<size_t>(character_matrix[0].size());
        size_t max_alphabet_size = 0;
        for (size_t i = 0; i < alphabet_sizes.size(); i++) {
            int alphabet_size = 0;
            for (size_t j = 0; j < character_matrix.size(); j++) {
                alphabet_size = std::max(alphabet_size, character_matrix[j][i]);
            }

            alphabet_sizes[i] = alphabet_size + 2;
            max_alphabet_size = std::max(max_alphabet_size, alphabet_sizes[i]);
        }

        log_mutation_priors = std::vector<std::vector<double>>(mutation_priors.size());
        for (size_t i = 0; i < mutation_priors.size(); i++) {
            log_mutation_priors[i] = std::vector<double>(mutation_priors[i].size());
            for (size_t j = 0; j < mutation_priors[i].size(); j++) {
                log_mutation_priors[i][j] = std::log(mutation_priors[i][j]);
            }
        }

        tmp_buffer = std::vector<double>(max_alphabet_size, 0.0);
    }

    void compute_log_pmatrix_vector_product(size_t character, double branch_length, const std::vector<double>& log_vector, std::vector<double>& result) override;
    void compute_log_pmatrix_transpose_vector_product(size_t character, double branch_length, const std::vector<double>& log_vector, std::vector<double>& result) override;
    void compute_taxa_log_inside_likelihood(size_t character, size_t taxa_id, std::vector<double>& result) override;
    void compute_root_distribution(size_t character, std::vector<double>& result) override;
};

#endif