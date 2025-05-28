#ifndef LAML_MODEL_H
#define LAML_MODEL_H

#include "../digraph.h"
#include "../phylogenetic_model.h"

/* 
 * Precomputed per-node values and buffers in the LAML model. Avoids recomputation
 * of these values across characters, since they are constant for a given branch length.
 */
struct laml_data {
    std::vector<double> *buffer;
    double log_phi;
    double log_one_minus_phi;
    double v1;
    double v2;

    laml_data(std::vector<double> *buffer, double log_phi, double log_one_minus_phi, double v1, double v2) // copy 
        : buffer(buffer), log_phi(log_phi), log_one_minus_phi(log_one_minus_phi), v1(v1), v2(v2) {}
    laml_data() {} 
};

/*
* In the LAML model, we order the alphabet as {-1, 0, 1, 2, ...} where -1 is 
* the missing data state, 0 is the unmutated state, and the remaining states
* are the mutated states.
*/
class laml_model : public phylogenetic_model<laml_data> {
    public:
    std::vector<std::vector<int>> character_matrix;       // [leaf_id][character]
    std::vector<std::vector<std::vector<double>>> observation_matrix; // change
    std::vector<std::vector<double>> mutation_priors;     // [character][state]
    std::vector<std::vector<double>> log_mutation_priors; // [character][state]   
    std::string data_type; // change

    laml_model(
        const std::vector<std::vector<int>>& character_matrix,
        const std::vector<std::vector<std::vector<double>>> observation_matrix,
        const std::vector<std::vector<double>>& mutation_priors,
        double nu,
        double phi,
        const std::string data_type
    ) : character_matrix(character_matrix), observation_matrix(observation_matrix), mutation_priors(mutation_priors), data_type(data_type) {

        //std::cout << "[laml_model] beginning constructor"<< std::endl;

        parameters = {nu, phi}; // nu, phi
        if (data_type == "character-matrix") {
            alphabet_sizes = std::vector<size_t>(character_matrix[0].size());
            // for character i
            for (size_t i = 0; i < alphabet_sizes.size(); i++) {
                int alphabet_size = 0;
                // for cell j
                for (size_t j = 0; j < character_matrix.size(); j++) {
                    alphabet_size = std::max(alphabet_size, character_matrix[j][i]); // because the states were remapped
                }
                alphabet_sizes[i] = alphabet_size + 2;
            }

            log_mutation_priors = std::vector<std::vector<double>>(mutation_priors.size());
            for (size_t i = 0; i < mutation_priors.size(); i++) {
                log_mutation_priors[i] = std::vector<double>(mutation_priors[i].size());
                for (size_t j = 0; j < mutation_priors[i].size(); j++) {
                    if (mutation_priors[i][j] == 0.0) {
                        log_mutation_priors[i][j] = -1e12;
                    } else {
                        log_mutation_priors[i][j] = std::log(mutation_priors[i][j]);
                    }
                }
            }
        } else {
            // initialize vector to be of len num characters
            alphabet_sizes = std::vector<size_t>(observation_matrix[0].size()); 
            //std::cout << "[laml_model] Observation matrix (first cell, first character): " << observation_matrix[0][0].size() << std::endl;
            // for character i
            for (size_t i = 0; i < alphabet_sizes.size(); i++) {
                size_t alphabet_size = 0;
                for (size_t j = 0; j < observation_matrix.size(); j++) {
                    // includes unedited state
                    alphabet_size = std::max(alphabet_size, observation_matrix[j][i].size()); // size includes unedited state
                }
                alphabet_sizes[i] = alphabet_size + 1; // silenced state, but +1 for unedited to be a count
                                                       // (already includes) unedited and edited
            }

            //std::cout << "[laml_model] alphabet_sizes: " << alphabet_sizes[0] << std::endl;
            log_mutation_priors = std::vector<std::vector<double>>(mutation_priors.size());
            for (size_t i = 0; i < mutation_priors.size(); i++) {
                log_mutation_priors[i] = std::vector<double>(mutation_priors[i].size());
                for (size_t j = 0; j < mutation_priors[i].size(); j++) {
                    if (mutation_priors[i][j] == 0.0) {
                        log_mutation_priors[i][j] = -1e12;
                    } else {
                        log_mutation_priors[i][j] = std::log(mutation_priors[i][j]);
                    }
                }
            }
        }
    }

    void compute_log_pmatrix_vector_product(laml_data& d, size_t character, double branch_length, const std::vector<double>& log_vector, std::vector<double>& result) const override;
    void compute_log_pmatrix_transpose_vector_product(laml_data& d, size_t character, double branch_length, const std::vector<double>& log_vector, std::vector<double>& result) const override;
    void compute_taxa_log_inside_likelihood(laml_data& d, size_t character, size_t taxa_id, std::vector<double>& result) const override;
    void compute_root_distribution(laml_data& d, size_t character, std::vector<double>& result) const override;

    std::vector<laml_data> initialize_data(
        const digraph<size_t> &tree,
        const std::vector<double> &branch_lengths,
        std::vector<double> *buffer
    ) const {
        std::vector<laml_data> result(tree.size());
        for (size_t i = 0; i < tree.size(); ++i) {
            int node = tree[i].data;
            result[node] = laml_data(
                buffer,
                std::log(parameters[1]),
                std::log(1 - parameters[1]),
                std::log(1 - std::exp(-branch_lengths[node])),
                std::log(1 - std::exp(-parameters[0] * branch_lengths[node]))
            );
        }

        return result;
    }
};

#endif
