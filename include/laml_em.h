#include "phylogeny.h"
#include "models/laml.h"

struct em_results {
    double log_likelihood;
    int num_iterations;
};

void laml_expectation_step(
    const tree& t,
    const laml_model& model,
    const std::vector<double>& likelihoods,
    const likelihood_buffer& inside_ll,
    const likelihood_buffer& outside_ll,
    const likelihood_buffer& edge_inside_ll,
    std::vector<laml_data>& node_data,
    std::vector<std::array<double, 6>>& responsibilities,
    double& leaf_responsibility
);

em_results laml_expectation_maximization(
    tree& t, 
    laml_model& model,
    int max_em_iterations = 100,
    bool verbose = false
);

void print_likelihood_buffer(const std::string& label,
                             const likelihood_buffer& buf,
                             size_t num_characters,
                             size_t max_alphabet_size,
                             size_t num_nodes);

