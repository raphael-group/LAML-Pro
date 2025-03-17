#include "phylogeny.h"
#include "models/laml.h"

struct em_results {
    double log_likelihood;
    int num_iterations;
};

em_results laml_expectation_maximization(
    tree& t, 
    laml_model& model,
    int max_em_iterations = 100,
    bool verbose = false
);