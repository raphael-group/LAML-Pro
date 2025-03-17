#include "phylogeny.h"
#include "models/laml.h"

double laml_expectation_maximization(
    tree& t, 
    laml_model& model,
    bool verbose = false
);