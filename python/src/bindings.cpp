#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "tree_conversion.h"
#include "numpy_conversion.h"

#include "phylogeny.h"
#include "models/laml.h"
#include "laml_em.h"
#include "ultrametric.h"
#include "constants.h"

namespace py = pybind11;

/**
 * Result object returned from optimization.
 */
struct PyEMResults {
    double log_likelihood;
    int num_iterations;
    py::dict optimized_tree;
    double nu;
    double phi;
    py::array_t<double> posterior_probabilities;
};

/**
 * Generate uniform mutation priors for a character matrix.
 */
std::vector<std::vector<double>> generate_uniform_priors(
    const std::vector<std::vector<int>>& char_matrix
) {
    size_t num_chars = char_matrix[0].size();

    // Find max state for each character
    std::vector<int> max_states(num_chars, 0);
    for (const auto& row : char_matrix) {
        for (size_t j = 0; j < num_chars; j++) {
            if (row[j] > max_states[j]) {
                max_states[j] = row[j];
            }
        }
    }

    // Create uniform priors
    std::vector<std::vector<double>> priors(num_chars);
    for (size_t j = 0; j < num_chars; j++) {
        int num_mutated_states = max_states[j];  // states 1..max are mutated
        if (num_mutated_states > 0) {
            double prob = 1.0 / num_mutated_states;
            priors[j] = std::vector<double>(num_mutated_states, prob);
        } else {
            priors[j] = std::vector<double>(1, 1.0);  // fallback
        }
    }

    return priors;
}

/**
 * Run EM optimization on a tree with character data.
 */
PyEMResults run_optimization(
    py::dict tree_dict,
    py::object character_matrix_obj,
    py::object observation_matrix_obj,
    py::object mutation_priors_obj,
    double initial_nu,
    double initial_phi,
    bool ultrametric,
    int max_iterations,
    double min_branch_length,
    bool verbose
) {
    // Convert tree
    tree t = dict_to_tree(tree_dict);

    // Determine data type and convert matrices
    std::string data_type;
    std::vector<std::vector<int>> char_matrix;
    std::vector<std::vector<std::vector<double>>> obs_matrix;

    if (!character_matrix_obj.is_none()) {
        data_type = "character-matrix";
        char_matrix = numpy_to_character_matrix(
            character_matrix_obj.cast<py::array_t<int32_t>>()
        );
    } else if (!observation_matrix_obj.is_none()) {
        data_type = "observation-matrix";
        obs_matrix = numpy_to_observation_matrix(
            observation_matrix_obj.cast<py::array_t<double>>()
        );
    } else {
        throw std::invalid_argument("Must provide either character_matrix or observation_matrix");
    }

    // Convert or generate mutation priors
    std::vector<std::vector<double>> mutation_priors;
    if (!mutation_priors_obj.is_none()) {
        mutation_priors = numpy_to_mutation_priors(
            mutation_priors_obj.cast<py::array_t<double>>()
        );
    } else {
        if (data_type == "character-matrix") {
            mutation_priors = generate_uniform_priors(char_matrix);
        } else {
            // For observation matrix, generate uniform priors based on state count
            size_t num_chars = obs_matrix[0].size();
            mutation_priors.resize(num_chars);
            for (size_t j = 0; j < num_chars; j++) {
                size_t num_states = obs_matrix[0][j].size();
                if (num_states > 1) {
                    double prob = 1.0 / (num_states - 1);  // exclude unedited state
                    mutation_priors[j] = std::vector<double>(num_states - 1, prob);
                } else {
                    mutation_priors[j] = std::vector<double>(1, 1.0);
                }
            }
        }
    }

    // Create model
    laml_model model(
        char_matrix,
        obs_matrix,
        mutation_priors,
        initial_nu,
        initial_phi,
        data_type,
        ultrametric,
        min_branch_length,
        1.0  // timescale
    );

    // Run EM
    em_results results = laml_expectation_maximization(t, model, max_iterations, verbose);

    // Build result object
    PyEMResults py_results;
    py_results.log_likelihood = results.log_likelihood;
    py_results.num_iterations = results.num_iterations;
    py_results.optimized_tree = tree_to_dict(t);
    py_results.nu = model.parameters[0];
    py_results.phi = model.parameters[1];
    py_results.posterior_probabilities = posterior_to_numpy(results.posterior_llh);

    return py_results;
}

/**
 * Compute log-likelihood without optimization.
 */
double compute_log_likelihood(
    py::dict tree_dict,
    py::object character_matrix_obj,
    py::object observation_matrix_obj,
    py::object mutation_priors_obj,
    double nu,
    double phi
) {
    // Convert tree
    tree t = dict_to_tree(tree_dict);

    // Determine data type and convert matrices
    std::string data_type;
    std::vector<std::vector<int>> char_matrix;
    std::vector<std::vector<std::vector<double>>> obs_matrix;

    if (!character_matrix_obj.is_none()) {
        data_type = "character-matrix";
        char_matrix = numpy_to_character_matrix(
            character_matrix_obj.cast<py::array_t<int32_t>>()
        );
    } else if (!observation_matrix_obj.is_none()) {
        data_type = "observation-matrix";
        obs_matrix = numpy_to_observation_matrix(
            observation_matrix_obj.cast<py::array_t<double>>()
        );
    } else {
        throw std::invalid_argument("Must provide either character_matrix or observation_matrix");
    }

    // Convert or generate mutation priors
    std::vector<std::vector<double>> mutation_priors;
    if (!mutation_priors_obj.is_none()) {
        mutation_priors = numpy_to_mutation_priors(
            mutation_priors_obj.cast<py::array_t<double>>()
        );
    } else {
        if (data_type == "character-matrix") {
            mutation_priors = generate_uniform_priors(char_matrix);
        } else {
            size_t num_chars = obs_matrix[0].size();
            mutation_priors.resize(num_chars);
            for (size_t j = 0; j < num_chars; j++) {
                size_t num_states = obs_matrix[0][j].size();
                if (num_states > 1) {
                    double prob = 1.0 / (num_states - 1);
                    mutation_priors[j] = std::vector<double>(num_states - 1, prob);
                } else {
                    mutation_priors[j] = std::vector<double>(1, 1.0);
                }
            }
        }
    }

    // Create model
    laml_model model(
        char_matrix,
        obs_matrix,
        mutation_priors,
        nu,
        phi,
        data_type,
        false,  // ultrametric
        0.01,   // min_branch_length
        1.0     // timescale
    );

    // Compute likelihood
    size_t num_characters = model.alphabet_sizes.size();
    size_t max_alphabet_size = *std::max_element(
        model.alphabet_sizes.begin(),
        model.alphabet_sizes.end()
    );

    likelihood_buffer inside_ll(num_characters, max_alphabet_size, t.num_nodes);
    std::vector<double> internal_buffer(max_alphabet_size);
    auto model_data = model.initialize_data(t.tree, t.branch_lengths, &internal_buffer);

    auto likelihoods = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);

    double total_llh = 0.0;
    for (double llh : likelihoods) {
        total_llh += llh;
    }

    return total_llh;
}

/**
 * Project tree branch lengths to ultrametric.
 */
py::dict project_to_ultrametric(py::dict tree_dict) {
    tree t = dict_to_tree(tree_dict);
    ultrametric_projection(t);
    return tree_to_dict(t);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Python bindings for fastLAML - Fast Lineage-Aware Maximum Likelihood";

    // EMResults class
    py::class_<PyEMResults>(m, "EMResults")
        .def_readonly("log_likelihood", &PyEMResults::log_likelihood,
            "Final log-likelihood value")
        .def_readonly("num_iterations", &PyEMResults::num_iterations,
            "Number of EM iterations performed")
        .def_readonly("optimized_tree", &PyEMResults::optimized_tree,
            "Tree dict with optimized branch lengths")
        .def_readonly("nu", &PyEMResults::nu,
            "Optimized nu parameter (mutation rate)")
        .def_readonly("phi", &PyEMResults::phi,
            "Optimized phi parameter (dropout rate)")
        .def_readonly("posterior_probabilities", &PyEMResults::posterior_probabilities,
            "Posterior probabilities array (characters, nodes, states)");

    // Main optimization function
    m.def("optimize", &run_optimization,
        py::arg("tree"),
        py::arg("character_matrix") = py::none(),
        py::arg("observation_matrix") = py::none(),
        py::arg("mutation_priors") = py::none(),
        py::arg("initial_nu") = 0.5,
        py::arg("initial_phi") = 0.5,
        py::arg("ultrametric") = false,
        py::arg("max_iterations") = 100,
        py::arg("min_branch_length") = 0.01,
        py::arg("verbose") = false,
        R"pbdoc(
            Run EM optimization on a phylogenetic tree.

            Parameters
            ----------
            tree : dict
                Tree structure with keys: num_leaves, num_nodes, root, edges, branch_lengths
            character_matrix : numpy.ndarray, optional
                Character state matrix with shape (leaves, characters), dtype int32.
                Use -1 for missing data.
            observation_matrix : numpy.ndarray, optional
                Observation probability matrix with shape (leaves, characters, states).
            mutation_priors : numpy.ndarray, optional
                Prior probabilities for mutations with shape (characters, states).
                If not provided, uniform priors are used.
            initial_nu : float
                Initial value for nu parameter (mutation rate). Default 0.5.
            initial_phi : float
                Initial value for phi parameter (dropout rate). Default 0.5.
            ultrametric : bool
                Whether to constrain tree to be ultrametric. Default False.
            max_iterations : int
                Maximum number of EM iterations. Default 100.
            min_branch_length : float
                Minimum branch length fraction for ultrametric trees. Default 0.01.
            verbose : bool
                Whether to print progress information. Default False.

            Returns
            -------
            EMResults
                Object containing optimization results.
        )pbdoc");

    // Likelihood computation
    m.def("compute_likelihood", &compute_log_likelihood,
        py::arg("tree"),
        py::arg("character_matrix") = py::none(),
        py::arg("observation_matrix") = py::none(),
        py::arg("mutation_priors") = py::none(),
        py::arg("nu") = 0.5,
        py::arg("phi") = 0.5,
        R"pbdoc(
            Compute log-likelihood of a tree given character data.

            Parameters
            ----------
            tree : dict
                Tree structure with keys: num_leaves, num_nodes, root, edges, branch_lengths
            character_matrix : numpy.ndarray, optional
                Character state matrix with shape (leaves, characters), dtype int32.
            observation_matrix : numpy.ndarray, optional
                Observation probability matrix with shape (leaves, characters, states).
            mutation_priors : numpy.ndarray, optional
                Prior probabilities for mutations.
            nu : float
                Nu parameter (mutation rate). Default 0.5.
            phi : float
                Phi parameter (dropout rate). Default 0.5.

            Returns
            -------
            float
                Log-likelihood value.
        )pbdoc");

    // Ultrametric projection
    m.def("project_ultrametric", &project_to_ultrametric,
        py::arg("tree"),
        R"pbdoc(
            Project tree branch lengths to satisfy ultrametric constraint.

            Parameters
            ----------
            tree : dict
                Tree structure.

            Returns
            -------
            dict
                Tree with projected branch lengths.
        )pbdoc");

    // Constants
    m.attr("BRANCH_LENGTH_LB") = BRANCH_LENGTH_LB;
    m.attr("BRANCH_LENGTH_UB") = BRANCH_LENGTH_UB;
    m.attr("NU_LB") = NU_LB;
    m.attr("NU_UB") = NU_UB;
    m.attr("PHI_LB") = PHI_LB;
    m.attr("PHI_UB") = PHI_UB;

    m.attr("__version__") = "0.1.0";
}
