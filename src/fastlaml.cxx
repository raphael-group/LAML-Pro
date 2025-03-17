#include <cstdlib>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <stdexcept>
#include <chrono>

#include "digraph.h"
#include "io.h"

#include "laml_em.h"
#include "phylogenetic_model.h"
#include "phylogeny.h"
#include "models/laml.h"

#include <argparse/argparse.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/fmt/ostr.h>

#define FASTLAML_VERSION_MAJOR 1
#define FASTLAML_VERSION_MINOR 0

struct nni { // swap subtrees rooted at u and v
    int u;
    int v;
};

std::vector<std::pair<nni, double>> evaluate_nni_neighborhood(
    const phylogeny_data& data,
    const tree& initial_tree, // tree MUST be binary,
    int threads = 1
) {
    // compute initial likelihood and parameter estimates
    tree t                  = initial_tree;
    laml_model model        = laml_model(data.character_matrix, data.mutation_priors, 0.5, 0.5);
    auto initial_em_results = laml_expectation_maximization(t, model);
    spdlog::info("Initial log likelihood: {}", initial_em_results.log_likelihood);
    std::vector<nni> nni_moves;
    for (int node_id = 0; node_id < t.num_nodes; ++node_id) {
        if (node_id == t.root_id || t.tree.out_degree(node_id) == 0) { // skip root and leaves
            continue;
        }

        int p_id = t.tree.predecessors(node_id)[0];
        int w_id = t.tree.successors(p_id)[0];
        if (w_id == node_id) {
            w_id = t.tree.successors(p_id)[1];
        }

        for (int u_id : t.tree.successors(node_id)) {
            nni_moves.push_back({node_id, u_id});
        }
    }

    // invariant: 
    //   t does not change. specifically, at end of each iteration, 
    //   t is the initial tree
    std::vector<std::pair<nni, double>> neighborhood;
    for (const nni& move : nni_moves) {
        auto branch_lengths_copy = t.branch_lengths;
        auto params_copy = model.parameters;

        // perform NNI move
        auto [u, v] = move;
        int parent_u = t.tree.predecessors(u)[0];
        int parent_v = t.tree.predecessors(v)[0];

        t.tree.remove_edge(parent_u, u);
        t.tree.remove_edge(parent_v, v);
        t.tree.add_edge(parent_u, v);
        t.tree.add_edge(parent_v, u);

        auto em_result = laml_expectation_maximization(t, model);
        neighborhood.push_back({move, em_result.log_likelihood});

        spdlog::info("NNI move: ({}, {}) -> ({}, {}), LLH: {}, EM Iterations: {}", 
            parent_u, u, parent_v, v, em_result.log_likelihood, em_result.num_iterations);

        // revert NNI move
        t.tree.remove_edge(parent_u, v);
        t.tree.remove_edge(parent_v, u);
        t.tree.add_edge(parent_u, u);
        t.tree.add_edge(parent_v, v);

        // revert model parameters
        t.branch_lengths = branch_lengths_copy;
        model.parameters = params_copy;
    }

    return neighborhood;
}

int main(int argc, char ** argv) {
    auto console_logger = spdlog::stdout_color_mt("fastlaml");
    auto error_logger = spdlog::stderr_color_mt("error");
    spdlog::set_default_logger(console_logger);

    argparse::ArgumentParser program(
        "fastlaml",
        std::to_string(FASTLAML_VERSION_MAJOR) + "." + std::to_string(FASTLAML_VERSION_MINOR),
        argparse::default_arguments::help
    );

    program.add_argument("--version")
        .action([&](const auto & /*unused*/) {
            std::cout << "fastppm version " << FASTLAML_VERSION_MAJOR << "." << FASTLAML_VERSION_MINOR << std::endl;
            std::exit(0);
        })
        .default_value(false)
        .help("prints version information and exits")
        .implicit_value(true)
        .nargs(0);

    program.add_argument("-m", "--mutation-priors")
        .help("Path to the mutation priors file")
        .default_value(std::string(""));

    program.add_argument("-c", "--character-matrix")
        .help("Path to the character matrix file (CSV)")
        .required();

    program.add_argument("-t", "--tree")
        .help("Path to the tree file")
        .required();

    program.add_argument("-o", "--output")
        .help("Path to the output file")
        .required();

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    spdlog::info("Loading tree from Newick file...");
    tree t = parse_newick_tree(program.get<std::string>("--tree"));
    
    // Check if tree is binary
    bool is_binary = true;
    for (size_t node_id = 0; node_id < t.num_nodes; ++node_id) {
        if (node_id != t.root_id && t.tree.in_degree(node_id) != 1) {
            is_binary = false;
            break;
        }
        
        size_t out_degree = t.tree.out_degree(node_id);
        if (out_degree != 0 && out_degree != 2) {
            is_binary = false;
            break;
        }
    }
    
    if (!is_binary) {
        spdlog::error("Input tree is not binary. Each node must have exactly 0 or 2 children.");
        std::exit(1);
    }

    spdlog::info("Processing character matrix and mutation priors...");

    phylogeny_data data = process_phylogeny_data(
        t, 
        program.get<std::string>("--character-matrix"),
        program.get<std::string>("--mutation-priors")
    );
    
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<nni, double>> neighborhood = evaluate_nni_neighborhood(data, t);
    auto end = std::chrono::high_resolution_clock::now();
    double runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    //spdlog::info("Log likelihood: {}", llh);
    spdlog::info("Computation time: {} ms", runtime);

    return 0;
}