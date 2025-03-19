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

template <typename T>
digraph<T> arbitrarily_resolve_polytomies(
    digraph<T> tree
) {
    digraph<T> binary_tree = tree;
    int num_nodes = binary_tree.nodes().size();
    int next_id = num_nodes;
    
    for(int node_id = 0; node_id < num_nodes; ++node_id) {
        std::vector<int> children = binary_tree.successors(node_id);
        
        if(children.size() > 2) {
            for(int child : children) {
                binary_tree.remove_edge(node_id, child);
            }
            
            binary_tree.add_edge(node_id, children[0]);
            
            int current_parent = next_id++;
            int current_parent_id = binary_tree.add_vertex(current_parent);
            binary_tree.add_edge(node_id, current_parent_id);
            binary_tree.add_edge(current_parent_id, children[1]);
            
            for(size_t i = 2; i < children.size(); ++i) {
                int new_node = next_id++;
                int new_node_id = binary_tree.add_vertex(new_node);
                binary_tree.add_edge(current_parent_id, new_node_id);
                binary_tree.add_edge(new_node_id, children[i]);
                current_parent_id = new_node_id;
            }
        }
    }
    
    return binary_tree;
}

struct nni { // swap subtrees rooted at u and v
    int u;
    int v;
};

struct nni_thread_data {
    tree t;
    laml_model model;
    std::vector<nni> nni_moves;
};

/* 
    tree t is binary and t and model are not modified.
*/
std::vector<std::pair<nni, double>> evaluate_nnis(
    tree& t,
    laml_model& model,
    const std::vector<nni>& nni_moves,
    std::atomic<int>& nni_counter,
    int total_nni_moves
) {
    std::vector<std::pair<nni, double>> evaluations;
    for (const nni& move : nni_moves) {
        if (nni_counter % 50 == 1) {
            spdlog::info("Evaluated {}/{} NNI moves", nni_counter.load(), total_nni_moves);
        }

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

        auto em_result = laml_expectation_maximization(t, model, 1);
        evaluations.push_back({move, em_result.log_likelihood});

        // revert NNI move
        t.tree.remove_edge(parent_u, v);
        t.tree.remove_edge(parent_v, u);
        t.tree.add_edge(parent_u, u);
        t.tree.add_edge(parent_v, v);

        // revert model parameters
        t.branch_lengths = branch_lengths_copy;
        model.parameters = params_copy;
        nni_counter++;
    }

    return evaluations;
}

std::vector<std::pair<nni, double>> evaluate_nni_neighborhood(
    const phylogeny_data& data,
    const tree& initial_tree, // tree MUST be binary,
    int threads = 8
) {
    // compute initial likelihood and parameter estimates
    tree t                  = initial_tree;
    laml_model model        = laml_model(data.character_matrix, data.mutation_priors, 0.5, 0.5);
    auto initial_em_results = laml_expectation_maximization(t, model, 100, true);
    spdlog::info("Initial log likelihood: {}", initial_em_results.log_likelihood);

    std::vector<nni> nni_moves;
    spdlog::info("Root ID: {}", t.root_id);
    for (size_t node_id = 0; node_id < t.num_nodes; ++node_id) {
        if (node_id == t.root_id || t.tree.out_degree(node_id) == 0) { // skip root and leaves
            continue;
        }

        int p_id = t.tree.predecessors(node_id)[0];
        int w_id = t.tree.successors(p_id)[0];
        if (w_id == (int) node_id) {
            w_id = t.tree.successors(p_id)[1];
        }

        for (int u_id : t.tree.successors(node_id)) {
            nni_moves.push_back({w_id, u_id});
        }
    }

    std::vector<std::pair<nni, double>> neighborhood;
    std::atomic<int> nni_counter(0);

    if (threads <= 1) {
        neighborhood = evaluate_nnis(t, model, nni_moves, nni_counter, nni_moves.size());
    } else {
        std::vector<std::thread> thread_pool;
        std::vector<std::vector<nni>> thread_nni_moves(threads);
        std::vector<std::vector<std::pair<nni, double>>> thread_results(threads);

        
        for (size_t i = 0; i < nni_moves.size(); ++i) {
            thread_nni_moves[i % threads].push_back(nni_moves[i]);
        }
        
        for (int i = 0; i < threads; ++i) {
            thread_pool.emplace_back([&, i]() {
                tree thread_tree = t;
                laml_model thread_model = model;
                thread_results[i] = evaluate_nnis(thread_tree, thread_model, thread_nni_moves[i], nni_counter, nni_moves.size());
            });
        }
        
        for (auto& thread : thread_pool) {
            thread.join();
        }
        
        for (const auto& result : thread_results) {
            neighborhood.insert(neighborhood.end(), result.begin(), result.end());
        }
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

    program.add_argument("--threads")
        .help("number of threads to use")
        .default_value(std::thread::hardware_concurrency())
        .scan<'u', unsigned int>();

    program.add_argument("--mode")
        .help("Operation mode: 'optimize' for parameter optimization or 'search' for tree search")
        .default_value(std::string("optimize"))
        .choices("optimize", "search");

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
        spdlog::info("Input tree is not binary. Arbitrarily resolving polytomies.");
        t.tree = arbitrarily_resolve_polytomies(t.tree);
    }

    spdlog::info("Processing character matrix and mutation priors...");

    phylogeny_data data = process_phylogeny_data(
        t, 
        program.get<std::string>("--character-matrix"),
        program.get<std::string>("--mutation-priors")
    );
    
    if (program.get<std::string>("mode") == "optimize") {
        spdlog::info("Optimizing parameters...");

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.05f, 0.95f);
        
        double initial_phi = dist(gen);
        double initial_nu = dist(gen);

        for (size_t i = 0; i < t.branch_lengths.size(); ++i) {
            t.branch_lengths[i] = dist(gen);
        }

        laml_model model = laml_model(data.character_matrix, data.mutation_priors,initial_phi, initial_nu);
        laml_expectation_maximization(t, model, 100, true);
    } else {
        spdlog::info("Searching for optimal tree...");

        for (size_t i = 0; i < t.branch_lengths.size(); ++i) {
            t.branch_lengths[i] = 0.01 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.4-0.01)));
        }

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::pair<nni, double>> neighborhood = evaluate_nni_neighborhood(data, t, program.get<unsigned int>("--threads"));
        auto end = std::chrono::high_resolution_clock::now();
        double runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        spdlog::info("Computation time: {} ms", runtime);
    }    

    return 0;
}