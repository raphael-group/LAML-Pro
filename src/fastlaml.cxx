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

#include "phylogenetic_model.h"
#include "phylogeny.h"
#include "models/laml.h"

#include <argparse/argparse.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/fmt/ostr.h>

#define FASTLAML_VERSION_MAJOR 1
#define FASTLAML_VERSION_MINOR 0

void laml_expectation_step(
    const tree& t,
    const laml_model& model,
    const std::vector<double>& likelihoods,
    const likelihood_buffer& inside_ll,
    const likelihood_buffer& outside_ll,
    const likelihood_buffer& edge_inside_ll,
    const std::vector<laml_data>& node_data,
    std::vector<std::array<double, 6>>& responsibilities
) {
    /* 
      Order of Responsibilities:
        0: C_zero_zero
        1: C_zero_alpha 
        2: C_zero_miss
        3: C_alpha_alpha
        4: C_alpha_miss
        5: C_miss_miss
    */

    double nu = model.parameters[0];
    double phi = model.parameters[1];

    for (size_t v_id = 0; v_id < t.num_nodes; ++v_id) {
        for (int i = 0; i < 6; i++) responsibilities[v_id][i] = 0.0;
    }

    size_t num_characters = inside_ll.num_characters;
    for (size_t character = 0; character < num_characters; ++character) {
        size_t alphabet_size = model.alphabet_sizes[character];
        std::vector<double> tmp_buffer(alphabet_size - 2, 0.0);

        for (size_t v_id = 0; v_id < t.num_nodes; ++v_id) {
            if (t.tree.in_degree(v_id) == 0) {
                continue;
            }

            size_t v = t.tree[v_id].data;
            
            size_t u_id = t.tree.predecessors(v_id)[0]; 
            size_t u = t.tree[u_id].data;

            size_t w_id = t.tree.successors(u_id)[0];
            if (v_id == w_id) {
                w_id = t.tree.successors(u_id)[1];
            }

            size_t w = t.tree[w_id].data; // w is the sibling of v, u is the parent of v
            double blen = t.branch_lengths[v];
            double log_C_zero_zero = outside_ll(character, u, 1) + edge_inside_ll(character, w, 1) 
                                   + inside_ll(character, v, 1) - blen * (1.0 + nu)
                                   - likelihoods[character];
            double log_C_zero_miss = outside_ll(character, u, 1) + edge_inside_ll(character, w, 1) 
                                   + inside_ll(character, v, 0) + node_data[v].v2
                                   - likelihoods[character];
            double log_C_miss_miss = outside_ll(character, u, 0) + edge_inside_ll(character, w, 0) 
                                   + inside_ll(character, v, 0) - likelihoods[character];

            // compute log_C_zero_alpha
            for (size_t j = 0; j < alphabet_size - 2; j++) {
                tmp_buffer[j] = outside_ll(character, u, 1) + edge_inside_ll(character, w, 1) + inside_ll(character, v, j + 2);
                tmp_buffer[j] += model.log_mutation_priors[character][j] + node_data[v].v1 - blen * nu;
                tmp_buffer[j] -= likelihoods[character];
            }
            double log_C_zero_alpha = log_sum_exp(tmp_buffer.begin(), tmp_buffer.end());

            // compute log_C_alpha_alpha
            for (size_t j = 0; j < alphabet_size - 2; j++) {
                tmp_buffer[j] = outside_ll(character, u, j + 2) + edge_inside_ll(character, w, j + 2) + inside_ll(character, v, j + 2) - blen * nu;
                tmp_buffer[j] -= likelihoods[character];
            }
            double log_C_alpha_alpha = log_sum_exp(tmp_buffer.begin(), tmp_buffer.end());

            // compute log_C_alpha_miss
            for (size_t j = 0; j < alphabet_size - 2; j++) {
                tmp_buffer[j] = outside_ll(character, u, j + 2) + edge_inside_ll(character, w, j + 2) + inside_ll(character, v, 0) + node_data[v].v2;
                tmp_buffer[j] -= likelihoods[character];
            }
            double log_C_alpha_miss = log_sum_exp(tmp_buffer.begin(), tmp_buffer.end());

            responsibilities[v_id][0] += fast_exp(log_C_zero_zero);
            responsibilities[v_id][1] += fast_exp(log_C_zero_alpha);
            responsibilities[v_id][2] += fast_exp(log_C_zero_miss);
            responsibilities[v_id][3] += fast_exp(log_C_alpha_alpha);
            responsibilities[v_id][4] += fast_exp(log_C_alpha_miss);
            responsibilities[v_id][5] += fast_exp(log_C_miss_miss);
        }
    }
}

double laml_expectation_maximization(tree t, phylogeny_data data, double initial_nu, double initial_phi) {
    laml_model model(t.tree, data.character_matrix, data.mutation_priors, initial_nu, initial_phi);

    likelihood_buffer inside_ll(data.num_characters, data.max_alphabet_size + 2, t.num_nodes);
    likelihood_buffer edge_inside_ll(data.num_characters, data.max_alphabet_size + 2, t.num_nodes);
    likelihood_buffer outside_ll(data.num_characters, data.max_alphabet_size + 2, t.num_nodes);

    std::vector<double> internal_comp_buffer(data.max_alphabet_size + 2);
    auto model_data = model.initialize_data(&internal_comp_buffer, t.branch_lengths);
    auto likelihood = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);
    phylogeny::compute_edge_inside_log_likelihood(model, t, inside_ll, edge_inside_ll, model_data);
    phylogeny::compute_outside_log_likelihood(model, t, edge_inside_ll, outside_ll, model_data);

    std::vector<std::array<double, 6>> responsibilities(t.num_nodes);
    laml_expectation_step(t, model, likelihood, inside_ll, outside_ll, edge_inside_ll, model_data, responsibilities);

    return 0.0;
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
    double llh = laml_expectation_maximization(t, data, 0.5, 0.5);
    auto end = std::chrono::high_resolution_clock::now();
    double runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    spdlog::info("Log likelihood: {}", llh);
    spdlog::info("Computation time: {} ms", runtime);

    return 0;
}
