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

#include "csv.hpp"
#include "digraph.h"
#include "compact_tree.h"

#include "phylogenetic_model.h"
#include "phylogeny.h"
#include "models/laml.h"

#include <argparse/argparse.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/fmt/ostr.h>

#define FASTLAML_VERSION_MAJOR 1
#define FASTLAML_VERSION_MINOR 0

/*! 
* @brief A struct to hold a rooted tree without attaching parameters.
*
* @param tree The tree as a directed graph, where the vertices are numbered from 0 to 
*             2N - 1 and the leaves are numbered from 0 to N - 1.
* @param branch_lengths Stores the branch length on the edge leading into each node.
* @param node_names Stores the name of each node.
* @param root_id The ID of the root vertex of the tree.
*/
struct tree {
    size_t num_leaves;
    size_t num_nodes;
    size_t root_id;   
    digraph<size_t> tree;
    std::vector<double> branch_lengths;
    std::vector<std::string> node_names;
};

/*!
 * @brief Data structure to hold character matrix, mutation priors, and associated metadata.
 */
struct phylogeny_data {
    size_t num_characters;
    size_t max_alphabet_size;
    std::vector<std::vector<int>> character_matrix; // [leaf_id][character]
    std::vector<std::vector<double>> mutation_priors; // [character][state]
};

/*!
* @brief Parse a Newick tree file into a Tree struct.
* @param fname The path to the Newick tree file.
* @return Tree The parsed tree.
*/
tree parse_newick_tree(std::string fname) {
    compact_tree tree(fname);

    size_t num_nodes = tree.get_num_nodes();
    size_t num_leaves = tree.get_num_leaves();
    std::vector<int> name_map(num_nodes, -1);

    digraph<size_t> g;
    std::vector<double> branch_lengths(num_nodes);
    std::vector<std::string> node_names(num_nodes);
    size_t leaf_idx = 0;
    size_t internal_idx = num_leaves;
    for (size_t i = 0; i < num_nodes; i++) {
        int j;
        if (tree.get_children(i).size() == 0) {
            j = leaf_idx++;
        } else {
            j = internal_idx++;
        }

        int id = g.add_vertex(j);
        branch_lengths[j] = tree.get_edge_length(i);
        node_names[j] = tree.get_label(i);
        name_map[i] = id;
    }

    int root_id = name_map[0];
    for (size_t i = 0; i < num_nodes; i++) {
        for (auto child : tree.get_children(i)) {
            g.add_edge(name_map[i], name_map[child]);
        }
    }

    if (root_id == -1) {
        throw std::runtime_error("Root node not found in tree");
    }

    return {num_leaves, num_nodes, (size_t) root_id, g, branch_lengths, node_names};
}

/*!
 * @brief Parse a character matrix from a CSV file.
 * @param filename Path to the CSV file
 * @return Pair of (taxa_names, matrix) where matrix[i][j] is the state for taxon i, character j
 */
std::pair<std::vector<std::string>, std::vector<std::vector<int>>> parse_character_matrix(const std::string& filename) {
    std::vector<std::string> taxa_names;
    std::vector<std::vector<int>> matrix;
    
    try {
        csv::CSVReader reader(filename);
        
        for (auto& row: reader) {
            std::string taxon_name = row[0].get<std::string>();
            taxa_names.push_back(taxon_name);
            
            std::vector<int> character_states;
            for (size_t i = 1; i < row.size(); i++) {
                if (row[i].is_null() || row[i].get<std::string>() == "?") {
                    character_states.push_back(-1); // Unknown data
                } else {
                    try {
                        character_states.push_back(row[i].get<int>());
                    } catch (const std::exception& e) {
                        throw std::runtime_error("Invalid character state in CSV: " + row[i].get<std::string>());
                    }
                }
            }
            matrix.push_back(character_states);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse character matrix file: " + std::string(e.what()));
    }
    
    if (taxa_names.empty()) {
        throw std::runtime_error("No data found in character matrix file");
    }
    
    return {taxa_names, matrix};
}

/*!
 * @brief Parse mutation priors from a CSV file.
 * @param filename Path to the CSV file
 * @return Vector of (character, state, probability) tuples
 */
std::vector<std::tuple<int, int, double>> parse_mutation_priors(const std::string& filename) {
    std::vector<std::tuple<int, int, double>> priors;
    
    try {
        csv::CSVFormat format;
        format.delimiter(',').quote('"').no_header();

        csv::CSVReader reader(filename, format);
        
        for (auto& row: reader) {
            if (row.size() < 3) {
                throw std::runtime_error("Invalid format in mutation priors file: expected at least 3 columns");
            }
            
            int character = row[0].get<int>();
            int state = row[1].get<int>();
            double probability = row[2].get<double>();
            
            priors.emplace_back(character, state, probability);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse mutation priors file: " + std::string(e.what()));
    }
    
    return priors;
}

/*!
 * @brief Generate uniform mutation priors if none are provided
 * @param character_matrix The character matrix
 * @param num_characters Number of characters
 * @return Vector of (character, state, probability) tuples
 */
std::vector<std::tuple<int, int, double>> generate_uniform_priors(
    const std::vector<std::vector<int>>& character_matrix, int num_characters) {
    
    std::vector<std::tuple<int, int, double>> priors;
    
    for (int c = 0; c < num_characters; ++c) {
        std::set<int> unique_states;
        for (const auto& row : character_matrix) {
            if (c < row.size() && row[c] > 0) {
                unique_states.insert(row[c]);
            }
        }
        
        double probability = 1.0 / unique_states.size();
        for (int state : unique_states) {
            priors.emplace_back(c, state, probability);
        }
    }
    
    return priors;
}

/*!
 * @brief Process character matrix and mutation priors to create a phylogeny_data object.
 * @param t The tree structure
 * @param character_matrix_file Path to the character matrix CSV
 * @param mutation_priors_file Optional path to the mutation priors CSV
 * @return Processed phylogeny_data
 */
phylogeny_data process_phylogeny_data(
    const tree& t, const std::string& character_matrix_file, 
    const std::string& mutation_priors_file = "") {
    
    auto [taxa_names, raw_matrix] = parse_character_matrix(character_matrix_file);
    
    size_t num_characters = 0;
    for (const auto& row : raw_matrix) {
        num_characters = std::max(num_characters, row.size());
    }
    
    spdlog::info("Found {} taxa and {} characters", taxa_names.size(), num_characters);
    
    std::vector<std::tuple<int, int, double>> raw_priors;
    if (!mutation_priors_file.empty()) {
        raw_priors = parse_mutation_priors(mutation_priors_file);
    } else {
        spdlog::info("No mutation priors provided. Assuming uniform priors.");
        raw_priors = generate_uniform_priors(raw_matrix, num_characters);
    }
    
    std::vector<std::vector<int>> character_matrix_recode(raw_matrix.size(), std::vector<int>(num_characters));
    std::vector<std::map<int, int>> original_to_new_mappings(num_characters);
    size_t max_alphabet_size = 0;
    
    for (size_t c = 0; c < num_characters; ++c) {
        std::set<int> valid_states;
        for (const auto& row : raw_matrix) {
            if (c < row.size() && row[c] > 0) {
                valid_states.insert(row[c]);
            }
        }
        
        int new_idx = 1;
        for (int orig_state : valid_states) {
            original_to_new_mappings[c][orig_state] = new_idx++;
        }
        
        if (valid_states.size() > max_alphabet_size) {
            max_alphabet_size = valid_states.size();
        }
        
        for (size_t i = 0; i < raw_matrix.size(); ++i) {
            if (c < raw_matrix[i].size()) {
                int orig_state = raw_matrix[i][c];
                if (orig_state > 0 && original_to_new_mappings[c].count(orig_state)) {
                    character_matrix_recode[i][c] = original_to_new_mappings[c][orig_state];
                } else if (orig_state == 0) {
                    character_matrix_recode[i][c] = 0;
                } else {
                    character_matrix_recode[i][c] = -1;
                }
            } else {
                character_matrix_recode[i][c] = -1;
            }
        }
    }
    
    std::unordered_map<std::string, size_t> taxa_name_to_idx;
    for (size_t i = 0; i < taxa_names.size(); ++i) {
        taxa_name_to_idx[taxa_names[i]] = i;
    }
    
    std::vector<std::vector<int>> reordered_matrix(t.num_leaves, std::vector<int>(num_characters));
    std::vector<bool> leaf_mapped(t.num_leaves, false);
    
    for (size_t node_id = 0; node_id < t.num_nodes; ++node_id) {
        if (t.tree.out_degree(node_id) == 0) {
            size_t leaf_id = t.tree[node_id].data;
            
            if (leaf_id >= t.num_leaves) {
                throw std::runtime_error("Leaf ID out of range: " + std::to_string(leaf_id));
            }
            
            const std::string& leaf_name = t.node_names[leaf_id];
            
            if (taxa_name_to_idx.find(leaf_name) == taxa_name_to_idx.end()) {
                throw std::runtime_error("Taxon name from tree not found in character matrix: " + leaf_name);
            }
            
            size_t orig_row = taxa_name_to_idx[leaf_name];
            
            for (size_t c = 0; c < num_characters; ++c) {
                reordered_matrix[leaf_id][c] = character_matrix_recode[orig_row][c];
            }
            
            leaf_mapped[leaf_id] = true;
        }
    }
    
    for (size_t i = 0; i < t.num_leaves; ++i) {
        if (!leaf_mapped[i]) {
            spdlog::warn("Leaf {} not mapped to any taxon in character matrix", i);
        }
    }
    
    std::vector<std::vector<double>> recoded_priors(num_characters, std::vector<double>(max_alphabet_size, 0.0));
    
    for (const auto& [character, orig_state, probability] : raw_priors) {
        if (character < 0 || character >= static_cast<int>(num_characters)) {
            throw std::runtime_error("Character index out of range in mutation priors: " + std::to_string(character));
        }
        
        if (orig_state <= 0) {
            continue; // skip states 0 and -1
        }
        
        auto& mapping = original_to_new_mappings[character];
        if (mapping.find(orig_state) == mapping.end()) {
            spdlog::warn("State {} for character {} not found in character matrix", orig_state, character);
            continue;
        }
        
        int new_state = mapping[orig_state] - 1;
        recoded_priors[character][new_state] = probability;
    }
    
    phylogeny_data result;
    result.num_characters = num_characters;
    result.max_alphabet_size = max_alphabet_size;
    result.character_matrix = reordered_matrix;
    result.mutation_priors = recoded_priors;
    
    return result;
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
    
    laml_model model(t.tree, data.character_matrix, data.mutation_priors, 0.5, 0.5);

    phylogeny phylo = phylogeny<laml_data>(
        data.character_matrix.size(), 
        t.num_nodes, 
        t.root_id, 
        t.tree, 
        t.branch_lengths
    );

    likelihood_buffer inside_ll(data.num_characters, data.max_alphabet_size + 2, t.num_nodes);
    likelihood_buffer edge_inside_ll(data.num_characters, data.max_alphabet_size + 2, t.num_nodes);
    likelihood_buffer outside_ll(data.num_characters, data.max_alphabet_size + 2, t.num_nodes);

    auto start = std::chrono::high_resolution_clock::now();

    int num_iters = 100;
    double llh;
    for (int i = 0; i < num_iters - 1; i++) {
        std::vector<double> internal_comp_buffer(data.max_alphabet_size + 2);
        auto model_data = model.initialize_data(&internal_comp_buffer, t.branch_lengths);
        llh = phylo.compute_inside_log_likelihood(model, inside_ll, model_data);
        phylo.compute_edge_inside_log_likelihood(model, inside_ll, edge_inside_ll, model_data);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / ((double) num_iters);

    spdlog::info("Log likelihood: {}", llh);
    spdlog::info("Computation time: {} ms", runtime);

    
    return 0;
}
