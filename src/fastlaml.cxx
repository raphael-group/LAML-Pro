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
#include "digraph.h"
#include "compact_tree.h"

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
    std::vector<std::string> character_names;
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
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open character matrix file: " + filename);
    }

    std::vector<std::string> taxa_names;
    std::vector<std::vector<int>> matrix;
    
    // Read header line to get character names
    std::string line;
    std::getline(file, line);
    
    // Parse each data line
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string taxon_name;
        std::getline(ss, taxon_name, ','); // First column is taxon name
        taxa_names.push_back(taxon_name);
        
        std::vector<int> character_states;
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            if (cell.empty() || cell == "?") {
                character_states.push_back(-1); // Unknown data
            } else {
                try {
                    character_states.push_back(std::stoi(cell));
                } catch (const std::exception& e) {
                    throw std::runtime_error("Invalid character state in CSV: " + cell);
                }
            }
        }
        matrix.push_back(character_states);
    }
    
    return {taxa_names, matrix};
}

/*!
 * @brief Parse mutation priors from a CSV file.
 * @param filename Path to the CSV file
 * @return Vector of (character, state, probability) tuples
 */
std::vector<std::tuple<int, int, double>> parse_mutation_priors(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open mutation priors file: " + filename);
    }
    
    std::vector<std::tuple<int, int, double>> priors;
    std::string line;
    
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string cell;
        
        std::getline(ss, cell, ',');
        int character = std::stoi(cell);
        
        std::getline(ss, cell, ',');
        int state = std::stoi(cell);
        
        std::getline(ss, cell, ',');
        double probability = std::stod(cell);
        
        priors.emplace_back(character, state, probability);
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
    
    // Parse character matrix
    spdlog::info("Parsing character matrix...");
    auto [taxa_names, raw_matrix] = parse_character_matrix(character_matrix_file);
    
    // Get number of characters
    size_t num_characters = 0;
    for (const auto& row : raw_matrix) {
        num_characters = std::max(num_characters, row.size());
    }
    
    spdlog::info("Found {} taxa and {} characters", taxa_names.size(), num_characters);
    
    // Parse or generate mutation priors
    std::vector<std::tuple<int, int, double>> raw_priors;
    if (!mutation_priors_file.empty()) {
        spdlog::info("Parsing mutation priors...");
        raw_priors = parse_mutation_priors(mutation_priors_file);
    } else {
        spdlog::info("No mutation priors provided. Assuming uniform priors.");
        raw_priors = generate_uniform_priors(raw_matrix, num_characters);
    }
    
    // Create state mappings and recode character matrix
    std::vector<std::vector<int>> character_matrix_recode(raw_matrix.size(), std::vector<int>(num_characters));
    std::vector<std::map<int, int>> original_to_new_mappings(num_characters);
    size_t max_alphabet_size = 0;
    
    spdlog::info("Recoding character states...");
    for (size_t c = 0; c < num_characters; ++c) {
        // Find unique valid states for this character
        std::set<int> valid_states;
        for (const auto& row : raw_matrix) {
            if (c < row.size() && row[c] > 0) {
                valid_states.insert(row[c]);
            }
        }
        
        // Create mapping from original to new states
        int new_idx = 1;
        for (int orig_state : valid_states) {
            original_to_new_mappings[c][orig_state] = new_idx++;
        }
        
        // Update max alphabet size
        if (valid_states.size() > max_alphabet_size) {
            max_alphabet_size = valid_states.size();
        }
        
        // Apply mapping to character matrix
        for (size_t i = 0; i < raw_matrix.size(); ++i) {
            if (c < raw_matrix[i].size()) {
                int orig_state = raw_matrix[i][c];
                if (orig_state > 0 && original_to_new_mappings[c].count(orig_state)) {
                    character_matrix_recode[i][c] = original_to_new_mappings[c][orig_state];
                } else if (orig_state == 0) {
                    character_matrix_recode[i][c] = 0; // Missing state
                } else {
                    character_matrix_recode[i][c] = -1; // Unknown state
                }
            } else {
                character_matrix_recode[i][c] = -1; // Unknown state
            }
        }
    }
    
    // Create mapping from taxa names to row indices
    std::unordered_map<std::string, size_t> taxa_name_to_idx;
    for (size_t i = 0; i < taxa_names.size(); ++i) {
        taxa_name_to_idx[taxa_names[i]] = i;
    }
    
    // Reorder character matrix to match leaf order in tree
    std::vector<std::vector<int>> reordered_matrix(t.num_leaves, std::vector<int>(num_characters));
    std::vector<bool> leaf_mapped(t.num_leaves, false);
    
    spdlog::info("Reordering rows to match tree structure...");
    for (size_t node_id = 0; node_id < t.num_nodes; ++node_id) {
        // Only process leaf nodes
        if (t.tree.out_degree(node_id) == 0) {
            size_t leaf_id = t.tree[node_id].data;
            
            if (leaf_id >= t.num_leaves) {
                throw std::runtime_error("Leaf ID out of range: " + std::to_string(leaf_id));
            }
            
            // Look up this leaf's name
            const std::string& leaf_name = t.node_names[leaf_id];
            
            if (taxa_name_to_idx.find(leaf_name) == taxa_name_to_idx.end()) {
                throw std::runtime_error("Taxon name from tree not found in character matrix: " + leaf_name);
            }
            
            // Get row from original matrix
            size_t orig_row = taxa_name_to_idx[leaf_name];
            
            // Copy to reordered matrix
            for (size_t c = 0; c < num_characters; ++c) {
                reordered_matrix[leaf_id][c] = character_matrix_recode[orig_row][c];
            }
            
            leaf_mapped[leaf_id] = true;
        }
    }
    
    // Check that all leaves were mapped
    for (size_t i = 0; i < t.num_leaves; ++i) {
        if (!leaf_mapped[i]) {
            spdlog::warn("Leaf {} not mapped to any taxon in character matrix", i);
        }
    }
    
    // Recode mutation priors
    std::vector<std::vector<double>> recoded_priors(num_characters, std::vector<double>(max_alphabet_size, 0.0));
    
    spdlog::info("Recoding mutation priors...");
    for (const auto& [character, orig_state, probability] : raw_priors) {
        if (character < 0 || character >= static_cast<int>(num_characters)) {
            throw std::runtime_error("Character index out of range in mutation priors: " + std::to_string(character));
        }
        
        if (orig_state <= 0) {
            continue; // Skip states 0 and -1
        }
        
        auto& mapping = original_to_new_mappings[character];
        if (mapping.find(orig_state) == mapping.end()) {
            spdlog::warn("State {} for character {} not found in character matrix", orig_state, character);
            continue;
        }
        
        int new_state = mapping[orig_state] - 1; // 0-indexed in the priors array
        recoded_priors[character][new_state] = probability;
    }
    
    // Extract character names if present in the first line
    std::vector<std::string> character_names;
    std::ifstream file(character_matrix_file);
    if (file.is_open()) {
        std::string header_line;
        if (std::getline(file, header_line)) {
            std::istringstream header_ss(header_line);
            std::string cell;
            std::getline(header_ss, cell, ','); // Skip first column (row names)
            
            while (std::getline(header_ss, cell, ',')) {
                character_names.push_back(cell);
            }
        }
    }
    
    // Fill in character names if not enough
    while (character_names.size() < num_characters) {
        character_names.push_back("character_" + std::to_string(character_names.size()));
    }
    
    phylogeny_data result;
    result.num_characters = num_characters;
    result.max_alphabet_size = max_alphabet_size;
    result.character_matrix = reordered_matrix;
    result.mutation_priors = recoded_priors;
    result.character_names = character_names;
    
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
        .required();

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

    spdlog::info("Parsing Newick tree file...");
    tree t = parse_newick_tree(program.get<std::string>("--tree"));
    spdlog::info("Tree Summary:");
    spdlog::info("  Number of leaves: {}", t.num_leaves);
    spdlog::info("  Total number of nodes: {}", t.num_nodes);
    spdlog::info("  Root: {}", t.tree[t.root_id].data);

    spdlog::info("Node details:");
    for (size_t i = 0; i < t.num_nodes; i++) {
        int j = t.tree[i].data;
        std::ostringstream children;
        for (auto child : t.tree.successors(i)) {
            children << t.tree[child].data << " ";
        }
        
        spdlog::info("  Node {}: name='{}', branch_length={}, children=[{}]", 
                    j, 
                    t.node_names[j].empty() ? "<unnamed>" : t.node_names[j], 
                    t.branch_lengths[j],
                    children.str());
    }

    phylogeny_data data = process_phylogeny_data(
        t, 
        program.get<std::string>("--character-matrix"),
        program.get<std::string>("--mutation-priors")
    );
    
    spdlog::info("Processed phylogeny data:");
    spdlog::info("  Number of characters: {}", data.num_characters);
    spdlog::info("  Max alphabet size: {}", data.max_alphabet_size);
    
    if (data.character_matrix.size() > 0) {
        spdlog::info("First few entries of the recoded character matrix:");
        for (size_t i = 0; i < std::min(data.character_matrix.size(), size_t(5)); ++i) {
            std::ostringstream row;
            for (size_t j = 0; j < std::min(data.character_matrix[i].size(), size_t(10)); ++j) {
                row << data.character_matrix[i][j] << " ";
            }
            spdlog::info("  Leaf {}: [{}]", i, row.str());
        }
    }
    
    if (data.mutation_priors.size() > 0) {
        spdlog::info("First few entries of the recoded mutation priors:");
        for (size_t c = 0; c < std::min(data.mutation_priors.size(), size_t(5)); ++c) {
            std::ostringstream row;
            for (size_t s = 0; s < std::min(data.mutation_priors[c].size(), size_t(5)); ++s) {
                row << data.mutation_priors[c][s] << " ";
            }
            spdlog::info("  Character {}: [{}]", c, row.str());
        }
    }

    return 0;
}
