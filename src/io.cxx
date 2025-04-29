#include "io.h"
#include <variant>

#include <iostream>
#include <iomanip> // for std::setprecision

#include <spdlog/spdlog.h>
#include "csv.hpp"
#include "extern/compact_tree.h"

#define NEGATIVE_INFINITY (-1e7)
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
        if (branch_lengths[j] == 0) {
            branch_lengths[j] = 1.0;
        }
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

std::string write_newick_tree(const tree& t) {
    std::function<std::string(size_t)> recursive_write = [&](size_t node) -> std::string {
        auto& g = t.tree;
        std::string result = "";
        
        std::vector<size_t> children;
        for (auto child : g.successors(node)) {
            children.push_back(child);
        }
        
        if (!children.empty()) {
            result += "(";
            for (size_t i = 0; i < children.size(); ++i) {
                if (i > 0) result += ",";
                result += recursive_write(children[i]);
            }
            result += ")";
        }
        
        int node_id = g[node].data;
        if (!t.node_names[node_id].empty()) {
            result += t.node_names[node_id];
        }
        
        result += ":" + std::to_string(t.branch_lengths[node_id]);
        return result;
    };
    
    return recursive_write(t.root_id) + ";";
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
 * @brief Parse an observation matrix from a CSV file.
 * @param filename Path to the CSV file
 * @return Triplet of (taxa_names, column_names, states) where matrix[i][j][k] is the probability for taxon i, character j and state k
 */
std::tuple<
    std::vector<std::string>, 
    std::vector<std::pair<int, int>>,
    std::vector<std::vector<std::vector<double>>> 
> parse_observation_matrix(const std::string& filename) {
    std::vector<std::string> taxa_names;
    std::map<std::string, size_t> taxa_index_map;
    std::set<std::pair<int, int>> character_key_set;
    std::vector<std::tuple<std::string, std::pair<int, int>, std::vector<double>>> parsed_rows;

    try {
        csv::CSVReader reader(filename);
       
        std::vector<std::string> col_names = reader.get_col_names();

        for (auto& row: reader) {
            // column names are hard coded for now @TODO
            std::string taxon_name = row["cell_name"].get<std::string>();
            int cassette_idx = static_cast<int>(row["cassette_idx"].get<double>());
            int target_site = static_cast<int>(row["target_site"].get<double>());
            std::pair<int, int> character_key = {cassette_idx, target_site};
            character_key_set.insert(character_key);

            std::vector<double> state_probs;

            // Dynamically find all keys matching "state*_prob"
            for (size_t i = 0; i < row.size(); ++i) {
                const std::string& key = col_names[i];
                if (key.rfind("state", 0) == 0 && key.find("_prob") != std::string::npos) {
                    try {
                        double prob = row[i].get<double>();
                        state_probs.push_back(prob);
                    } catch (...) {
                        state_probs.push_back(NEGATIVE_INFINITY); // fallback in case of bad data
                    }
                }
            }
            //std::vector<double> state_probs(4, NEGATIVE_INFINITY);
            //state_probs[0] = row["state0_prob"].get<double>();
            //state_probs[1] = row["state1_prob"].get<double>();
            //state_probs[2] = row["state2_prob"].get<double>();
            //state_probs[3] = row["state3_prob"].get<double>();

            parsed_rows.emplace_back(taxon_name, character_key, state_probs);
            if (taxa_index_map.find(taxon_name) == taxa_index_map.end()) {
                taxa_index_map[taxon_name] = taxa_names.size(); // quick lookup for taxa index i
                taxa_names.push_back(taxon_name); // ordered list of taxa
            }
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse character matrix file: " + std::string(e.what()));
    }
    
    if (taxa_names.empty()) {
        throw std::runtime_error("No data found in character matrix file");
    }

    std::vector<std::pair<int, int>> character_keys(character_key_set.begin(), character_key_set.end());
    std::sort(character_keys.begin(), character_keys.end());

    std::map<std::pair<int, int>, size_t> character_index_map;
    for (size_t j = 0; j < character_keys.size(); ++j) {
        character_index_map[character_keys[j]] = j;
    } // for any (cassette_idx, target_site) pair, find the column index j

    // building 3D matrix[i][j][k]
    std::vector<std::vector<std::vector<double>>> matrix(
        taxa_names.size(), // num_taxa as rows, num columns (cassette_idx, target_site)
                           // num state probabilities 
        std::vector<std::vector<double>>(character_keys.size(), std::vector<double>(4, 0.0))
    );

    // fill in the matrix
    for (const auto& [taxon_name, character_key, probs] : parsed_rows) {
        size_t i = taxa_index_map[taxon_name];
        size_t j = character_index_map[character_key];
        matrix[i][j] = probs;
    }

    /*for (size_t i = 0; i < matrix.size(); ++i) {
        spdlog::info("Taxon {}:", i);
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            std::ostringstream oss;
            oss << "[ ";
            for (double val : matrix[i][j]) {
                oss << val << " ";
            }
            oss << "]";
            spdlog::info("  Character {} → {}", j, oss.str());
        }
    }*/

    /*
    // std::cout << "Character_keys[0]: " << character_keys[0] << std::endl;
    std::cout << "Character keys:\n";
    for (size_t i = 0; i < character_keys.size(); ++i) {
        const auto& [cassette_idx, target_site] = character_keys[i];
        std::cout << "  index " << i << ": (cassette_idx = " << cassette_idx
                  << ", target_site = " << target_site << ")\n";
    }
    std::cout << "taxa_names[0]: " << taxa_names[0] << std::endl;
    */ 
    return {taxa_names, character_keys, matrix};
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

    for (const auto& [character, state, probability] : priors) {
        spdlog::debug("Parsed prior: character = {}, state = {}, probability = {:.6f}",
                      character, state, probability);
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
    const std::vector<std::vector<int>>& character_matrix, size_t num_characters
) {
    
    std::vector<std::tuple<int, int, double>> priors;
    
    for (size_t c = 0; c < num_characters; ++c) {
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
 * @brief Generate uniform mutation priors if none are provided
 * Overloads above function, but tensor-compatible for observation matrix
 * @param raw_matrix The observation matrix
 * @param num_characters Number of characters
 * @return Vector of (character, state, probability) tuples
 */
std::vector<std::tuple<int, int, double>> generate_uniform_priors(
    const std::vector<std::vector<std::vector<double>>>& raw_matrix,
    size_t num_characters
) {
    std::vector<std::tuple<int, int, double>> priors;

    for (size_t c = 0; c < num_characters; ++c) {
        std::set<int> unique_states;

        for (const auto& row : raw_matrix) {
            if (c < row.size()) {
                const auto& probs = row[c];
                for (size_t s = 0; s < probs.size(); ++s) {
                    if (probs[s] > 0.0) {
                        unique_states.insert(static_cast<int>(s));
                    }
                }
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
 * @brief Helper function to print the result struct from process_phylogeny_data
 * @param result The phylogeny_data result object
 */
void print_result_summary(const phylogeny_data& result) {
    spdlog::info("=== END OF IO SUMMARY ===");
    spdlog::info("Number of characters: {}", result.num_characters);
    spdlog::info("Max alphabet size: {}", result.max_alphabet_size);
    spdlog::info("Data type: {}\n", result.data_type);

    spdlog::info("Mutation priors:");
    for (size_t i = 0; i < result.mutation_priors.size(); ++i) {
        std::ostringstream oss;
        oss << "  Character " << i << ": [";
        for (size_t j = 0; j < result.mutation_priors[i].size(); ++j) {
            oss << std::fixed << std::setprecision(4) << result.mutation_priors[i][j];
            if (j + 1 < result.mutation_priors[i].size()) oss << ", ";
        }
        oss << "]";
        spdlog::info("{}", oss.str());
    }

    spdlog::info("\nObservation matrix:");
    for (size_t i = 0; i < result.observation_matrix.size(); ++i) {
        spdlog::info("  Taxon {}:", i);
        for (size_t j = 0; j < result.observation_matrix[i].size(); ++j) {
            std::ostringstream oss;
            oss << "    Character " << j << ": [";
            for (size_t k = 0; k < result.observation_matrix[i][j].size(); ++k) {
                oss << std::fixed << std::setprecision(10) << result.observation_matrix[i][j][k];
                if (k + 1 < result.observation_matrix[i][j].size()) oss << ", ";
            }
            oss << "]";
            spdlog::info("{}", oss.str());
        }
    }
    spdlog::info("\nCharacter matrix:");
    for (size_t i = 0; i < result.character_matrix.size(); ++i) {
        std::ostringstream oss;
        oss << "  Taxon " << i << ": [";
        for (size_t j = 0; j < result.character_matrix[i].size(); ++j) {
            oss << result.character_matrix[i][j];
            if (j + 1 < result.character_matrix[i].size()) oss << ", ";
        }
        oss << "]";
        spdlog::info("{}", oss.str());
    }

    spdlog::info("=======================");
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
    const std::string& mutation_priors_file,
    const std::string& data_type) {
  
    std::vector<std::string> taxa_names;
    std::vector<std::pair<int, int>> character_keys;
    std::vector<std::vector<int>> raw_matrix;
    std::vector<std::vector<std::vector<double>>> raw_tensor_matrix;
    size_t num_characters = 0;

    if (data_type == "observation-matrix") { 
        std::tie(taxa_names, character_keys, raw_tensor_matrix) = parse_observation_matrix(character_matrix_file);

        for (const auto& row : raw_tensor_matrix) {
            num_characters = std::max(num_characters, row.size());
        }
        //throw std::runtime_error("observation-matrix requires a tensor based workflow, which is not yet supported.");
    } else if (data_type == "character-matrix") {
        std::tie(taxa_names, raw_matrix) = parse_character_matrix(character_matrix_file);
        for (const auto& row : raw_matrix) {
            num_characters = std::max(num_characters, row.size());
        }
    } else {
        throw std::invalid_argument("Unknown data_type: " + data_type + ". Expected 'character-matrix' or 'observation-matrix'.");
    }


    spdlog::info("Found {} taxa and {} characters", taxa_names.size(), num_characters);
    
    std::vector<std::tuple<int, int, double>> raw_priors;
    if (!mutation_priors_file.empty()) {
        raw_priors = parse_mutation_priors(mutation_priors_file);
    } else {
        spdlog::info("No mutation priors provided. Assuming uniform priors.");
        // overloaded tensor-compatible method 
        raw_priors = generate_uniform_priors(raw_matrix, num_characters);
    }

    phylogeny_data result;
    if (data_type == "character-matrix") {
        // recode the character matrix
        std::vector<std::vector<int>> character_matrix_recode(raw_matrix.size(), std::vector<int>(num_characters));
        // maps original states to new contiguous state indices starting from 1
        std::vector<std::map<int, int>> original_to_new_mappings(num_characters);
        size_t max_alphabet_size = 0;
       
        // for each column, gather the unique states
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

   
        // number of leaves x number of characters
        std::vector<std::vector<int>> reordered_matrix(t.num_leaves, std::vector<int>(num_characters));
        std::vector<bool> leaf_mapped(t.num_leaves, false);
        
        for (size_t node_id = 0; node_id < t.num_nodes; ++node_id) {
            if (t.tree.out_degree(node_id) == 0) { // for each leaf node
                size_t leaf_id = t.tree[node_id].data; // get leaf_id in tree
                
                if (leaf_id >= t.num_leaves) {
                    throw std::runtime_error("Leaf ID out of range: " + std::to_string(leaf_id));
                }
                
                const std::string& leaf_name = t.node_names[leaf_id]; // retrieve the leaf name
                
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
        
        // std::vector<std::tuple<int, int, double>> raw_priors; need to transform raw_priors
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
                spdlog::warn("State {} for character {} in mutation priors not found in character matrix", orig_state, character);
                continue;
            }
            
            int new_state = mapping[orig_state] - 1;
            recoded_priors[character][new_state] = probability;
        }

        for (size_t c = 0; c < num_characters; ++c) {
            double sum = 0.0;
            for (size_t j = 0; j < max_alphabet_size; ++j) {
                sum += recoded_priors[c][j];
            }

            // It's possible that there are no edits observed at a given state
            if (sum <= 0) {
                spdlog::warn("Priors for character {} are all zero.", c);
                //spdlog::error("Priors for character {} are all zero.", c);
                // throw std::runtime_error("Priors for character " + std::to_string(c) + " are all zero.");
            } else {
                if (std::abs(sum - 1.0) > 1e-6) {
                    spdlog::warn("Priors for character {} do not sum to 1 ({}), renormalizing.", c, sum);
                }

                for (size_t j = 0; j < max_alphabet_size; ++j) {
                    if (sum > 0) {
                        recoded_priors[c][j] /= sum;
                    } else {
                        recoded_priors[c][j] = 1.0 / max_alphabet_size;
                    }
                }
            }


        }

        result.num_characters = num_characters;
        result.max_alphabet_size = max_alphabet_size;
        result.character_matrix = reordered_matrix;
        result.mutation_priors = recoded_priors;
        result.data_type = data_type;
        print_result_summary(result); 

    } else {
        size_t max_alphabet_size = raw_tensor_matrix[0][0].size()-1; //3; // number of edited states

        std::unordered_map<std::string, size_t> taxa_name_to_idx;
        for (size_t i = 0; i < taxa_names.size(); ++i) {
            taxa_name_to_idx[taxa_names[i]] = i;
        }

        std::vector<std::vector<std::vector<double>>> reordered_tensor_matrix(
            t.num_leaves,
            std::vector<std::vector<double>>(num_characters, std::vector<double>(max_alphabet_size - 1, 0.0))
        );
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
                    reordered_tensor_matrix[leaf_id][c] = raw_tensor_matrix[orig_row][c];
                }
                
                leaf_mapped[leaf_id] = true;
            }
        }
        
        for (size_t i = 0; i < t.num_leaves; ++i) {
            if (!leaf_mapped[i]) {
                spdlog::warn("Leaf {} not mapped to any taxon in character matrix", i);
            }
        }
        // std::vector<std::tuple<int, int, double>> raw_priors; need to transform raw_priors
        // use max_alphabet_size instead of hardcoding
        std::vector<std::vector<double>> recoded_priors(num_characters, std::vector<double>(max_alphabet_size, 0.0));
        
        for (const auto& [character, orig_state, probability] : raw_priors) {
            if (character < 0 || character >= static_cast<int>(num_characters)) {
                throw std::runtime_error("Character index out of range in mutation priors: " + std::to_string(character));
            }
            
            if (orig_state <= 0) {
                continue; // skip states 0 and -1
            }
            
            recoded_priors[character][orig_state-1] = probability;
        }

        for (size_t c = 0; c < num_characters; ++c) {
            double sum = 0.0;
            for (size_t j = 0; j < max_alphabet_size; ++j) {
                sum += recoded_priors[c][j];
            }

            if (sum <= 0.0) {
                spdlog::error("Priors for character {} are all zero.", c);
                spdlog::warn("Priors for character " + std::to_string(c) + " are all zero.");
                double uniform_prob = 1.0 / static_cast<double>(max_alphabet_size);  // max_alphabet_size is num edit states
                for (size_t j = 0; j < max_alphabet_size; ++j) { // priors are only for edit states
                    recoded_priors[c][j] = uniform_prob;
                }
            } else {
                for (size_t j = 0; j < max_alphabet_size; ++j) { // priors are only for edit states
                    recoded_priors[c][j] /= sum;
                }
            }
        }

        result.num_characters = num_characters;
        result.max_alphabet_size = max_alphabet_size; // hard coded for now @TODO: number of edited states
        result.observation_matrix = raw_tensor_matrix;
        result.mutation_priors = recoded_priors;
        result.data_type = data_type;
        print_result_summary(result); 

    }
    return result;
}
