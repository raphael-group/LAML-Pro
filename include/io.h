#ifndef IO_H
#define IO_H

#include <string>
#include <vector>
#include <map>
#include <set>

#include "digraph.h"
#include "phylogeny.h"

/*!
 * @brief Data structure to hold character matrix, mutation priors, and associated metadata.
 */
struct phylogeny_data {
    size_t num_characters;
    size_t max_alphabet_size;
    std::vector<std::vector<int>> character_matrix;   // [leaf_id][character]
    std::vector<std::vector<double>> mutation_priors; // [character][state]
};

/*!
* @brief Parse a Newick tree file into a Tree struct.
* @param fname The path to the Newick tree file.
* @return Tree The parsed tree.
*/
tree parse_newick_tree(std::string fname);

/*!
 * @brief Parse a character matrix from a CSV file.
 * @param filename Path to the CSV file
 * @return Pair of (taxa_names, matrix) where matrix[i][j] is the state for taxon i, character j
 */
std::pair<std::vector<std::string>, std::vector<std::vector<int>>> parse_character_matrix(const std::string& filename);

/*!
 * @brief Parse mutation priors from a CSV file.
 * @param filename Path to the CSV file
 * @return Vector of (character, state, probability) tuples
 */
std::vector<std::tuple<int, int, double>> parse_mutation_priors(const std::string& filename);

/*!
 * @brief Generate uniform mutation priors if none are provided
 * @param character_matrix The character matrix
 * @param num_characters Number of characters
 * @return Vector of (character, state, probability) tuples
 */
std::vector<std::tuple<int, int, double>> generate_uniform_priors(
    const std::vector<std::vector<int>>& character_matrix, size_t num_characters
);
/*!
 * @brief Process character matrix and mutation priors to create a phylogeny_data object.
 * @param t The tree structure
 * @param character_matrix_file Path to the character matrix CSV
 * @param mutation_priors_file Optional path to the mutation priors CSV
 * @return Processed phylogeny_data
 */
phylogeny_data process_phylogeny_data(
    const tree& t, 
    const std::string& character_matrix_file, 
    const std::string& mutation_priors_file = ""
);

#endif 