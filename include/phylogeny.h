#ifndef PHYLOGENETIC_H
#define PHYLOGENETIC_H

#include <vector>
#include <memory>

#include "math_utilities.h"
#include "digraph.h"
#include "phylogenetic_model.h"

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

/**
 * @brief Buffer for storing likelihood values efficiently during phylogenetic calculations.
 * 
 * This class manages a three-dimensional array of likelihood values for phylogenetic computation.
 * The dimensions represent:
 * - characters (sites in a sequence)
 * - nodes in the phylogenetic tree
 * - possible states for each character
 * 
 * The buffer is implemented as a flat vector with indexing operations to access the three-dimensional data.
 * It is primarily used to store log-likelihood values during inside (post-order) and outside (pre-order)
 * probability calculations on phylogenetic trees.
 */
class likelihood_buffer {
    private:
    std::vector<double> buffer;

    public:
    size_t num_nodes;
    size_t num_characters;
    size_t max_alphabet_size;

    likelihood_buffer(size_t num_characters, size_t max_alphabet_size, size_t num_nodes, double fill_value = -1e9)
        : num_nodes(num_nodes),
          num_characters(num_characters),
          max_alphabet_size(max_alphabet_size) {
        buffer.resize(num_nodes * num_characters * max_alphabet_size, fill_value);
    }

    double& operator()(size_t character, size_t node, size_t state) {
        return buffer[node * (num_characters * max_alphabet_size) + character * max_alphabet_size + state];
    }

    const double& operator()(size_t character, size_t node, size_t state) const {
        return buffer[node * (num_characters * max_alphabet_size) + character * max_alphabet_size + state];
    }
};
namespace phylogeny {

template <typename D>
std::vector<double> compute_inside_log_likelihood(
    const phylogenetic_model<D>& model,
    const tree& t,
    likelihood_buffer& b,
    std::vector<D>& node_data
) {
    size_t num_characters = model.alphabet_sizes.size();
    size_t root = t.tree[t.root_id].data;
    size_t max_alphabet_size = *std::max_element(model.alphabet_sizes.begin(), model.alphabet_sizes.end());
        
    std::vector<int> post_order = t.tree.postorder_traversal(t.root_id);
    std::vector<double> tmp_buffer_1(max_alphabet_size, 0.0);
    std::vector<double> tmp_buffer_2(max_alphabet_size, 0.0);

    for (auto node_id : post_order) {
        size_t node = t.tree[node_id].data;

        for (size_t character = 0; character < num_characters; character++) {
            size_t alphabet_size = model.alphabet_sizes[character];

            if (t.tree.out_degree(node_id) == 0) {
                model.compute_taxa_log_inside_likelihood(node_data[node], character, node, tmp_buffer_1);
                for (size_t j = 0; j < alphabet_size; j++) {
                    b(character, node, j) = tmp_buffer_1[j];
                }
                continue;
            }

            for (size_t j = 0; j < alphabet_size; j++) {
                b(character, node, j) = 0.0;
            }

            for (auto u_id : t.tree.successors(node_id)) {
                size_t u = t.tree[u_id].data;
                double blen = t.branch_lengths[u];

                for (size_t j = 0; j < alphabet_size; j++) {
                    tmp_buffer_2[j] = b(character, u, j);
                }

                model.compute_log_pmatrix_vector_product(node_data[u], character, blen, tmp_buffer_2, tmp_buffer_1);
                for (size_t j = 0; j < alphabet_size; j++) {
                    b(character, node, j) += tmp_buffer_1[j];
                } 
            }
        }
    }


    std::vector<double> llh(num_characters, 0.0);
    for (size_t c = 0; c < num_characters; ++c) {
        std::vector<double> root_llh(model.alphabet_sizes[c]);
        std::vector<double> buff1(model.alphabet_sizes[c]);
        std::vector<double> buff2(model.alphabet_sizes[c]);

        for (size_t j = 0; j < model.alphabet_sizes[c]; j++) {
            root_llh[j] = b(c, root, j);
        }

        model.compute_log_pmatrix_vector_product(node_data[root], c, t.branch_lengths[root], root_llh, buff1);
        model.compute_root_distribution(node_data[root], c, buff2);

        for (size_t j = 0; j < model.alphabet_sizes[c]; j++) {
            buff1[j] += buff2[j];
        }

        llh[c] = log_sum_exp(buff1.begin(), buff1.end());
    }

    return llh;
}

template <typename D>
void compute_edge_inside_log_likelihood(
    const phylogenetic_model<D>& model,
    const tree& t,
    const likelihood_buffer& inside_ll,
    likelihood_buffer& b,
    std::vector<D>& node_data
) {
    size_t num_characters = model.alphabet_sizes.size();
    int max_alphabet_size = *std::max_element(model.alphabet_sizes.begin(), model.alphabet_sizes.end());
    std::vector<double> tmp_buffer_1(max_alphabet_size, 0.0);
    std::vector<double> tmp_buffer_2(max_alphabet_size, 0.0);

    for (auto node_id : t.tree.nodes()) {
        size_t node = t.tree[node_id].data;

        for (size_t character = 0; character < num_characters; character++) {
            size_t alphabet_size = model.alphabet_sizes[character];

            for (size_t j = 0; j < alphabet_size; j++) {
                tmp_buffer_2[j] = inside_ll(character, node, j);
            }

            double blen = t.branch_lengths[node];
            model.compute_log_pmatrix_vector_product(node_data[node], character, blen, tmp_buffer_2, tmp_buffer_1);
            for (size_t j = 0; j < alphabet_size; j++) {
                b(character, node, j) = tmp_buffer_1[j];
            } 
        }
    }
}

template <typename D>
void compute_outside_log_likelihood(
    const phylogenetic_model<D>& model,
    const tree& t,
    const likelihood_buffer& edge_inside_ll,
    likelihood_buffer& b,
    std::vector<D>& node_data
) {
    std::vector<int> preorder = t.tree.preorder_traversal(t.root_id);

    size_t num_characters = model.alphabet_sizes.size();
    int max_alphabet_size = *std::max_element(model.alphabet_sizes.begin(), model.alphabet_sizes.end());
    std::vector<double> tmp_buffer_1(max_alphabet_size, 0.0);
    std::vector<double> tmp_buffer_2(max_alphabet_size, 0.0);

    size_t root = t.tree[t.root_id].data;
    for (size_t character = 0; character < num_characters; character++) {
        size_t alphabet_size = model.alphabet_sizes[character];
        model.compute_root_distribution(node_data[root], character, tmp_buffer_1);
        model.compute_log_pmatrix_vector_product(node_data[root], character, t.branch_lengths[root], tmp_buffer_1, tmp_buffer_2);
        for (size_t j = 0; j < alphabet_size; j++) {
            b(character, root, j) = tmp_buffer_2[j];
        }
    }

    for (size_t u_id : preorder) {
        if (t.tree.in_degree(u_id) == 0) continue;

        size_t u = t.tree[u_id].data;

        /* Load parent w and sibling v */
        size_t w_id = t.tree.predecessors(u_id)[0]; 
        size_t w = t.tree[w_id].data;

        size_t v_id = t.tree.successors(w_id)[0];
        if (v_id == u_id) {
            v_id = t.tree.successors(w_id)[1];
        }

        size_t v = t.tree[v_id].data;

        for (size_t character = 0; character < num_characters; character++) {
            size_t alphabet_size = model.alphabet_sizes[character];
            for (size_t j = 0; j < alphabet_size; j++) {
                tmp_buffer_1[j] = edge_inside_ll(character, v, j) + b(character, w, j);
            }

            double blen = t.branch_lengths[u];
            model.compute_log_pmatrix_transpose_vector_product(node_data[u], character, blen, tmp_buffer_1, tmp_buffer_2);
            for (size_t j = 0; j < alphabet_size; j++) {
                b(character, u, j) = tmp_buffer_2[j];
            }
        }
    }
}

} // namespace phylogeny

#endif