#ifndef PHYLOGENETIC_H
#define PHYLOGENETIC_H

#include <vector>
#include <memory>

#include "math_utilities.h"
#include "digraph.h"
#include "phylogenetic_model.h"

class likelihood_buffer {
    private:
    std::vector<double> buffer;
    size_t num_nodes;
    size_t num_characters;
    size_t max_alphabet_size;

    public:
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

template <typename D>
class phylogeny {
public:
    public:
    size_t num_leaves;
    size_t num_nodes;

    size_t root_id;   
    digraph<size_t> tree;
    std::vector<double> branch_lengths;
    
    phylogeny(
        size_t num_leaves,
        size_t num_nodes,
        size_t root_id,
        const digraph<size_t>& tree,
        const std::vector<double>& branch_lengths
    ) : 
        num_leaves(num_leaves),
        num_nodes(num_nodes),
        root_id(root_id),
        tree(tree),
        branch_lengths(branch_lengths)
    {}

    double compute_inside_log_likelihood(
        const phylogenetic_model<D> &m, 
        likelihood_buffer& b, 
        std::vector<D> &node_data
    ) const; 

    void compute_edge_inside_log_likelihood(
        const phylogenetic_model<D> &m, 
        const likelihood_buffer& inside_ll, 
        likelihood_buffer& b, 
        std::vector<D> &node_data
    ) const;

    void compute_outside_log_likelihood(
        const phylogenetic_model<D> &m, 
        const likelihood_buffer& inside_ll, 
        const likelihood_buffer& edge_inside_ll,
        likelihood_buffer& b, 
        std::vector<D> &node_data
    ) const;
};

template <typename D>
static void compute_inside_helper(
    const phylogenetic_model<D> &model,
    std::vector<D> &node_data,
    const std::vector<int> &post_order, 
    const phylogeny<D> &p,
    likelihood_buffer &b
) {
    size_t num_characters = model.alphabet_sizes.size();
    int max_alphabet_size = *std::max_element(model.alphabet_sizes.begin(), model.alphabet_sizes.end());
    std::vector<double> tmp_buffer_1(max_alphabet_size, 0.0);
    std::vector<double> tmp_buffer_2(max_alphabet_size, 0.0);

    for (auto node_id : post_order) {
        size_t node = p.tree[node_id].data;

        for (size_t character = 0; character < num_characters; character++) {
            size_t alphabet_size = model.alphabet_sizes[character];

            if (p.tree.out_degree(node_id) == 0) {
                model.compute_taxa_log_inside_likelihood(node_data[node], character, node, tmp_buffer_1);
                for (size_t j = 0; j < alphabet_size; j++) {
                    b(character, node, j) = tmp_buffer_1[j];
                }

                continue;
            }

            for (size_t j = 0; j < alphabet_size; j++) {
                b(character, node, j) = 0.0;
            }

            for (auto u_id : p.tree.successors(node_id)) {
                size_t u = p.tree[u_id].data;
                double blen = p.branch_lengths[u];

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
}

template <typename D>
double phylogeny<D>::compute_inside_log_likelihood(
    const phylogenetic_model<D> &model,
    likelihood_buffer& b, 
    std::vector<D> &node_data
) const {
    std::vector<int> post_order = tree.postorder_traversal(root_id);

    compute_inside_helper(model, node_data, post_order, *this, b);

    size_t num_characters = model.alphabet_sizes.size();
    size_t root = tree[root_id].data;
    double llh = 0.0;
    for (size_t c = 0; c < num_characters; ++c) {
        std::vector<double> root_llh(model.alphabet_sizes[c]);
        std::vector<double> buff1(model.alphabet_sizes[c]);
        std::vector<double> buff2(model.alphabet_sizes[c]);

        for (size_t j = 0; j < model.alphabet_sizes[c]; j++) {
            root_llh[j] = b(c, root, j);
        }

        model.compute_log_pmatrix_vector_product(node_data[root], c, branch_lengths[root], root_llh, buff1);
        model.compute_root_distribution(node_data[root], c, buff2);

        for (size_t j = 0; j < model.alphabet_sizes[c]; j++) {
            buff1[j] += buff2[j];
        }

        llh += log_sum_exp(buff1.begin(), buff1.end());
    }

    return llh;
}

template <typename D>
void phylogeny<D>::compute_edge_inside_log_likelihood(
    const phylogenetic_model<D> &model,
    const likelihood_buffer& inside_ll,
    likelihood_buffer& b, 
    std::vector<D> &node_data
) const {
    size_t num_characters = model.alphabet_sizes.size();
    int max_alphabet_size = *std::max_element(model.alphabet_sizes.begin(), model.alphabet_sizes.end());
    std::vector<double> tmp_buffer_1(max_alphabet_size, 0.0);
    std::vector<double> tmp_buffer_2(max_alphabet_size, 0.0);

    for (auto node_id : tree.nodes()) {
        size_t node = tree[node_id].data;

        for (size_t character = 0; character < num_characters; character++) {
            size_t alphabet_size = model.alphabet_sizes[character];

            for (size_t j = 0; j < alphabet_size; j++) {
                tmp_buffer_2[j] = inside_ll(character, node, j);
            }

            double blen = branch_lengths[node];
            model.compute_log_pmatrix_vector_product(node_data[node], character, blen, tmp_buffer_2, tmp_buffer_1);
            for (size_t j = 0; j < alphabet_size; j++) {
                b(character, node, j) += tmp_buffer_1[j];
            } 
        }
    }
}

template <typename D>
void phylogeny<D>::compute_outside_log_likelihood(
    const phylogenetic_model<D> &model, 
    const likelihood_buffer& inside_ll, 
    const likelihood_buffer& edge_inside_ll,
    likelihood_buffer& b, 
    std::vector<D> &node_data
) const {
    size_t num_characters = model.alphabet_sizes.size();
    int max_alphabet_size = *std::max_element(model.alphabet_sizes.begin(), model.alphabet_sizes.end());
    std::vector<double> tmp_buffer_1(max_alphabet_size, 0.0);
    std::vector<double> tmp_buffer_2(max_alphabet_size, 0.0);
    for (auto u_id : tree.nodes()) {
        if (tree.in_degree(u_id) == 0) continue;

        size_t u = tree[u_id].data;

        /* Load parent w and sibling v */
        size_t w_id = tree.predecessors(u_id)[0]; 
        size_t w = tree[w_id].data;

        size_t v_id = tree.successors(w_id)[0];
        if (v_id == u_id) {
            v_id = tree.successors(w_id)[1];
        }

        size_t v = tree[v_id].data;

        for (size_t character = 0; character < num_characters; character++) {
            size_t alphabet_size = model.alphabet_sizes[character];
            for (size_t j = 0; j < alphabet_size; j++) {
                tmp_buffer_1[j] = edge_inside_ll(character, v, j) + inside_ll(character, w, j);
            }

            double blen = branch_lengths[w];
            model.compute_log_pmatrix_transpose_vector_product(node_data[u], character, blen, tmp_buffer_1, tmp_buffer_2);
            for (size_t j = 0; j < alphabet_size; j++) {
                b(character, u, j) = tmp_buffer_2[j];
            }
        }
    }
}

#endif