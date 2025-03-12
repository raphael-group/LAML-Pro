#ifndef PHYLOGENETIC_H
#define PHYLOGENETIC_H

#include <vector>
#include <memory>

#include "digraph.h"
#include "phylogenetic_model.h"

class likelihood_buffer {
    private:
    std::vector<double> buffer;
    size_t num_characters;
    size_t max_alphabet_size;
    size_t num_nodes;

    public:
    likelihood_buffer(size_t num_characters, size_t max_alphabet_size, size_t num_nodes, double fill_value = -1e9)
        : num_characters(num_characters),
          max_alphabet_size(max_alphabet_size),
          num_nodes(num_nodes) {
        buffer.resize(num_characters * num_nodes * max_alphabet_size, fill_value);
    }

    double& operator()(size_t character, size_t node, size_t symbol) {
        return buffer[character * (num_nodes * max_alphabet_size) + node * max_alphabet_size + symbol];
    }

    const double& operator()(size_t character, size_t node, size_t symbol) const {
        return buffer[character * (num_nodes * max_alphabet_size) + node * max_alphabet_size + symbol];
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

    double compute_inside_log_likelihood(const phylogenetic_model<D> &m, likelihood_buffer& b, std::vector<D> &node_data); 
};

template <typename D>
static void compute_inside_for_character(
    const phylogenetic_model<D> &model,
    std::vector<D> &node_data,
    const std::vector<int> &post_order, 
    const phylogeny<D> &p,
    likelihood_buffer &b, 
    size_t character
) {
    size_t alphabet_size = model.alphabet_sizes[character];
    std::vector<double> tmp_buffer_1(alphabet_size, 0.0);
    std::vector<double> tmp_buffer_2(alphabet_size, 0.0);

    for (auto node_id : post_order) {
        size_t node = p.tree[node_id].data;

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

template <typename D>
double phylogeny<D>::compute_inside_log_likelihood(
    const phylogenetic_model<D> &model,
    likelihood_buffer& b, 
    std::vector<D> &node_data
) {
    std::vector<int> post_order = this->tree.postorder_traversal(this->root_id);

    size_t num_characters = model.alphabet_sizes.size();
    for (size_t c = 0; c < num_characters; ++c) {
        compute_inside_for_character(model, node_data, post_order, *this, b, c);
    }

    size_t root = this->tree[this->root_id].data;
    double llh = 0.0;
    for (size_t c = 0; c < num_characters; ++c) {
        std::vector<double> root_llh(model.alphabet_sizes[c]);
        std::vector<double> buff(model.alphabet_sizes[c]);

        for (size_t j = 0; j < model.alphabet_sizes[c]; j++) {
            root_llh[j] = b(c, root, j);
        }
        model.compute_log_pmatrix_vector_product(node_data[root], c, this->branch_lengths[root], root_llh, buff);
        llh += buff[1];
    }

    return llh;
}
#endif