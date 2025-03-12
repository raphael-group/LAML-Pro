#ifndef PHYLOGENETIC_H
#define PHYLOGENETIC_H

#include <vector>

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

class phylogeny {
public:
    public:
    size_t num_leaves;
    size_t num_nodes;

    size_t root_id;   
    digraph<size_t> tree;
    std::vector<double> branch_lengths;
    std::unique_ptr<phylogenetic_model> model;
    
    phylogeny(
        size_t num_leaves,
        size_t num_nodes,
        size_t root_id,
        const digraph<size_t>& tree,
        const std::vector<double>& branch_lengths,
        std::unique_ptr<phylogenetic_model> model
    ) : 
        num_leaves(num_leaves),
        num_nodes(num_nodes),
        root_id(root_id),
        tree(tree),
        branch_lengths(branch_lengths),
        model(std::move(model))
    {}

    void compute_inside_log_likelihood(likelihood_buffer& b); 
};

#endif