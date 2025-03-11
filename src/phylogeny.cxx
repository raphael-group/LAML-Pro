#include "phylogeny.h"
#include <vector>

static void compute_inside_for_character(size_t character, likelihood_buffer &b, const phylogeny &p) {
    // TODO: fix excessive (ideally, all) copying of memory.
    size_t alphabet_size = static_cast<size_t>(p.model->alphabet_sizes[character]);
    std::vector<int> post_order = p.tree.postorder_traversal(p.root_id);
    for (auto node_id : post_order) {
        size_t node = p.tree[node].data;
        if (p.tree.out_degree(node_id) == 0) {
            std::vector<double> ll = p.model->compute_taxa_log_inside_likelihood(character, node);
            for (int j = 0; j < alphabet_size; j++) {
                b(character, node, j) = ll[j];
            }
            continue;
        }

        for (int j = 0; j < alphabet_size; j++) {
            b(character, node, j) = 0.0;
        }

        for (auto u_id : p.tree.successors(node_id)) {
            size_t u = p.tree[u_id].data;
            double blen = p.branch_lengths[u];

            std::vector u_ll(alphabet_size, 0.0);
            for (int j = 0; j < alphabet_size; j++) {
                u_ll[j] = b(character, u, j);
            }

            auto res = p.model->compute_log_pmatrix_vector_product(character, blen, u_ll);
            for (int j = 0; j < alphabet_size; j++) {
                b(character, node, j) += res[j];
            } 
        }
    }
}

void phylogeny::compute_inside_log_likelihood(likelihood_buffer& b) {
    size_t num_characters = model->alphabet_sizes.size();
    for (size_t c = 0; c < num_characters; ++c) {
        compute_inside_for_character(c, b, *this);
    }
}