#include "phylogeny.h"
#include <vector>

static void compute_inside_for_character(
    const std::vector<int> &post_order, 
    const phylogeny &p,
    likelihood_buffer &b, 
    size_t character
) {
    // TODO: fix excessive (ideally, all) copying of memory.
    size_t alphabet_size = p.model->alphabet_sizes[character];
    std::vector<double> tmp_buffer(alphabet_size, 0.0);

    for (auto node_id : post_order) {
        size_t node = p.tree[node_id].data;

        if (p.tree.out_degree(node_id) == 0) {
            p.model->compute_taxa_log_inside_likelihood(character, node, tmp_buffer);
            for (size_t j = 0; j < alphabet_size; j++) {
                b(character, node, j) = tmp_buffer[j];
            }

            continue;
        }

        for (size_t j = 0; j < alphabet_size; j++) {
            b(character, node, j) = 0.0;
        }

        for (auto u_id : p.tree.successors(node_id)) {
            size_t u = p.tree[u_id].data;
            double blen = p.branch_lengths[u];

            std::vector u_ll(alphabet_size, 0.0);
            for (size_t j = 0; j < alphabet_size; j++) {
                u_ll[j] = b(character, u, j);
            }

            p.model->compute_log_pmatrix_vector_product(character, blen, u_ll, tmp_buffer);
            for (size_t j = 0; j < alphabet_size; j++) {
                b(character, node, j) += tmp_buffer[j];
            } 
        }
    }
}

double phylogeny::compute_inside_log_likelihood(likelihood_buffer& b) {
    std::vector<int> post_order = this->tree.postorder_traversal(this->root_id);

    size_t num_characters = model->alphabet_sizes.size();
    for (size_t c = 0; c < num_characters; ++c) {
        compute_inside_for_character(post_order, *this, b, c);
    }

    size_t root = this->tree[this->root_id].data;
    double llh = 0.0;
    for (size_t c = 0; c < num_characters; ++c) {
        std::vector<double> root_llh(this->model->alphabet_sizes[c]);
        std::vector<double> buff(this->model->alphabet_sizes[c]);

        for (size_t j = 0; j < this->model->alphabet_sizes[c]; j++) {
            root_llh[j] = b(c, root, j);
        }
        this->model->compute_log_pmatrix_vector_product(c, this->branch_lengths[root], root_llh, buff);
        llh += buff[1];
    }

    return llh;
}