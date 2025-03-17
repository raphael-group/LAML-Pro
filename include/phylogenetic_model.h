#ifndef PHYLOGENETIC_MODEL_H
#define PHYLOGENETIC_MODEL_H

#include <vector>

template <typename D>
class phylogenetic_model {
public:
    std::vector<size_t> alphabet_sizes; /* The size of this is the number of characters */
    std::vector<double> parameters;  /* The set of non-branch length parameters for the model */

    virtual ~phylogenetic_model() = default;

    /*!
     * @brief Computes the product of the probability matrix with a vector in 
     *        log space for a specific character. Specifically, this computes 
     *        log(P^{c}v) where P^{c} is the probability transition matrix for 
     *        character c with branch length b.
     * 
     * @param data An arbitrary, model specific data object
     * @param character The character index
     * @param branch_length The branch length
     * @param log_vector Input vector in log space
     * @param result solution vector for the log-product of size alphabet_sizes[character]
     */
    virtual void compute_log_pmatrix_vector_product(
        D& data,
        size_t character, 
        double branch_length, 
        const std::vector<double>& log_vector,
        std::vector<double>& result
    ) const = 0;

    /*!
     * @brief Computes the product of the transpose probability matrix 
     *        with a vector in log space for a specific character. Specifically, 
     *        this computes log(v^TP^{c}) where P^{c} is the probability transition
     *        matrix for character c with branch length b.
     * 
     * @param data An arbitrary, model specific data object
     * @param character The character index
     * @param branch_length The branch length
     * @param log_vector Input vector in log space
     * @param result solution vector for the log-product of size alphabet_sizes[character]
     */
    virtual void compute_log_pmatrix_transpose_vector_product(
        D& data,
        size_t character,
        double branch_length,
        const std::vector<double>& log_vector,
        std::vector<double>& result
    ) const = 0;

    /*!
     * Computes the log likelihood for a specific taxa and character.
     * 
     * @param data An arbitrary, model specific data object
     * @param character The character index
     * @param taxa_id The taxa identifier which is between 0 and num_taxa - 1
     * @param result log inside likelihood for the taxa at the specified character
     */
    virtual void compute_taxa_log_inside_likelihood(
        D& data,
        size_t character, 
        size_t taxa_id,
        std::vector<double>& result
    ) const = 0;

    /*!
     * Computes the root distribution for a specific character.
     *
     * @param data An arbitrary, model specific data object
     * @param character The character index
     * @param result log of the root distribution for the character
     */
    virtual void compute_root_distribution(
        D& data,
        size_t character,
        std::vector<double>& result
    ) const = 0;
};

#endif // PHYLOGENETIC_MODEL_H