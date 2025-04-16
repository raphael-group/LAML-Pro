#include <vector>
#include <array>
#include <cmath>

#include <iostream>
#include <spdlog/spdlog.h>

#include "math_utilities.h"
#include "phylogeny.h"
#include "models/laml.h"
#include "laml_em.h"

#include <nlopt.hpp>

struct e_step_data {
    std::vector<std::array<double, 6>>& responsibilities;
    double leaf_responsibility;
    int num_missing;
    int num_not_missing;
};

double m_step_objective_and_grad(const std::vector<double>& parameters, std::vector<double>& gradient, void* data)
{
    e_step_data* m_data = static_cast<e_step_data*>(data);

    std::vector<std::array<double, 6>>& responsibilities = m_data->responsibilities;
    double leaf_responsibility = m_data->leaf_responsibility;
    int num_missing = m_data->num_missing;
    int num_not_missing = m_data->num_not_missing;

    double nu = parameters[0];
    double phi = parameters[1];

    double result = 0.0;
    double dnu = 0.0;
    double dphi = 0.0;

    for (size_t i = 0; i < gradient.size(); i++) {
        gradient[i] = 0.0;
    }
    
    for (size_t i = 0; i < responsibilities.size(); ++i) {
        double blen = parameters[i+2];

        double exp_blen     = std::exp(-blen);
        double exp_blen_nu  = std::exp(-blen * nu);
        double log_exp_blen = std::log(1.0 - exp_blen);

        {
            result += -responsibilities[i][0] * blen * (1.0 + nu);
            dnu    += -responsibilities[i][0] * blen;
            gradient[i + 2] += -responsibilities[i][0] * (1.0 + nu);
        }

        {
            double log_part = log_exp_blen;
            result += responsibilities[i][1] * (log_part - blen * nu);
            dnu -= responsibilities[i][1] * blen;
            double d_log_part = exp_blen / (1.0 - exp_blen);
            gradient[i + 2] += responsibilities[i][1] * (d_log_part - nu);
        }

        {
            double val = 1.0 - exp_blen_nu;
            result += (responsibilities[i][2] + responsibilities[i][4]) * std::log(val);
            double d_nu = (blen * exp_blen_nu) / val;
            dnu += (responsibilities[i][2] + responsibilities[i][4]) * d_nu;
            double d_blen = (nu * exp_blen_nu) / val;
            gradient[i + 2] += (responsibilities[i][2] + responsibilities[i][4]) * d_blen;
        }

        {
            result += -responsibilities[i][3] * blen * nu;
            dnu -= responsibilities[i][3] * blen;
            gradient[i + 2] += -responsibilities[i][3] * nu;
        }

        gradient[i + 2] *= blen;
    }

    {
        double a = num_not_missing * std::log(1.0 - phi);
        double b = (num_missing - leaf_responsibility) * std::log(phi);
        result += a + b;

        dphi += num_not_missing * (-1.0 / (1.0 - phi));
        dphi += (num_missing - leaf_responsibility) * (1.0 / phi);
    }

    gradient[0] = -dnu;
    gradient[1] = -dphi;
    for (size_t i = 0; i < responsibilities.size(); ++i) {
        gradient[i + 2] = -gradient[i + 2];
    }

    return -result;
}

void laml_expectation_step(
    const tree& t,
    const laml_model& model,
    const std::vector<double>& likelihoods,
    const likelihood_buffer& inside_ll,
    const likelihood_buffer& outside_ll,
    const likelihood_buffer& edge_inside_ll,
    std::vector<laml_data>& node_data,
    std::vector<std::array<double, 6>>& responsibilities,
    double& leaf_responsibility
) {
    double nu = model.parameters[0];

    for (size_t v_id = 0; v_id < t.num_nodes; ++v_id) {
        for (int i = 0; i < 6; i++) responsibilities[v_id][i] = 0.0;
    }

    size_t num_characters = inside_ll.num_characters;
    for (size_t character = 0; character < num_characters; ++character) {
        size_t alphabet_size = model.alphabet_sizes[character];
        std::vector<double> tmp_buffer(alphabet_size - 2, 0.0);

        {
            size_t root = t.tree[t.root_id].data;
            double blen = t.branch_lengths[root];

            // model.compute_root_distribution(node_data[root], character, tmp_buffer);
            double p_zero = 0.0; //tmp_buffer[1];

            double log_C_zero_zero = p_zero + inside_ll(character, root, 1) - blen * (1.0 + nu)  - likelihoods[character];
            double log_C_zero_miss = p_zero + inside_ll(character, root, 0) + node_data[root].v2 - likelihoods[character];

            double log_C_zero_alpha = -1e12;
            if (alphabet_size > 2) {
                for (size_t j = 0; j < alphabet_size - 2; j++) {
                    tmp_buffer[j]  = p_zero + inside_ll(character, root, j + 2);
                    tmp_buffer[j] += model.log_mutation_priors[character][j] + node_data[root].v1 - blen * nu;
                    tmp_buffer[j] -= likelihoods[character];
                }
                log_C_zero_alpha = log_sum_exp(tmp_buffer.begin(), tmp_buffer.end());
            }

            responsibilities[root][0] += std::exp(log_C_zero_zero);
            responsibilities[root][1] += std::exp(log_C_zero_alpha);
            responsibilities[root][2] += std::exp(log_C_zero_miss);
        }

        for (size_t v_id = 0; v_id < t.num_nodes; ++v_id) {
            if (t.tree.in_degree(v_id) == 0) {
                continue;
            }

            size_t v = t.tree[v_id].data;
            
            size_t u_id = t.tree.predecessors(v_id)[0]; 
            size_t u = t.tree[u_id].data;

            size_t w_id = t.tree.successors(u_id)[0];
            if (v_id == w_id) {
                w_id = t.tree.successors(u_id)[1];
            }

            size_t w = t.tree[w_id].data; // w is the sibling of v, u is the parent of v
            double blen = t.branch_lengths[v];
            double log_C_zero_zero = outside_ll(character, u, 1) + edge_inside_ll(character, w, 1) 
                                   + inside_ll(character, v, 1) - blen * (1.0 + nu)
                                   - likelihoods[character];
            double log_C_zero_miss = outside_ll(character, u, 1) + edge_inside_ll(character, w, 1) 
                                   + inside_ll(character, v, 0) + node_data[v].v2
                                   - likelihoods[character];
            double log_C_miss_miss = outside_ll(character, u, 0) + edge_inside_ll(character, w, 0) 
                                   + inside_ll(character, v, 0) - likelihoods[character];

            double log_C_zero_alpha = -1e12;
            double log_C_alpha_alpha = -1e12;
            double log_C_alpha_miss = -1e12;
            if (alphabet_size > 2) {
                // compute log_C_zero_alpha
                for (size_t j = 0; j < alphabet_size - 2; j++) {
                    tmp_buffer[j] = outside_ll(character, u, 1) + edge_inside_ll(character, w, 1) + inside_ll(character, v, j + 2);
                    tmp_buffer[j] += model.log_mutation_priors[character][j] + node_data[v].v1 - blen * nu;
                    tmp_buffer[j] -= likelihoods[character];
                }
                log_C_zero_alpha = log_sum_exp(tmp_buffer.begin(), tmp_buffer.end());

                // compute log_C_alpha_alpha
                for (size_t j = 0; j < alphabet_size - 2; j++) {
                    tmp_buffer[j] = outside_ll(character, u, j + 2) + edge_inside_ll(character, w, j + 2) + inside_ll(character, v, j + 2) - blen * nu;
                    tmp_buffer[j] -= likelihoods[character];
                }
                log_C_alpha_alpha = log_sum_exp(tmp_buffer.begin(), tmp_buffer.end());

                // compute log_C_alpha_miss
                for (size_t j = 0; j < alphabet_size - 2; j++) {
                    tmp_buffer[j] = outside_ll(character, u, j + 2) + edge_inside_ll(character, w, j + 2) + inside_ll(character, v, 0) + node_data[v].v2;
                    tmp_buffer[j] -= likelihoods[character];
                }
                log_C_alpha_miss = log_sum_exp(tmp_buffer.begin(), tmp_buffer.end());
            }

            responsibilities[v][0] += std::exp(log_C_zero_zero);
            responsibilities[v][1] += std::exp(log_C_zero_alpha);
            responsibilities[v][2] += std::exp(log_C_zero_miss);
            responsibilities[v][3] += std::exp(log_C_alpha_alpha);
            responsibilities[v][4] += std::exp(log_C_alpha_miss);
            responsibilities[v][5] += std::exp(log_C_miss_miss);
        }
    }

    leaf_responsibility = 0.0;
    for (size_t v_id = 0; v_id < t.num_nodes; ++v_id) {
        if (t.tree.out_degree(v_id) != 0) continue;
        size_t v = t.tree[v_id].data;
        for (size_t character = 0; character < num_characters; ++character) {
            leaf_responsibility += std::exp(inside_ll(character, v, 0) + outside_ll(character, v, 0) - likelihoods[character]);
        }
    } 
}

em_results laml_expectation_maximization(
    tree& t, 
    laml_model& model,
    int max_em_iterations,
    bool verbose
) {
    int num_characters = model.alphabet_sizes.size();
    int max_alphabet_size = *std::max_element(model.alphabet_sizes.begin(), model.alphabet_sizes.end());
    
    int num_missing = 0;
    int num_not_missing = 0;

    for (size_t i = 0; i < model.character_matrix.size(); ++i) {
        for (size_t j = 0; j < model.character_matrix[i].size(); ++j) {
            if (model.character_matrix[i][j] == -1) {
                num_missing++;
            } else {
                num_not_missing++;
            }
        }
    }

    likelihood_buffer inside_ll(num_characters, max_alphabet_size, t.num_nodes);
    likelihood_buffer edge_inside_ll(num_characters, max_alphabet_size, t.num_nodes);
    likelihood_buffer outside_ll(num_characters, max_alphabet_size, t.num_nodes);

    // set up the M-step solver
    nlopt::opt opt(nlopt::LD_CCSAQ, t.num_nodes + 2);
    
    std::vector<double> params(t.num_nodes + 2);
    std::vector<double> lb(t.num_nodes + 2, 1e-5);
    std::vector<double> ub(t.num_nodes + 2, std::numeric_limits<double>::infinity());
    ub[1] = 1.0 - 1e-5; // phi in [0, 1]
    
    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);
    opt.set_xtol_rel(1e-7);

    // initialize model parameters
    std::vector<double> internal_comp_buffer(max_alphabet_size);
    auto model_data = model.initialize_data(t.tree, t.branch_lengths, &internal_comp_buffer);

    params[0] = model.parameters[0];
    params[1] = model.parameters[1];
    for (size_t i = 0; i < t.num_nodes; ++i) {
        params[i + 2] = t.branch_lengths[i];
    }
    
    double llh = 0.0;
    double EM_STOPPING_CRITERION = 1e-5;
    int i = 0;
    for (; i < max_em_iterations; i++) {
        auto likelihood = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);

        phylogeny::compute_edge_inside_log_likelihood(model, t, inside_ll, edge_inside_ll, model_data);
        phylogeny::compute_outside_log_likelihood(model, t, edge_inside_ll, outside_ll, model_data);

        std::vector<std::array<double, 6>> responsibilities(t.num_nodes);
        double leaf_responsibility = 0.0;
        laml_expectation_step(t, model, likelihood, inside_ll, outside_ll, edge_inside_ll, model_data, responsibilities, leaf_responsibility);

        double llh_before = 0.0;
        for (int character = 0; character < num_characters; ++character) {
            llh_before += likelihood[character];
        }

        if (verbose) {
            spdlog::info("EM iteration: {} Log likelihood: {}", i, llh_before);
            spdlog::info("Nu: {} Phi: {}", model.parameters[0], model.parameters[1]);
        }

        double fx;
        try {
            e_step_data params_data = {responsibilities, leaf_responsibility, num_missing, num_not_missing};
            opt.set_min_objective(m_step_objective_and_grad, &params_data); 
            opt.optimize(params, fx);
        } catch (const std::runtime_error &e) {
            throw e;
        }

        auto saved_params = model.parameters;
        auto saved_branch_lengths = t.branch_lengths;

        model.parameters[0] = params[0];
        model.parameters[1] = params[1];
        for (size_t i = 0; i < t.num_nodes; ++i) {
            t.branch_lengths[i] = params[i + 2];
        }

        // update model parameters
        model_data = model.initialize_data(t.tree, t.branch_lengths, &internal_comp_buffer);
        likelihood = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);

        double llh_after = 0.0;
        for (int character = 0; character < num_characters; ++character) {
            llh_after += likelihood[character];
        }

        if (llh_after < llh_before) { 
            throw std::runtime_error("LLH decreased in M-step.");
        }

        llh = llh_after;

        if ((llh_after - llh_before) / abs(llh_before) < EM_STOPPING_CRITERION) {
            break;
        }
    }

    return {llh, i + 1};
}