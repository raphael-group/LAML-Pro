#include <cstdlib>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <stdexcept>
#include <chrono>

#include "digraph.h"
#include "io.h"

#include "phylogenetic_model.h"
#include "phylogeny.h"
#include "models/laml.h"

#include <argparse/argparse.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/fmt/ostr.h>

#define FASTLAML_VERSION_MAJOR 1
#define FASTLAML_VERSION_MINOR 0

#include <Eigen/Core>
#include <iostream>
#include "extern/LBFGSB.h"
#include "extern/LBFGS.h"

using Eigen::VectorXd;
using namespace LBFGSpp;

class laml_m_step_objective
{
private:
    std::vector<std::array<double, 6>> responsibilities;
    double leaf_responsibility;
    int num_missing;
    int num_not_missing;
public:
    laml_m_step_objective(
        std::vector<std::array<double, 6>>& responsibilities,
        double leaf_responsibility,
        int num_missing,
        int num_not_missing
    ) : responsibilities(responsibilities), 
        leaf_responsibility(leaf_responsibility),
        num_missing(num_missing), 
        num_not_missing(num_not_missing) {}

    double operator()(const VectorXd& parameters, VectorXd& gradient)
    {
        double nu = parameters[0];
        double phi = parameters[1];

        double result = 0.0;
        double dnu = 0.0;
        double dphi = 0.0;

        gradient.setZero();

        for (size_t i = 0; i < responsibilities.size(); ++i) {
            double blen = parameters[i + 2];

            double exp_blen     = std::exp(-blen);
            double exp_blen_nu  = std::exp(-blen * nu);
            double log_exp_blen = std::log(1.0 - exp_blen);

            result += -responsibilities[i][0] * blen * (1.0 + nu);
            dnu    += -responsibilities[i][0] * blen;
            gradient[i + 2] += -responsibilities[i][0] * (1.0 + nu);

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
};

void laml_expectation_step(
    const tree& t,
    const laml_model& model,
    const std::vector<double>& likelihoods,
    const likelihood_buffer& inside_ll,
    const likelihood_buffer& outside_ll,
    const likelihood_buffer& edge_inside_ll,
    const std::vector<laml_data>& node_data,
    std::vector<std::array<double, 6>>& responsibilities,
    double& leaf_responsibility
) {
    double nu = model.parameters[0];
    double phi = model.parameters[1];

    for (size_t v_id = 0; v_id < t.num_nodes; ++v_id) {
        for (int i = 0; i < 6; i++) responsibilities[v_id][i] = 0.0;
    }

    size_t num_characters = inside_ll.num_characters;
    for (size_t character = 0; character < num_characters; ++character) {
        size_t alphabet_size = model.alphabet_sizes[character];
        std::vector<double> tmp_buffer(alphabet_size - 2, 0.0);

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

            // compute log_C_zero_alpha
            for (size_t j = 0; j < alphabet_size - 2; j++) {
                tmp_buffer[j] = outside_ll(character, u, 1) + edge_inside_ll(character, w, 1) + inside_ll(character, v, j + 2);
                tmp_buffer[j] += model.log_mutation_priors[character][j] + node_data[v].v1 - blen * nu;
                tmp_buffer[j] -= likelihoods[character];
            }
            double log_C_zero_alpha = log_sum_exp(tmp_buffer.begin(), tmp_buffer.end());

            // compute log_C_alpha_alpha
            for (size_t j = 0; j < alphabet_size - 2; j++) {
                tmp_buffer[j] = outside_ll(character, u, j + 2) + edge_inside_ll(character, w, j + 2) + inside_ll(character, v, j + 2) - blen * nu;
                tmp_buffer[j] -= likelihoods[character];
            }
            double log_C_alpha_alpha = log_sum_exp(tmp_buffer.begin(), tmp_buffer.end());

            // compute log_C_alpha_miss
            for (size_t j = 0; j < alphabet_size - 2; j++) {
                tmp_buffer[j] = outside_ll(character, u, j + 2) + edge_inside_ll(character, w, j + 2) + inside_ll(character, v, 0) + node_data[v].v2;
                tmp_buffer[j] -= likelihoods[character];
            }
            double log_C_alpha_miss = log_sum_exp(tmp_buffer.begin(), tmp_buffer.end());

            responsibilities[v][0] += fast_exp(log_C_zero_zero);
            responsibilities[v][1] += fast_exp(log_C_zero_alpha);
            responsibilities[v][2] += fast_exp(log_C_zero_miss);
            responsibilities[v][3] += fast_exp(log_C_alpha_alpha);
            responsibilities[v][4] += fast_exp(log_C_alpha_miss);
            responsibilities[v][5] += fast_exp(log_C_miss_miss);
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

double laml_expectation_maximization(tree t, phylogeny_data data, double initial_nu, double initial_phi) {
    laml_model model(t.tree, data.character_matrix, data.mutation_priors, initial_nu, initial_phi);

    int num_missing = 0;
    int num_not_missing = 0;

    // initialize branch lengths to 1.0
    for (size_t i = 0; i < t.num_nodes; ++i) {
        t.branch_lengths[i] = 1.0; //((double) rand()) / double(RAND_MAX);
    }

    for (size_t i = 0; i < data.character_matrix.size(); ++i) {
        for (size_t j = 0; j < data.character_matrix[i].size(); ++j) {
            if (data.character_matrix[i][j] == -1) {
                num_missing++;
            } else {
                num_not_missing++;
            }
        }
    }

    likelihood_buffer inside_ll(data.num_characters, data.max_alphabet_size + 2, t.num_nodes);
    likelihood_buffer edge_inside_ll(data.num_characters, data.max_alphabet_size + 2, t.num_nodes);
    likelihood_buffer outside_ll(data.num_characters, data.max_alphabet_size + 2, t.num_nodes);

    // Set up the box-constrained L-BFGS solver
    LBFGSBParam<double> lbfgs_params;
    lbfgs_params.epsilon = 1e-3;
    lbfgs_params.epsilon_rel = 1e-3;
    lbfgs_params.max_iterations = 100;

    VectorXd params = VectorXd::Zero(t.num_nodes + 2);
    VectorXd lb = VectorXd::Constant(t.num_nodes + 2, 1e-6);
    VectorXd ub = VectorXd::Constant(t.num_nodes + 2, std::numeric_limits<double>::infinity());
    ub[1] = 1.0;

    LBFGSBSolver<double> solver(lbfgs_params);

    // initialize model parameters
    std::vector<double> internal_comp_buffer(data.max_alphabet_size + 2);
    auto model_data = model.initialize_data(&internal_comp_buffer, t.branch_lengths);

    params[0] = model.parameters[0];
    params[1] = model.parameters[1];
    for (size_t i = 0; i < t.num_nodes; ++i) {
        params[i + 2] = t.branch_lengths[i];
    }

    int MAX_EM_ITERATIONS = 100;
    double EM_STOPPING_CRITERION = 1e-5;
    for (int i = 0; i < MAX_EM_ITERATIONS; i++) {
        auto likelihood = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);
        phylogeny::compute_edge_inside_log_likelihood(model, t, inside_ll, edge_inside_ll, model_data);
        phylogeny::compute_outside_log_likelihood(model, t, edge_inside_ll, outside_ll, model_data);

        std::vector<std::array<double, 6>> responsibilities(t.num_nodes);
        double leaf_responsibility = 0.0;
        laml_expectation_step(t, model, likelihood, inside_ll, outside_ll, edge_inside_ll, model_data, responsibilities, leaf_responsibility);

        double llh_before = 0.0;
        for (size_t character = 0; character < data.num_characters; ++character) {
            llh_before += likelihood[character];
        }

        spdlog::info("EM iteration: {} Log likelihood: {}", i, llh_before);
        spdlog::info("Nu: {} Phi: {}", model.parameters[0], model.parameters[1]);
        laml_m_step_objective fun(responsibilities, leaf_responsibility, num_missing, num_not_missing);
        
        double fx;
        int niter = solver.minimize(fun, params, fx, lb, ub);

        model.parameters[0] = params[0];
        model.parameters[1] = params[1];
        for (size_t i = 0; i < t.num_nodes; ++i) {
            t.branch_lengths[i] = params[i + 2];
        }

        // update model parameters
        model_data = model.initialize_data(&internal_comp_buffer, t.branch_lengths);
        likelihood = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);

        double llh_after = 0.0;
        for (size_t character = 0; character < data.num_characters; ++character) {
            llh_after += likelihood[character];
        }

        if (abs(llh_after - llh_before) / abs(llh_before) < EM_STOPPING_CRITERION) {
            break;
        }
    }

    return 0.0;
}

int main(int argc, char ** argv) {
    auto console_logger = spdlog::stdout_color_mt("fastlaml");
    auto error_logger = spdlog::stderr_color_mt("error");
    spdlog::set_default_logger(console_logger);

    argparse::ArgumentParser program(
        "fastlaml",
        std::to_string(FASTLAML_VERSION_MAJOR) + "." + std::to_string(FASTLAML_VERSION_MINOR),
        argparse::default_arguments::help
    );

    program.add_argument("--version")
        .action([&](const auto & /*unused*/) {
            std::cout << "fastppm version " << FASTLAML_VERSION_MAJOR << "." << FASTLAML_VERSION_MINOR << std::endl;
            std::exit(0);
        })
        .default_value(false)
        .help("prints version information and exits")
        .implicit_value(true)
        .nargs(0);

    program.add_argument("-m", "--mutation-priors")
        .help("Path to the mutation priors file")
        .default_value(std::string(""));

    program.add_argument("-c", "--character-matrix")
        .help("Path to the character matrix file (CSV)")
        .required();

    program.add_argument("-t", "--tree")
        .help("Path to the tree file")
        .required();

    program.add_argument("-o", "--output")
        .help("Path to the output file")
        .required();

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    spdlog::info("Loading tree from Newick file...");
    tree t = parse_newick_tree(program.get<std::string>("--tree"));
    
    // Check if tree is binary
    bool is_binary = true;
    for (size_t node_id = 0; node_id < t.num_nodes; ++node_id) {
        if (node_id != t.root_id && t.tree.in_degree(node_id) != 1) {
            is_binary = false;
            break;
        }
        
        size_t out_degree = t.tree.out_degree(node_id);
        if (out_degree != 0 && out_degree != 2) {
            is_binary = false;
            break;
        }
    }
    
    if (!is_binary) {
        spdlog::error("Input tree is not binary. Each node must have exactly 0 or 2 children.");
        std::exit(1);
    }

    spdlog::info("Processing character matrix and mutation priors...");

    phylogeny_data data = process_phylogeny_data(
        t, 
        program.get<std::string>("--character-matrix"),
        program.get<std::string>("--mutation-priors")
    );
    
    
    auto start = std::chrono::high_resolution_clock::now();
    double llh = laml_expectation_maximization(t, data, 0.5, 0.5);
    auto end = std::chrono::high_resolution_clock::now();
    double runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    spdlog::info("Log likelihood: {}", llh);
    spdlog::info("Computation time: {} ms", runtime);

    return 0;
}