#ifndef TOPOLOGY_SEARCH_H
#define TOPOLOGY_SEARCH_H

#include <thread>
#include <atomic>
#include <vector>
#include <random>
#include <cmath>

#include "phylogeny.h"
#include "laml_em.h"
#include "models/laml.h"
#include "io.h"

#include <spdlog/spdlog.h>

/* 
 * Defines an NNI operation which swaps
 * the subtrees rooted at u and v.
 */
struct nni {
    int u;
    int v;
};

template<typename D>
struct nni_thread_data {
    tree t;
    D model_data;
    std::vector<nni> nni_moves;
};

void stochastically_perturb_tree(
    tree& t, 
    int nni_count,
    std::mt19937& gen
) {
    std::uniform_int_distribution<int> dist(0, t.num_nodes - 1);
    for (int i = 0; i < nni_count; ++i) {
        int node_id = dist(gen);

        if (node_id == (int) t.root_id || t.tree.out_degree(node_id) == 0) { // skip root and leaves
            continue;
        }

        int p_id = t.tree.predecessors(node_id)[0];
        int w_id = t.tree.successors(p_id)[0];
        if (w_id == (int) node_id) {
            w_id = t.tree.successors(p_id)[1];
        }

        // choose random child
        std::vector<int> children = t.tree.successors(node_id);
        int u_id = children[dist(gen) % children.size()];

        // perform NNI move
        int parent_u = t.tree.predecessors(u_id)[0];
        int parent_v = t.tree.predecessors(w_id)[0];

        t.tree.remove_edge(parent_u, u_id);
        t.tree.remove_edge(parent_v, w_id);
        t.tree.add_edge(parent_u, w_id);
        t.tree.add_edge(parent_v, u_id);
    }
}

template<typename D>
double evaluate_single_nni(const std::function<double(tree&, D&)>& scoring_function,
                           tree&                           t,
                           D&                              model,
                           const nni&                      move)
{
    auto [u, v]  = move;
    int parent_u = t.tree.predecessors(u)[0];
    int parent_v = t.tree.predecessors(v)[0];

    // apply move
    t.tree.remove_edge(parent_u, u);
    t.tree.remove_edge(parent_v, v);
    t.tree.add_edge(parent_u, v);
    t.tree.add_edge(parent_v, u);

    // score current topology
    double log_likelihood = scoring_function(t, model);

    // revert move
    t.tree.remove_edge(parent_u, v);
    t.tree.remove_edge(parent_v, u);
    t.tree.add_edge(parent_u, u);
    t.tree.add_edge(parent_v, v);

    return log_likelihood;
}

template<typename D>
std::vector<std::pair<nni, double>>
evaluate_nnis(const std::function<double(tree&, D&)>& scoring_function,
              tree&                                   t,
              D&                                      model,
              const std::vector<nni>&                 nni_moves,
              std::atomic<int>&                       nni_counter,
              int                                     total_nni_moves)
{
    std::vector<std::pair<nni, double>> evaluations;
    evaluations.reserve(nni_moves.size());

    for (const auto& move : nni_moves) {
        if (nni_counter % 100 == 1) {
            spdlog::info("Evaluated {}/{} NNI moves", nni_counter.load(), total_nni_moves);
        }

        double score = evaluate_single_nni(scoring_function, t, model, move);
        evaluations.emplace_back(move, score);
        ++nni_counter;
    }

    return evaluations;
}

inline std::vector<nni> compute_nni_neighborhood(const tree& t) {
    std::vector<nni> nni_moves;
    for (size_t node_id = 0; node_id < t.num_nodes; ++node_id) {
        if (node_id == t.root_id || t.tree.out_degree(node_id) == 0) { // skip root and leaves
            continue;
        }

        int p_id = t.tree.predecessors(node_id)[0];
        int w_id = t.tree.successors(p_id)[0];
        if (w_id == (int) node_id) {
            w_id = t.tree.successors(p_id)[1];
        }

        for (int u_id : t.tree.successors(node_id)) {
            nni_moves.push_back({w_id, u_id});
        }
    }
    return nni_moves;
}

/**
 * @brief Evaluates the nearest neighbor interchange (NNI) neighborhood of a given tree.
 * 
 * This function computes scores for every possible NNI move from the initial tree.
 * NNI moves are topological changes to the tree that involve swapping subtrees.
 * 
 * @tparam D The type of the evolutionary model.
 * @param scoring_function A function that computes a score (e.g., log-likelihood) for a tree and model.
 *                         Must not alter branch lengths or model parameters.
 * @param initial_tree The starting tree topology. Must be binary (each internal node has exactly two children).
 * @param initial_model The initial model parameters.
 * @param threads Number of threads to use for parallel evaluation. If <= 1, runs sequentially.
 * 
 * @return A vector of pairs, each containing an NNI move and its corresponding score.
 *         Higher scores typically indicate better tree topologies (e.g., when scoring_function 
 *         returns log-likelihood values).
 */
template<typename D>
std::vector<std::pair<nni, double>> evaluate_nni_neighborhood(
    const std::function<double(tree&, D&)>& scoring_function, // does not alter branch lengths or model parameters
    const tree& initial_tree, // tree MUST be binary,
    const D& initial_model,
    int threads = 8
) {
    // compute initial likelihood and parameter estimates
    tree t  = initial_tree;
    D model = initial_model;
    double log_likelihood = scoring_function(t, model);
    spdlog::info("Initial log likelihood: {}", log_likelihood);

    std::vector<nni> nni_moves = compute_nni_neighborhood(t);
    spdlog::info("Root ID: {}", t.root_id);
    
    std::vector<std::pair<nni, double>> neighborhood;
    std::atomic<int> nni_counter(0);

    if (threads <= 1) {
        neighborhood = evaluate_nnis(scoring_function, t, model, nni_moves, nni_counter, nni_moves.size());
    } else {
        std::vector<std::thread> thread_pool;
        std::vector<std::vector<nni>> thread_nni_moves(threads);
        std::vector<std::vector<std::pair<nni, double>>> thread_results(threads);

        for (size_t i = 0; i < nni_moves.size(); ++i) {
            thread_nni_moves[i % threads].push_back(nni_moves[i]);
        }
        
        for (int i = 0; i < threads; ++i) {
            thread_pool.emplace_back([&, i]() {
                tree thread_tree = t;
                D thread_model = model;
                thread_results[i] = evaluate_nnis(scoring_function, thread_tree, thread_model, thread_nni_moves[i], nni_counter, nni_moves.size());
            });
        }
        
        for (auto& thread : thread_pool) {
            thread.join();
        }
        
        for (const auto& result : thread_results) {
            neighborhood.insert(neighborhood.end(), result.begin(), result.end());
        }
    }

    return neighborhood;
}

struct hill_climbing_result {
    tree best_tree;
    double log_likelihood;
    size_t iterations;
    std::vector<double> log_likelihoods;
};

inline hill_climbing_result simulated_annealing(
    const tree& initial_tree,
    const phylogeny_data& data,
    double inital_phi,
    double initial_nu,
    unsigned int max_iterations,
    unsigned int num_threads,
    bool is_ultrametric,
    double min_branch_length,
    double T0 = 0.1,
    bool no_silencing = false
) {
    // Initialize simulated annealing parameters, inheriting from LAML
    const double alpha = 0.99;
    size_t no_accepts = 0;
    size_t no_improve_counter = 0;
    const size_t max_no_improve = 100; // number of small-improvement moves allowed
    const double eta = 1e-8; // minimum improvement

    tree current_tree = initial_tree;
    laml_model model(data.character_matrix, data.observation_matrix, data.mutation_priors, inital_phi, initial_nu, data.data_type, is_ultrametric, min_branch_length, 1.0, no_silencing);
    auto initial_result = laml_expectation_maximization(current_tree, model, 100, true);
    double current_log_likelihood = initial_result.log_likelihood;

    spdlog::info("Starting simulated annealing with initial log likelihood: {}", current_log_likelihood);

    size_t iteration = 0;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<double> log_likelihoods;
    log_likelihoods.push_back(current_log_likelihood);

    // assigns a uniform weight to each nni
    int nni_neighborhood_size = compute_nni_neighborhood(current_tree).size();
    std::vector<double> nni_weights(nni_neighborhood_size, 1.0);
    std::discrete_distribution<> nni_sampler(nni_weights.begin(), nni_weights.end());

    std::uniform_real_distribution<> annealing_sampler(0, 1);

    while (iteration < max_iterations) {
        bool move_accepted = false;
        std::vector<nni> neighborhood = compute_nni_neighborhood(current_tree);
        nni sampled_move = neighborhood[nni_sampler(gen)];

        auto [u, v]  = sampled_move;
        int parent_u = current_tree.tree.predecessors(u)[0];
        int parent_v = current_tree.tree.predecessors(v)[0];

        // apply move
        current_tree.tree.remove_edge(parent_u, u);
        current_tree.tree.remove_edge(parent_v, v);
        current_tree.tree.add_edge(parent_u, v);
        current_tree.tree.add_edge(parent_v, u);

        // score current topology
        auto blens = current_tree.branch_lengths;
        auto params = model.parameters;

        double move_log_likelihood = laml_expectation_maximization(current_tree, model, 100, false).log_likelihood;
        double delta = (move_log_likelihood  - current_log_likelihood);
        double relative_improvement = (delta)/abs(current_log_likelihood);

        // Compute temperature and acceptance probability according to schedule
        double T = T0 * std::pow(alpha, no_accepts);  // alpha ~ 0.95, T0 ~ 1.0
        double proposal = std::exp(delta / T);

        if (delta > 0 || proposal > annealing_sampler(gen)) {
            current_log_likelihood = move_log_likelihood;
            move_accepted = true;

            spdlog::info(
                "Iteration {}: Applied NNI move ({}, {}), no. accepts: {}, new log likelihood: {}, current phi: {}, current nu: {}",
                iteration, sampled_move.u, sampled_move.v, no_accepts, current_log_likelihood, model.parameters[1], model.parameters[0]
            );
        } else {
            spdlog::info(
                "Iteration {}: Rejected NNI move ({}, {}), proposed log likelihood: {}",
                iteration, sampled_move.u, sampled_move.v, current_log_likelihood
            );

            // revert move
            current_tree.tree.remove_edge(parent_u, v);
            current_tree.tree.remove_edge(parent_v, u);
            current_tree.tree.add_edge(parent_u, u);
            current_tree.tree.add_edge(parent_v, v);

            // revert parameter changes
            current_tree.branch_lengths = blens;
            model.parameters = params;
        }

        log_likelihoods.push_back(current_log_likelihood);
        iteration++; // advance iterations only after successful moves
        if (move_accepted) {
            no_accepts++;

            //if (std::abs(delta) < eta) {
            //   ++no_improve_counter;
            //   if (no_improve_counter >= max_no_improve) {
            //       spdlog::info("Terminating: {} small improvements below η = {}", max_no_improve, eta);
            //       break;
            //    }
            //} else {
            //  no_improve_counter = 0;  // reset if significant jump
            //}
        } else {
            // No move accepted — check for early termination
            if (neighborhood.empty()) { // || std::abs(relative_improvement) < eta) {
                //spdlog::info("Terminating: no acceptable move or improvement < η at iteration {}", iteration);
                spdlog::info("Terminating: no acceptable move at iteration {}", iteration);
                break;
            }
        }

    }

    spdlog::info("Simulated annealing completed after {} iterations. Final log likelihood: {}",
             iteration, current_log_likelihood);

    return {current_tree, current_log_likelihood, iteration, log_likelihoods};
}

#endif
