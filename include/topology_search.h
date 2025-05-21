#ifndef TOPOLOGY_SEARCH_H
#define TOPOLOGY_SEARCH_H

#include <thread>
#include <atomic>
#include <vector>

#include "phylogeny.h"
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

std::vector<nni> compute_nni_neighborhood(const tree& t) {
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

#endif