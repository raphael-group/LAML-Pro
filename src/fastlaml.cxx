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
#include <nlohmann/json.hpp>

#include "ultrametric.h"
#include "digraph.h"
#include "io.h"

#include "laml_em.h"
#include "phylogenetic_model.h"
#include "phylogeny.h"
#include "models/laml.h"
#include "topology_search.h"

#include <argparse/argparse.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/fmt/ostr.h>

#include <spdlog/sinks/basic_file_sink.h>

#define FASTLAML_VERSION_MAJOR 1
#define FASTLAML_VERSION_MINOR 0
#ifndef GIT_COMMIT_HASH
#define GIT_COMMIT_HASH "unknown"
#endif

std::string command;

using json = nlohmann::json;

void write_tree(std::string newick_tree, const std::string& filename) {
    FILE* out = std::fopen(filename.c_str(), "w");
    if (!out) {
        spdlog::error("Could not open file for writing: {}", filename);
        return;
    }

    std::fprintf(out, "%s", newick_tree.c_str());
    std::fclose(out);

    spdlog::info("Optimized tree written to {}", filename);
}

void write_results(
    const tree& t, 
    const std::string& prefix, 
    const em_results& em_res,
    const laml_model& model, 
    const std::string& command, 
    const double runtime,
    const std::vector<double>& log_likelihoods
) {
    size_t num_characters = em_res.posterior_llh.size();

    if (model.data_type == "observation-matrix") {
        spdlog::info("data_type: observation-matrix, writing posterior probs and argmax probs.");
        // --- Posterior Matrix ---
        FILE* out = std::fopen((prefix + "_posterior_probs.csv").c_str(), "w");
        if (!out) {
            spdlog::error("Failed to open output file for posterior matrix.");
            return;
        }

        std::fprintf(out, "node");
        for (size_t c = 0; c < num_characters; ++c)
            std::fprintf(out, ",character_%zu", c);
        std::fprintf(out, "\n");

        for (size_t node_id = 0; node_id < t.node_names.size(); ++node_id) {
            spdlog::debug("node_id = {}, name = {}", node_id, t.node_names[node_id]);
            std::fprintf(out, "%s", t.node_names[node_id].c_str());
            for (size_t c = 0; c < num_characters; ++c) {
                std::ostringstream cell;
                size_t alphabet_size = em_res.posterior_llh[c][node_id].size();
                for (size_t s = 0; s < alphabet_size; ++s) {
                    int state_label = static_cast<int>(s) - 1;
                    cell << state_label << ":" << std::fixed << std::setprecision(6) << em_res.posterior_llh[c][node_id][s];
                    if (s + 1 < alphabet_size) cell << "/";
                }
                std::fprintf(out, ",%s", cell.str().c_str());
            }
            std::fprintf(out, "\n");
        }
        std::fclose(out);
        spdlog::info("Posterior matrix written to {}_posterior_probs.csv", prefix);

        // --- Argmax Matrix ---
        FILE* out2 = std::fopen((prefix + "_posterior_argmax.csv").c_str(), "w");
        if (!out2) {
            spdlog::error("Failed to open output file for argmax matrix.");
            return;
        }
        std::string newick_tree = write_newick_tree(t) + ";";  // Assuming your tree object has this method
        std::fprintf(out2, "Newick Tree:\n%s\n", newick_tree.c_str());

        std::fprintf(out2, "node");
        for (size_t c = 0; c < num_characters; ++c)
            std::fprintf(out2, ",character_%zu", c);
        std::fprintf(out2, "\n");

        for (size_t node_id = 0; node_id < t.node_names.size(); ++node_id) {
            spdlog::debug("node_id = {}, name = {}", node_id, t.node_names[node_id]);
            std::fprintf(out2, "%s", t.node_names[node_id].c_str());
            for (size_t c = 0; c < num_characters; ++c) {
                const auto& probs = em_res.posterior_llh[c][node_id];
                auto max_it = std::max_element(probs.begin(), probs.end());
                size_t best_index = std::distance(probs.begin(), max_it);
                int state_label = static_cast<int>(best_index) - 1;
                std::fprintf(out2, ",%d", state_label);
            }
            std::fprintf(out2, "\n");
        }

        std::fclose(out2);
        spdlog::info("Argmax matrix written to {}_posterior_argmax.csv", prefix);
    }

    // --- JSON Summary ---
    json output_json; // need to pass in command, model
    output_json["phi"] = model.parameters[1];
    output_json["nu"] = model.parameters[0];
    output_json["em_iterations"] = em_res.num_iterations;
    output_json["best_log_likelihood"] = em_res.log_likelihood;
    output_json["command"] = command;
    output_json["git_commit"] = GIT_COMMIT_HASH;
    output_json["runtime_ms"] = runtime;
    output_json["log_likelihoods"] = log_likelihoods;

    FILE* jout = std::fopen((prefix + "_results.json").c_str(), "w");
    if (!jout) {
        spdlog::error("Could not open file to write JSON results: {}_results.json", prefix);
    } else {
        std::string serialized = output_json.dump(4);  // pretty-printed
        std::fprintf(jout, "%s\n", serialized.c_str());
        std::fclose(jout);
        spdlog::info("JSON summary written to {}_results.json", prefix);
    }
}

void optimize_parameters(tree& t, const phylogeny_data& data, unsigned int seed, std::string output_prefix) {
    spdlog::info("Optimizing model parameters and branch lengths...");

    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.05f, 0.95f);
    
    double initial_phi = dist(gen);
    double initial_nu = dist(gen);
    for (size_t i = 0; i < t.branch_lengths.size(); ++i) {
        t.branch_lengths[i] = dist(gen);
    }

    auto start = std::chrono::high_resolution_clock::now();
    // label internal nodes in t
    for (size_t node_id = 0; node_id < t.num_nodes; ++node_id) {
        if (node_id >= t.node_names.size() || t.node_names[node_id].empty()) {
            if (node_id >= t.node_names.size()) {
                t.node_names.resize(t.num_nodes);  // Ensure size matches num_nodes
            }
            t.node_names[node_id] = "internal_" + std::to_string(node_id);
        }
    }

    laml_model model = laml_model(data.character_matrix, data.observation_matrix, data.mutation_priors, initial_phi, initial_nu, data.data_type);
    auto em_res = laml_expectation_maximization(t, model, 100, true);
    
    auto end = std::chrono::high_resolution_clock::now();
    double runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    spdlog::info("Total runtime: {} ms", runtime);

    auto newick_tree = write_newick_tree(t);
    write_tree(newick_tree, output_prefix + "_tree.newick");

    spdlog::info("Fitting branch lengths to ultrametric tree...");
    ultrametric_projection(t);
    newick_tree = write_newick_tree(t);
    write_tree(newick_tree, output_prefix + "_ultrametric_tree.newick");

    write_results(t, output_prefix, em_res, model, command, runtime, std::vector<double>());

    spdlog::info("Optimization completed. Log likelihood: {}", em_res.log_likelihood);
}

struct hill_climbing_result {
    tree best_tree;
    double log_likelihood;
    size_t iterations;
    std::vector<double> log_likelihoods;
};

hill_climbing_result greedy_hill_climbing(
    const tree& initial_tree, 
    const phylogeny_data& data, 
    double initial_phi,  // fixed typo
    double initial_nu, 
    unsigned int max_iterations,
    unsigned int num_threads,
    double tolerance = 0.1,
    double temp = 0.1
) {
    tree best_tree = initial_tree;
    laml_model model(data.character_matrix, data.observation_matrix, data.mutation_priors, initial_phi, initial_nu, data.data_type);
    auto initial_result = laml_expectation_maximization(best_tree, model, 100, true);
    double best_log_likelihood = initial_result.log_likelihood;
    
    spdlog::info("Starting hill climbing with initial log likelihood: {}", best_log_likelihood);
    
    bool improved = true;
    size_t iteration = 0;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::function<double(tree& t, laml_model& model)> scoring_function = [&](tree& t, laml_model& model) {
        auto blens = t.branch_lengths;
        auto params = model.parameters;

        double score = laml_expectation_maximization(t, model, 100, false).log_likelihood;
        t.branch_lengths = blens;
        model.parameters = params;
        return score;
    };

    std::vector<double> log_likelihoods;
    log_likelihoods.push_back(best_log_likelihood);
    while (iteration < max_iterations && improved) {
        iteration++;
        improved = false;
        
        // Evaluate entire NNI neighborhood
        std::vector<std::pair<nni, double>> neighborhood = evaluate_nni_neighborhood(
            scoring_function, best_tree, model, num_threads
        );
        
        std::vector<std::pair<nni, double>> filtered_neighborhood;
        std::vector<double> weights;
        for (const auto& [move, log_likelihood] : neighborhood) {
            if (log_likelihood > best_log_likelihood - tolerance) {
                filtered_neighborhood.push_back({move, log_likelihood});
                double weight = std::exp((log_likelihood - best_log_likelihood) / temp);
                weights.push_back(weight);
            }
        }

        if (filtered_neighborhood.empty()) {
            spdlog::info("No valid NNI moves found, stopping hill climbing");
            break;
        }

        std::discrete_distribution<> dist(weights.begin(), weights.end());
        auto selected_move_index = dist(gen);
        auto selected_move = filtered_neighborhood[selected_move_index].first;
       
        int parent_u = best_tree.tree.predecessors(selected_move.u)[0];
        int parent_v = best_tree.tree.predecessors(selected_move.v)[0];
        
        best_tree.tree.remove_edge(parent_u, selected_move.u);
        best_tree.tree.remove_edge(parent_v, selected_move.v);
        best_tree.tree.add_edge(parent_u, selected_move.v);
        best_tree.tree.add_edge(parent_v, selected_move.u);
    
        auto result = laml_expectation_maximization(best_tree, model, 100, false);
        
        double improvement = result.log_likelihood - best_log_likelihood;
        best_log_likelihood = result.log_likelihood;
        log_likelihoods.push_back(best_log_likelihood);

        spdlog::info("Iteration {}: Applied NNI move ({}, {}), new log likelihood: {}, improvement: {}, current phi: {}, current nu: {}",
            iteration, selected_move.u, selected_move.v, best_log_likelihood, improvement, model.parameters[0], model.parameters[1]); 
        
        improved = true;
    }
    
    spdlog::info("Hill climbing completed after {} iterations. Final log likelihood: {}", 
             iteration, best_log_likelihood);
    
    return {best_tree, best_log_likelihood, iteration, log_likelihoods};
}

hill_climbing_result fixed_temp_simulated_annealing (
    const tree& initial_tree, 
    const phylogeny_data& data, 
    double inital_phi, 
    double initial_nu, 
    unsigned int max_iterations,
    unsigned int num_threads,
    double temp = 0.00001
) {
    tree current_tree = initial_tree;
    laml_model model(data.character_matrix, data.observation_matrix, data.mutation_priors, inital_phi, initial_nu, data.data_type);
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
        double proposal = std::exp(((move_log_likelihood  - current_log_likelihood) / (temp * std::abs(current_log_likelihood))));
    
        if (proposal > annealing_sampler(gen)) { // probabilistic accept
            current_log_likelihood = move_log_likelihood;

            spdlog::info(
                "Iteration {}: Applied NNI move ({}, {}), new log likelihood: {}, current phi: {}, current nu: {}",
                iteration, sampled_move.u, sampled_move.v, current_log_likelihood, model.parameters[0], model.parameters[1]
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
        iteration++;
    }
    
    spdlog::info("Fixed temperature simulated annealing completed after {} iterations. Final log likelihood: {}", 
             iteration, current_log_likelihood);
    
    return {current_tree, current_log_likelihood, iteration, log_likelihoods};
}

hill_climbing_result simulated_annealing(
    const tree& initial_tree, 
    const phylogeny_data& data, 
    double inital_phi, 
    double initial_nu, 
    unsigned int max_iterations,
    unsigned int num_threads,
    double temp = 0.00001 
) {
    // Cool the temperature with smooth decay. TODO: take out the temp parameer
    
    // Initialize simulated annealing parameters, inheriting from LAML
    const double epsilon = 1e-12;
    const double alpha = 0.99; //1.0;
    size_t no_accepts = 0;
    size_t no_improve_counter = 0;
    const size_t max_no_improve = 100; // number of small-improvement moves allowed
    const double eta = 1e-8; // minimum improvement
    const double T0 = 1.0; // starting temperature

    tree current_tree = initial_tree;
    laml_model model(data.character_matrix, data.observation_matrix, data.mutation_priors, inital_phi, initial_nu, data.data_type);
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

        // #double T = std::max(epsilon, (std::pow(alpha, iteration) - std::pow(alpha, c)) / (1.0 - std::pow(alpha, c)));
        //double proposal = std::min(1.0, std::exp(delta - epsilon) / T);
        // double proposal = std::exp(((move_log_likelihood  - current_log_likelihood) / (temp * std::abs(current_log_likelihood))));

        if (delta > 0 || proposal > annealing_sampler(gen)) {
            current_log_likelihood = move_log_likelihood;
            move_accepted = true;

            // if (proposal > annealing_sampler(gen)) { // probabilistic accept
            spdlog::info(
                "Iteration {}: Applied NNI move ({}, {}), no. accepts: {}, new log likelihood: {}, current phi: {}, current nu: {}",
                iteration, sampled_move.u, sampled_move.v, no_accepts, current_log_likelihood, model.parameters[0], model.parameters[1]
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

            if (std::abs(delta) < eta) {
                ++no_improve_counter;
                if (no_improve_counter >= max_no_improve) {
                    spdlog::info("Terminating: {} small improvements below η = {}", max_no_improve, eta);
                    break;
                }
            } else {
                no_improve_counter = 0;  // reset if significant jump
            }
        } else {
            // No move accepted — check for early termination
            if (neighborhood.empty() || std::abs(relative_improvement) < eta) {
                spdlog::info("Terminating: no acceptable move or improvement < η at iteration {}", iteration);
                break;
            }
        }

    }
    
    spdlog::info("Simulated annealing completed after {} iterations. Final log likelihood: {}", 
             iteration, current_log_likelihood);
    
    return {current_tree, current_log_likelihood, iteration, log_likelihoods};
}

void search_optimal_tree(
    tree& t, const phylogeny_data& data, unsigned int seed, 
    unsigned int num_threads, std::string output_prefix, size_t max_iterations, double temp
) {
    spdlog::info("Searching for optimal tree...");

    auto start = std::chrono::high_resolution_clock::now();

    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.05f, 0.95f);

    // randomly select phi and nu
    double initial_phi = dist(gen);
    double initial_nu = dist(gen);

    // perform hill climbing
    auto result = simulated_annealing(t, data, initial_phi, initial_nu, max_iterations, num_threads, temp);
    t = result.best_tree;

    initial_phi = dist(gen);
    initial_nu = dist(gen);
    laml_model model = laml_model(data.character_matrix, data.observation_matrix, data.mutation_priors, initial_phi, initial_nu, data.data_type);
    auto em_res = laml_expectation_maximization(t, model, 100, false);
    
    spdlog::info("Best log likelihood: {}", em_res.log_likelihood);

    auto end = std::chrono::high_resolution_clock::now();
    double runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    spdlog::info("Total runtime: {} ms", runtime);
    
    // label internal nodes in t
    for (size_t node_id = 0; node_id < t.num_nodes; ++node_id) {
        if (node_id >= t.node_names.size() || t.node_names[node_id].empty()) {
            if (node_id >= t.node_names.size()) {
                t.node_names.resize(t.num_nodes);  // Ensure size matches num_nodes
            }
            t.node_names[node_id] = "internal_" + std::to_string(node_id);
        }
    }

    spdlog::info("Writing best tree to file...");
    auto newick_tree = write_newick_tree(t);
    write_tree(newick_tree, output_prefix + "_tree.newick");

    spdlog::info("Fitting branch lengths to ultrametric tree...");
    ultrametric_projection(t);
    newick_tree = write_newick_tree(t);
    write_tree(newick_tree, output_prefix + "_ultrametric_tree.newick");
    
    write_results(t, output_prefix, em_res, model, command, runtime, result.log_likelihoods);
    
    spdlog::info("Tree search and optimization completed. Log likelihood: {}", em_res.log_likelihood);
}

int main(int argc, char** argv) {
    auto console_logger = spdlog::stdout_color_mt("fastlaml");
    auto error_logger = spdlog::stderr_color_mt("error");
    spdlog::set_default_logger(console_logger);

    argparse::ArgumentParser program(
        "fastlaml",
        std::to_string(FASTLAML_VERSION_MAJOR) + "." + std::to_string(FASTLAML_VERSION_MINOR),
        argparse::default_arguments::help
    );
    
    command.clear();
    for (int i = 0; i < argc; ++i) {
        if (i > 0) command += " ";
        command += argv[i];
    }

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

    program.add_argument("-c", "--matrix")
        .help("Path to the matrix file (CSV)")
        .required();

    program.add_argument("-d", "--data-type")
        .help("String. Options are 'character-matrix' or 'observation-matrix'.")
        .nargs(1)
        .default_value(std::string("character-matrix")); 

    program.add_argument("-t", "--tree")
        .help("Path to the tree file")
        .required();

    program.add_argument("-o", "--output")
        .help("Path to the output file")
        .required();
    
    program.add_argument("-v", "--verbose")
        .help("Additionally save all console logs to a file automatically.")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("--threads")
        .help("number of threads to use")
        .default_value(std::thread::hardware_concurrency())
        .scan<'u', unsigned int>();

    program.add_argument("--mode")
        .help("Operation mode: 'optimize' for parameter optimization or 'search' for tree search")
        .default_value(std::string("optimize"))
        .choices("optimize", "search");

    program.add_argument("--seed")
        .help("Random seed for reproducibility")
        .default_value(73U)
        .scan<'u', unsigned int>();

    program.add_argument("--max-iterations")
        .help("Maximum number of iterations for hill climbing")
        .default_value(20000U)
        .scan<'u', unsigned int>();

    program.add_argument("--temp")
        .help("Temperature for fixed temp topology search")
        .default_value(0.00001)
        .scan<'g', double>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    if (program.get<bool>("--verbose")) {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::info);

        // Create file sink
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(program.get<std::string>("--output") + "_fastlaml.log", true);
        file_sink->set_level(spdlog::level::info); // debug); // Log everything to file

        // Combine them into a multi-sink logger
        auto logger = std::make_shared<spdlog::logger>("multi_logger", spdlog::sinks_init_list{console_sink, file_sink});
        spdlog::set_default_logger(logger);
        spdlog::set_level(spdlog::level::info); // debug); // Set global log level
        spdlog::info("Logger initialized");
        spdlog::info("Command: {}", command);
    
    }


    spdlog::info("Loading tree from Newick file...");
    tree t = parse_newick_tree(program.get<std::string>("--tree"));

    // Check if tree is binary
    bool is_binary = true;
    for (size_t node_id = 0; node_id < t.num_nodes; ++node_id) {
        if (node_id != t.root_id && t.tree.in_degree(node_id) != 1) {
            is_binary = false;
            spdlog::error("Input tree is not rooted with a node of in-degree 1.");
            break;
        }
        
        size_t out_degree = t.tree.out_degree(node_id);
        if (out_degree != 0 && out_degree != 2) {
            is_binary = false;
            spdlog::error("Input tree has nodes which are not of degree either 0 or 2.");
            break;
        }
    }
    
    if (!is_binary) {
        spdlog::error("Input tree is not binary.");
        return 1;
    }

    spdlog::info("Processing matrix and mutation priors...");

    phylogeny_data data = process_phylogeny_data(
        t, 
        program.get<std::string>("--matrix"),
        program.get<std::string>("--mutation-priors"),
        program.get<std::string>("--data-type")
    );
    
    unsigned int seed = program.get<unsigned int>("--seed");
    std::string mode = program.get<std::string>("--mode");
    
    if (mode == "optimize") {
        optimize_parameters(t, data, seed, program.get<std::string>("--output"));
    } else {
        search_optimal_tree(
            t, data, seed, 
            program.get<unsigned int>("--threads"), 
            program.get<std::string>("--output"),
            program.get<unsigned int>("--max-iterations"),
            program.get<double>("--temp")
        );
    }

    return 0;
}
