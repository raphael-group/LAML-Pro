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
void optimize_parameters(tree& t, const phylogeny_data& data, unsigned int seed, std::string output_prefix) {
    spdlog::info("Optimizing model parameters and branch lengths...");

    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.05f, 0.95f);
    
    double initial_phi = dist(gen);
    double initial_nu = dist(gen);
    for (size_t i = 0; i < t.branch_lengths.size(); ++i) {
        t.branch_lengths[i] = dist(gen);
    }

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

    // TODO: save the posterior probabilities
    

    auto newick_tree = write_newick_tree(t);
    std::ofstream output_file(output_prefix + "_tree.newick");
    if (output_file.is_open()) {
        output_file << newick_tree;
        output_file.close();
        spdlog::info("Optimized tree written to {}", output_prefix + "_tree.newick");
    } else {
        spdlog::error("Could not open file for writing: {}", output_prefix + "_tree.newick");
    }

    spdlog::info("Fitting branch lengths to ultrametric tree...");
    ultrametric_projection(t);
    newick_tree = write_newick_tree(t);
    output_file.open(output_prefix + "_ultrametric_tree.newick");
    if (output_file.is_open()) {
        output_file << newick_tree;
        output_file.close();
        spdlog::info("Ultrametric tree written to {}", output_prefix + "_ultrametric_tree.newick");
    } else {
        spdlog::error("Could not open file for writing: {}", output_prefix + "_ultrametric_tree.newick");
    }
    

    json output_json;
    output_json["phi"] = model.parameters[1];
    output_json["nu"] = model.parameters[0];
    output_json["em_iterations"] = em_res.num_iterations;
    output_json["log_likelihood"] = em_res.log_likelihood;
    output_json["command"] = command;
    output_json["git_commit"] = GIT_COMMIT_HASH;

    std::ofstream json_file(output_prefix + "_results.json");
    if (json_file.is_open()) {
        json_file << output_json.dump(4);
        json_file.close();
        spdlog::info("Optimization results written to {}", output_prefix + "_results.json");
    } else {
        spdlog::error("Could not open file for writing: {}", output_prefix + "_results.json");
    }

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
    // #laml_model model = laml_model(data.character_matrix, data.mutation_priors, inital_phi, initial_nu);
    laml_model model = laml_model(data.character_matrix, data.observation_matrix, data.mutation_priors, initial_phi, initial_nu, data.data_type);
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
        auto selected_move_likelihood = filtered_neighborhood[selected_move_index].second;
       
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
    auto result = greedy_hill_climbing(t, data, initial_phi, initial_nu, max_iterations, num_threads, 0.1, temp);
    t = result.best_tree;

    initial_phi = dist(gen);
    initial_nu = dist(gen);
    laml_model model = laml_model(data.character_matrix, data.observation_matrix, data.mutation_priors, initial_phi, initial_nu, data.data_type);
    auto em_res = laml_expectation_maximization(t, model, 100, false);
    double current_phi = model.parameters[1];
    double current_nu = model.parameters[0];
    spdlog::info("Best log likelihood: {}", em_res.log_likelihood);

    auto end = std::chrono::high_resolution_clock::now();
    double runtime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    spdlog::info("Total runtime: {} ms", runtime);

    spdlog::info("Writing best tree to file...");
    std::string newick_tree = write_newick_tree(t);
    std::ofstream output_file(output_prefix + "_tree.newick");
    if (output_file.is_open()) {
        output_file << newick_tree;
        output_file.close();
        spdlog::info("Best tree written to {}", output_prefix + "_tree.newick");
    } else {
        spdlog::error("Could not open file for writing: {}", output_prefix + "_tree.newick");
    }

    spdlog::info("Fitting branch lengths to ultrametric tree...");
    ultrametric_projection(t);
    newick_tree = write_newick_tree(t);
    output_file.open(output_prefix + "_ultrametric_tree.newick");
    if (output_file.is_open()) {
        output_file << newick_tree;
        output_file.close();
        spdlog::info("Ultrametric tree written to {}", output_prefix + "_ultrametric_tree.newick");
    } else {
        spdlog::error("Could not open file for writing: {}", output_prefix + "_ultrametric_tree.newick");
    }

    json output_json;
    output_json["phi"] = current_phi;
    output_json["nu"] = current_nu;
    output_json["log_likelihood"] = em_res.log_likelihood;
    output_json["runtime"] = runtime;
    output_json["iterations"] = result.iterations;
    output_json["log_likelihoods"] = result.log_likelihoods;

    std::ofstream json_file(output_prefix + "_results.json");
    if (json_file.is_open()) {
        json_file << output_json.dump(4);
        json_file.close();
        spdlog::info("Search results written to {}", output_prefix + "_results.json");
    } else {
        spdlog::error("Could not open file for writing: {}", output_prefix + "_results.json");
    }
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
        .default_value(100U)
        .scan<'u', unsigned int>();

    program.add_argument("--temp")
        .help("Temperature for simulated annealing")
        .default_value(0.001)
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
        file_sink->set_level(spdlog::level::debug);  // Log everything to file

        // Combine them into a multi-sink logger
        auto logger = std::make_shared<spdlog::logger>("multi_logger", spdlog::sinks_init_list{console_sink, file_sink});
        spdlog::set_default_logger(logger);
        spdlog::set_level(spdlog::level::debug); // Set global log level
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
            break;
        }
        
        size_t out_degree = t.tree.out_degree(node_id);
        if (out_degree != 0 && out_degree != 2) {
            is_binary = false;
            break;
        }
    }
    
    if (!is_binary) {
        spdlog::error("Input tree is not binary.");
        return 1;
    }

    spdlog::info("Processing character matrix and mutation priors...");

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
