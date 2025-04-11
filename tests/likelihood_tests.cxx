#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdint>
#include "digraph.h"
#include "models/laml.h"
#include "phylogeny.h"
#include "laml_em.h"

std::pair<tree, laml_model> build_llh_unit_test(
    std::vector<std::vector<int>> character_matrix, 
    double phi, 
    double nu, 
    double mut_prior
) {
    // tree: ((a:1.0,b:1.0):1.0,c:1.0):1.0;
    digraph<size_t> tree_graph;
    for (int i = 0; i < 5; i++) {
        tree_graph.add_vertex(i);
    }
    
    // add edges (parent -> child)
    tree_graph.add_edge(4, 3); // root -> internal node
    tree_graph.add_edge(4, 2); // root -> leaf c
    tree_graph.add_edge(3, 1); // internal -> leaf a
    tree_graph.add_edge(3, 0); // internal -> leaf b
    
    tree t;
    t.num_leaves = 3;
    t.num_nodes = 5;
    t.root_id = 4;
    t.tree = tree_graph;
    t.branch_lengths = {1.0, 1.0, 1.0, 1.0, 1.0};
    t.node_names = {"c", "a", "b", "internal", "root"};
    
    std::vector<std::vector<double>> mutation_priors = {{mut_prior}};
    laml_model model(character_matrix, mutation_priors, nu, phi);
    model.alphabet_sizes[0] = 3;
    return {t, model};
}

TEST_CASE("INSIDE_OUTSIDE_TEST_LLH_1", "[insidellh]") {
    auto [t, model] = build_llh_unit_test({{1}, {1}, {1}}, 0.0, 0.0, 1.0);

    std::vector<double> buffer(model.alphabet_sizes[0]);
    std::vector<laml_data> model_data = model.initialize_data(t.tree, t.branch_lengths, &buffer);
    
    size_t num_characters = model.alphabet_sizes.size();
    size_t max_alphabet_size = model.alphabet_sizes[0];
    likelihood_buffer inside_ll(num_characters, max_alphabet_size, t.num_nodes);
    
    std::vector<double> llh = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);
    
    double expected_llh = -0.20665578828621584;
    REQUIRE(abs(llh[0] - expected_llh) < 1e-6);
}

TEST_CASE("INSIDE_OUTSIDE_TEST_LLH_2", "[insidellh]") {
    auto [t, model] = build_llh_unit_test({{1}, {1}, {0}}, 0.0, 0.0, 1.0);

    std::vector<double> buffer(model.alphabet_sizes[0]);
    std::vector<laml_data> model_data = model.initialize_data(t.tree, t.branch_lengths, &buffer);
    
    size_t num_characters = model.alphabet_sizes.size();
    size_t max_alphabet_size = model.alphabet_sizes[0];
    likelihood_buffer inside_ll(num_characters, max_alphabet_size, t.num_nodes);
    
    std::vector<double> llh = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);
    
    double expected_llh = -2.2495946917551692;
    REQUIRE(abs(llh[0] - expected_llh) < 1e-6);
}

TEST_CASE("INSIDE_OUTSIDE_TEST_LLH_3", "[insidellh]") {
    auto [t, model] = build_llh_unit_test({{1}, {0}, {1}}, 0.0, 0.0, 1.0);

    std::vector<double> buffer(model.alphabet_sizes[0]);
    std::vector<laml_data> model_data = model.initialize_data(t.tree, t.branch_lengths, &buffer);
    
    size_t num_characters = model.alphabet_sizes.size();
    size_t max_alphabet_size = model.alphabet_sizes[0];
    likelihood_buffer inside_ll(num_characters, max_alphabet_size, t.num_nodes);
    
    std::vector<double> llh = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);
    
    double expected_llh = -3.917350291274164;
    REQUIRE(abs(llh[0] - expected_llh) < 1e-6);
}

TEST_CASE("INSIDE_OUTSIDE_TEST_LLH_4", "[insidellh]") {
    auto [t, model] = build_llh_unit_test({{0}, {1}, {1}}, 0.0, 0.0, 1.0);

    std::vector<double> buffer(model.alphabet_sizes[0]);
    std::vector<laml_data> model_data = model.initialize_data(t.tree, t.branch_lengths, &buffer);
    
    size_t num_characters = model.alphabet_sizes.size();
    size_t max_alphabet_size = model.alphabet_sizes[0];
    likelihood_buffer inside_ll(num_characters, max_alphabet_size, t.num_nodes);
    
    std::vector<double> llh = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);
    
    double expected_llh = -3.917350291274164;
    REQUIRE(abs(llh[0] - expected_llh) < 1e-6);
}

TEST_CASE("INSIDE_OUTSIDE_TEST_LLH_5", "[insidellh]") {
    auto [t, model] = build_llh_unit_test({{1}, {0}, {0}}, 0.0, 0.0, 1.0);

    std::vector<double> buffer(model.alphabet_sizes[0]);
    std::vector<laml_data> model_data = model.initialize_data(t.tree, t.branch_lengths, &buffer);
    
    size_t num_characters = model.alphabet_sizes.size();
    size_t max_alphabet_size = model.alphabet_sizes[0];
    likelihood_buffer inside_ll(num_characters, max_alphabet_size, t.num_nodes);
    
    std::vector<double> llh = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);
    
    double expected_llh = -4.4586751457870815;
    REQUIRE(abs(llh[0] - expected_llh) < 1e-6);
}

TEST_CASE("INSIDE_OUTSIDE_TEST_LLH_6", "[insidellh]") {
    auto [t, model] = build_llh_unit_test({{0}, {1}, {0}}, 0.0, 0.0, 1.0);

    std::vector<double> buffer(model.alphabet_sizes[0]);
    std::vector<laml_data> model_data = model.initialize_data(t.tree, t.branch_lengths, &buffer);
    
    size_t num_characters = model.alphabet_sizes.size();
    size_t max_alphabet_size = model.alphabet_sizes[0];
    likelihood_buffer inside_ll(num_characters, max_alphabet_size, t.num_nodes);
    
    std::vector<double> llh = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);
    
    double expected_llh = -4.4586751457870815;
    REQUIRE(abs(llh[0] - expected_llh) < 1e-6);
}

TEST_CASE("INSIDE_OUTSIDE_TEST_LLH_7", "[insidellh]") {
    auto [t, model] = build_llh_unit_test({{0}, {0}, {1}}, 0.0, 0.0, 1.0);

    std::vector<double> buffer(model.alphabet_sizes[0]);
    std::vector<laml_data> model_data = model.initialize_data(t.tree, t.branch_lengths, &buffer);
    
    size_t num_characters = model.alphabet_sizes.size();
    size_t max_alphabet_size = model.alphabet_sizes[0];
    likelihood_buffer inside_ll(num_characters, max_alphabet_size, t.num_nodes);
    
    std::vector<double> llh = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);
    
    double expected_llh = -4.4586751457870815;
    REQUIRE(abs(llh[0] - expected_llh) < 1e-6);
}

TEST_CASE("INSIDE_OUTSIDE_TEST_LLH_8", "[insidellh]") {
    auto [t, model] = build_llh_unit_test({{0}, {0}, {0}}, 0.0, 0.0, 1.0);

    std::vector<double> buffer(model.alphabet_sizes[0]);
    std::vector<laml_data> model_data = model.initialize_data(t.tree, t.branch_lengths, &buffer);
    
    size_t num_characters = model.alphabet_sizes.size();
    size_t max_alphabet_size = model.alphabet_sizes[0];
    likelihood_buffer inside_ll(num_characters, max_alphabet_size, t.num_nodes);
    
    std::vector<double> llh = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);
    
    double expected_llh = -5.0;
    REQUIRE(abs(llh[0] - expected_llh) < 1e-6);
}

TEST_CASE("INSIDE_OUTSIDE_TEST_LLH_9", "[insidellh]") {
    auto [t, model] = build_llh_unit_test({{0}, {0}, {-1}}, 0.1, 0.0, 1.0);

    std::vector<double> buffer(model.alphabet_sizes[0]);
    std::vector<laml_data> model_data = model.initialize_data(t.tree, t.branch_lengths, &buffer);
    
    size_t num_characters = model.alphabet_sizes.size();
    size_t max_alphabet_size = model.alphabet_sizes[0];
    likelihood_buffer inside_ll(num_characters, max_alphabet_size, t.num_nodes);
    
    std::vector<double> llh = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);
    
    double expected_llh = -6.513306124309698;
    REQUIRE(abs(llh[0] - expected_llh) < 1e-6);
}

TEST_CASE("INSIDE_OUTSIDE_TEST_LLH_10", "[insidellh]") {
    auto [t, model] = build_llh_unit_test({{0}, {-1}, {0}}, 0.1, 0.0, 1.0);

    std::vector<double> buffer(model.alphabet_sizes[0]);
    std::vector<laml_data> model_data = model.initialize_data(t.tree, t.branch_lengths, &buffer);
    
    size_t num_characters = model.alphabet_sizes.size();
    size_t max_alphabet_size = model.alphabet_sizes[0];
    likelihood_buffer inside_ll(num_characters, max_alphabet_size, t.num_nodes);
    
    std::vector<double> llh = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);
    
    double expected_llh = -6.513306124309698;
    REQUIRE(abs(llh[0] - expected_llh) < 1e-6);
}

TEST_CASE("INSIDE_OUTSIDE_TEST_LLH_11", "[insidellh]") {
    auto [t, model] = build_llh_unit_test({{-1}, {0}, {0}}, 0.1, 0.0, 1.0);

    std::vector<double> buffer(model.alphabet_sizes[0]);
    std::vector<laml_data> model_data = model.initialize_data(t.tree, t.branch_lengths, &buffer);
    
    size_t num_characters = model.alphabet_sizes.size();
    size_t max_alphabet_size = model.alphabet_sizes[0];
    likelihood_buffer inside_ll(num_characters, max_alphabet_size, t.num_nodes);
    
    std::vector<double> llh = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);
    
    double expected_llh = -6.513306124309698;
    REQUIRE(abs(llh[0] - expected_llh) < 1e-6);
}

TEST_CASE("INSIDE_OUTSIDE_TEST_LLH_12", "[insidellh]") {
    auto [t, model] = build_llh_unit_test({{0}, {1}, {-1}}, 0.1, 0.0, 1.0);

    std::vector<double> buffer(model.alphabet_sizes[0]);
    std::vector<laml_data> model_data = model.initialize_data(t.tree, t.branch_lengths, &buffer);
    
    size_t num_characters = model.alphabet_sizes.size();
    size_t max_alphabet_size = model.alphabet_sizes[0];
    likelihood_buffer inside_ll(num_characters, max_alphabet_size, t.num_nodes);
    
    std::vector<double> llh = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);
    
    double expected_llh = -5.97198126969678;
    REQUIRE(abs(llh[0] - expected_llh) < 1e-6);
}

TEST_CASE("INSIDE_OUTSIDE_TEST_LLH_13", "[insidellh]") {
    auto [t, model] = build_llh_unit_test({{0}, {-1}, {1}}, 0.1, 0.0, 1.0);

    std::vector<double> buffer(model.alphabet_sizes[0]);
    std::vector<laml_data> model_data = model.initialize_data(t.tree, t.branch_lengths, &buffer);
    
    size_t num_characters = model.alphabet_sizes.size();
    size_t max_alphabet_size = model.alphabet_sizes[0];
    likelihood_buffer inside_ll(num_characters, max_alphabet_size, t.num_nodes);
    
    std::vector<double> llh = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);
    
    double expected_llh = -5.97198126969678;
    REQUIRE(abs(llh[0] - expected_llh) < 1e-6);
}

TEST_CASE("INSIDE_OUTSIDE_TEST_LLH_14", "[insidellh]") {
    auto [t, model] = build_llh_unit_test({{-1}, {0}, {1}}, 0.1, 0.0, 1.0);

    std::vector<double> buffer(model.alphabet_sizes[0]);
    std::vector<laml_data> model_data = model.initialize_data(t.tree, t.branch_lengths, &buffer);
    
    size_t num_characters = model.alphabet_sizes.size();
    size_t max_alphabet_size = model.alphabet_sizes[0];
    likelihood_buffer inside_ll(num_characters, max_alphabet_size, t.num_nodes);
    
    std::vector<double> llh = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);
    
    double expected_llh = -5.97198126969678;
    REQUIRE(abs(llh[0] - expected_llh) < 1e-6);
}

TEST_CASE("INSIDE_OUTSIDE_TEST_LLH_15", "[insidellh]") {
    auto [t, model] = build_llh_unit_test({{1}, {-1}, {0}}, 0.1, 0.0, 1.0);

    std::vector<double> buffer(model.alphabet_sizes[0]);
    std::vector<laml_data> model_data = model.initialize_data(t.tree, t.branch_lengths, &buffer);
    
    size_t num_characters = model.alphabet_sizes.size();
    size_t max_alphabet_size = model.alphabet_sizes[0];
    likelihood_buffer inside_ll(num_characters, max_alphabet_size, t.num_nodes);
    
    std::vector<double> llh = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);
    
    double expected_llh = -4.658719582178557;
    REQUIRE(abs(llh[0] - expected_llh) < 1e-6);
}

TEST_CASE("INSIDE_OUTSIDE_TEST_LLH_16", "[insidellh]") {
    auto [t, model] = build_llh_unit_test({{-1}, {1}, {0}}, 0.1, 0.0, 1.0);

    std::vector<double> buffer(model.alphabet_sizes[0]);
    std::vector<laml_data> model_data = model.initialize_data(t.tree, t.branch_lengths, &buffer);
    
    size_t num_characters = model.alphabet_sizes.size();
    size_t max_alphabet_size = model.alphabet_sizes[0];
    likelihood_buffer inside_ll(num_characters, max_alphabet_size, t.num_nodes);
    
    std::vector<double> llh = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);
    
    double expected_llh = -4.658719582178557;
    REQUIRE(abs(llh[0] - expected_llh) < 1e-6);
}

TEST_CASE("INSIDE_OUTSIDE_TEST_LLH_17", "[insidellh]") {
    auto [t, model] = build_llh_unit_test({{1}, {1}, {-1}}, 0.1, 0.0, 1.0);

    std::vector<double> buffer(model.alphabet_sizes[0]);
    std::vector<laml_data> model_data = model.initialize_data(t.tree, t.branch_lengths, &buffer);
    
    size_t num_characters = model.alphabet_sizes.size();
    size_t max_alphabet_size = model.alphabet_sizes[0];
    likelihood_buffer inside_ll(num_characters, max_alphabet_size, t.num_nodes);
    
    std::vector<double> llh = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);
    
    double expected_llh = -2.5980566021648364;
    REQUIRE(abs(llh[0] - expected_llh) < 1e-6);
}

TEST_CASE("INSIDE_OUTSIDE_TEST_LLH_18", "[insidellh]") {
    auto [t, model] = build_llh_unit_test({{1}, {-1}, {1}}, 0.1, 0.0, 1.0);

    std::vector<double> buffer(model.alphabet_sizes[0]);
    std::vector<laml_data> model_data = model.initialize_data(t.tree, t.branch_lengths, &buffer);
    
    size_t num_characters = model.alphabet_sizes.size();
    size_t max_alphabet_size = model.alphabet_sizes[0];
    likelihood_buffer inside_ll(num_characters, max_alphabet_size, t.num_nodes);
    
    std::vector<double> llh = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);
    
    double expected_llh = -2.695795750497349;
    REQUIRE(abs(llh[0] - expected_llh) < 1e-6);
}

TEST_CASE("INSIDE_OUTSIDE_TEST_LLH_19", "[insidellh]") {
    auto [t, model] = build_llh_unit_test({{-1}, {1}, {1}}, 0.1, 0.0, 1.0);

    std::vector<double> buffer(model.alphabet_sizes[0]);
    std::vector<laml_data> model_data = model.initialize_data(t.tree, t.branch_lengths, &buffer);
    
    size_t num_characters = model.alphabet_sizes.size();
    size_t max_alphabet_size = model.alphabet_sizes[0];
    likelihood_buffer inside_ll(num_characters, max_alphabet_size, t.num_nodes);
    
    std::vector<double> llh = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);
    
    double expected_llh = -2.695795750497349;
    REQUIRE(abs(llh[0] - expected_llh) < 1e-6);
}

TEST_CASE("INSIDE_OUTSIDE_TEST_LLH_20", "[insidellh]") {
    auto [t, model] = build_llh_unit_test({{1}, {1}, {1}}, 0.0, 0.0, 0.5);

    std::vector<double> buffer(model.alphabet_sizes[0]);
    std::vector<laml_data> model_data = model.initialize_data(t.tree, t.branch_lengths, &buffer);
    
    size_t num_characters = model.alphabet_sizes.size();
    size_t max_alphabet_size = model.alphabet_sizes[0];
    likelihood_buffer inside_ll(num_characters, max_alphabet_size, t.num_nodes);
    
    std::vector<double> llh = phylogeny::compute_inside_log_likelihood(model, t, inside_ll, model_data);
    
    double expected_llh = -1.0297894223949402;
    REQUIRE(abs(llh[0] - expected_llh) < 1e-6);
}