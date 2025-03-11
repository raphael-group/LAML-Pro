#include <cstdlib>
#include <vector>
#include "digraph.h"
#include "compact_tree.h"

#include <argparse/argparse.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/fmt/ostr.h>

#define FASTLAML_VERSION_MAJOR 1
#define FASTLAML_VERSION_MINOR 0

/* 
* @brief A struct to hold a rooted tree without attaching parameters.

* @param tree The tree as a directed graph, where the nodes are
*             numbered from 0 to 2N - 1 and the leaves are numbered 
*             from 0 to N - 1.
* @param branch_lengths Stores the branch length on the edge leading to each node.
* @param node_names Stores the name of each node.
*/
struct Tree {
    size_t num_leaves;
    size_t num_nodes;
    size_t root;   
    digraph<size_t> tree;
    std::vector<double> branch_lengths;
    std::vector<std::string> node_names;
};

/*
* @brief Parse a Newick tree file into a Tree struct.
* @param fname The path to the Newick tree file.
* @return Tree The parsed tree.
*/
Tree parse_newick_tree(std::string fname) {
    compact_tree tree(fname);

    size_t num_nodes = tree.get_num_nodes();
    size_t num_leaves = tree.get_num_leaves();
    std::vector<int> name_map(num_nodes, -1);

    digraph<size_t> g;
    std::vector<double> branch_lengths(num_nodes);
    std::vector<std::string> node_names(num_nodes);
    size_t leaf_idx = 0;
    size_t internal_idx = num_leaves;
    for (size_t i = 0; i < num_nodes; i++) {
        int j;
        if (tree.get_children(i).size() == 0) {
            j = leaf_idx++;
        } else {
            j = internal_idx++;
        }

        int id = g.add_vertex(j);
        branch_lengths[j] = tree.get_edge_length(i);
        node_names[j] = tree.get_label(i);
        name_map[i] = id;
    }

    for (size_t i = 0; i < num_nodes; i++) {
        for (auto child : tree.get_children(i)) {
            g.add_edge(name_map[i], name_map[child]);
        }
    }

    return {num_leaves, num_nodes, tree.get_root(), g, branch_lengths, node_names};
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
        .help("Path to the mutation priors file");

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

    Tree tree = parse_newick_tree(program.get<std::string>("--tree"));
    spdlog::info("Tree Summary:");
    spdlog::info("  Number of leaves: {}", tree.num_leaves);
    spdlog::info("  Total number of nodes: {}", tree.num_nodes);
    spdlog::info("  Root: {}", tree.root);

    spdlog::info("Node details:");
    for (size_t i = 0; i < tree.num_nodes; i++) {
        int j = tree.tree[i].data;
        std::ostringstream children;
        for (auto child : tree.tree.successors(i)) {
            children << tree.tree[child].data << " ";
        }
        
        spdlog::info("  Node {}: name='{}', branch_length={}, children=[{}]", 
                    j, 
                    tree.node_names[j].empty() ? "<unnamed>" : tree.node_names[j], 
                    tree.branch_lengths[j],
                    children.str());
    }

    return 0;
}
