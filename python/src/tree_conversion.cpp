#include "tree_conversion.h"
#include <stdexcept>

tree dict_to_tree(const py::dict& tree_dict) {
    // Extract required fields
    if (!tree_dict.contains("num_leaves")) {
        throw std::invalid_argument("Tree dict must contain 'num_leaves'");
    }
    if (!tree_dict.contains("num_nodes")) {
        throw std::invalid_argument("Tree dict must contain 'num_nodes'");
    }
    if (!tree_dict.contains("root")) {
        throw std::invalid_argument("Tree dict must contain 'root'");
    }
    if (!tree_dict.contains("edges")) {
        throw std::invalid_argument("Tree dict must contain 'edges'");
    }
    if (!tree_dict.contains("branch_lengths")) {
        throw std::invalid_argument("Tree dict must contain 'branch_lengths'");
    }

    size_t num_leaves = tree_dict["num_leaves"].cast<size_t>();
    size_t num_nodes = tree_dict["num_nodes"].cast<size_t>();
    size_t root = tree_dict["root"].cast<size_t>();
    auto edges = tree_dict["edges"].cast<std::vector<std::pair<int, int>>>();
    auto branch_lengths = tree_dict["branch_lengths"].cast<std::vector<double>>();

    // Validate sizes
    if (branch_lengths.size() != num_nodes) {
        throw std::invalid_argument("branch_lengths size must equal num_nodes");
    }

    // Extract optional node_names
    std::vector<std::string> node_names(num_nodes, "");
    if (tree_dict.contains("node_names")) {
        node_names = tree_dict["node_names"].cast<std::vector<std::string>>();
        if (node_names.size() != num_nodes) {
            throw std::invalid_argument("node_names size must equal num_nodes");
        }
    }

    // Build the digraph
    digraph<size_t> g;

    // Add vertices - vertex id maps to node data (which is the same as vertex id)
    for (size_t i = 0; i < num_nodes; i++) {
        g.add_vertex(i);
    }

    // Add edges
    for (const auto& edge : edges) {
        int parent = edge.first;
        int child = edge.second;
        if (parent < 0 || parent >= (int)num_nodes || child < 0 || child >= (int)num_nodes) {
            throw std::invalid_argument("Edge contains invalid node index");
        }
        g.add_edge(parent, child);
    }

    // Validate root
    if (root >= num_nodes) {
        throw std::invalid_argument("Root index out of bounds");
    }
    if (g.in_degree(root) != 0) {
        throw std::invalid_argument("Root node must have no incoming edges");
    }

    return tree{num_leaves, num_nodes, root, g, branch_lengths, node_names};
}

py::dict tree_to_dict(const tree& t) {
    py::dict result;

    result["num_leaves"] = t.num_leaves;
    result["num_nodes"] = t.num_nodes;
    result["root"] = t.root_id;

    // Extract edges from the digraph
    std::vector<std::pair<int, int>> edges;
    for (int node_id : t.tree.nodes()) {
        for (int child_id : t.tree.successors(node_id)) {
            edges.emplace_back(node_id, child_id);
        }
    }
    result["edges"] = edges;

    result["branch_lengths"] = t.branch_lengths;
    result["node_names"] = t.node_names;

    return result;
}
