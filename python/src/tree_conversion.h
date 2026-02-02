#ifndef TREE_CONVERSION_H
#define TREE_CONVERSION_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "phylogeny.h"

namespace py = pybind11;

/**
 * Convert a Python dict to the C++ tree struct.
 *
 * Expected dict format:
 * {
 *     "num_leaves": int,
 *     "num_nodes": int,
 *     "root": int,
 *     "edges": [(parent, child), ...],
 *     "branch_lengths": [float, ...],  # indexed by node
 *     "node_names": [str, ...]  # optional, indexed by node
 * }
 */
tree dict_to_tree(const py::dict& tree_dict);

/**
 * Convert a C++ tree struct to a Python dict.
 */
py::dict tree_to_dict(const tree& t);

#endif // TREE_CONVERSION_H
