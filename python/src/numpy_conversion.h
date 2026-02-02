#ifndef NUMPY_CONVERSION_H
#define NUMPY_CONVERSION_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

/**
 * Convert a 2D numpy array (int32) to a 2D vector of ints.
 * Used for character matrices where -1 represents missing data.
 */
std::vector<std::vector<int>> numpy_to_character_matrix(
    py::array_t<int32_t> arr
);

/**
 * Convert a 3D numpy array (float64) to a 3D vector of doubles.
 * Used for observation matrices with shape (leaves, characters, states).
 */
std::vector<std::vector<std::vector<double>>> numpy_to_observation_matrix(
    py::array_t<double> arr
);

/**
 * Convert a 2D numpy array (float64) to a 2D vector of doubles.
 * Used for mutation priors with shape (characters, states).
 */
std::vector<std::vector<double>> numpy_to_mutation_priors(
    py::array_t<double> arr
);

/**
 * Convert a 3D vector of doubles to a 3D numpy array.
 * Used for posterior probabilities with shape (characters, nodes, states).
 */
py::array_t<double> posterior_to_numpy(
    const std::vector<std::vector<std::vector<double>>>& posterior
);

#endif // NUMPY_CONVERSION_H
