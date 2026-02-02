#include "numpy_conversion.h"
#include <stdexcept>

std::vector<std::vector<int>> numpy_to_character_matrix(py::array_t<int32_t> arr) {
    auto buf = arr.request();

    if (buf.ndim != 2) {
        throw std::invalid_argument("Character matrix must be 2-dimensional");
    }

    size_t num_leaves = buf.shape[0];
    size_t num_chars = buf.shape[1];

    std::vector<std::vector<int>> result(num_leaves, std::vector<int>(num_chars));

    auto ptr = static_cast<int32_t*>(buf.ptr);
    for (size_t i = 0; i < num_leaves; i++) {
        for (size_t j = 0; j < num_chars; j++) {
            result[i][j] = ptr[i * num_chars + j];
        }
    }

    return result;
}

std::vector<std::vector<std::vector<double>>> numpy_to_observation_matrix(py::array_t<double> arr) {
    auto buf = arr.request();

    if (buf.ndim != 3) {
        throw std::invalid_argument("Observation matrix must be 3-dimensional (leaves, characters, states)");
    }

    size_t num_leaves = buf.shape[0];
    size_t num_chars = buf.shape[1];
    size_t num_states = buf.shape[2];

    std::vector<std::vector<std::vector<double>>> result(
        num_leaves,
        std::vector<std::vector<double>>(
            num_chars,
            std::vector<double>(num_states)
        )
    );

    auto ptr = static_cast<double*>(buf.ptr);
    for (size_t i = 0; i < num_leaves; i++) {
        for (size_t j = 0; j < num_chars; j++) {
            for (size_t k = 0; k < num_states; k++) {
                result[i][j][k] = ptr[i * num_chars * num_states + j * num_states + k];
            }
        }
    }

    return result;
}

std::vector<std::vector<double>> numpy_to_mutation_priors(py::array_t<double> arr) {
    auto buf = arr.request();

    if (buf.ndim != 2) {
        throw std::invalid_argument("Mutation priors must be 2-dimensional (characters, states)");
    }

    size_t num_chars = buf.shape[0];
    size_t num_states = buf.shape[1];

    std::vector<std::vector<double>> result(num_chars, std::vector<double>(num_states));

    auto ptr = static_cast<double*>(buf.ptr);
    for (size_t i = 0; i < num_chars; i++) {
        for (size_t j = 0; j < num_states; j++) {
            result[i][j] = ptr[i * num_states + j];
        }
    }

    return result;
}

py::array_t<double> posterior_to_numpy(
    const std::vector<std::vector<std::vector<double>>>& posterior
) {
    if (posterior.empty()) {
        return py::array_t<double>({0, 0, 0});
    }

    size_t num_chars = posterior.size();
    size_t num_nodes = posterior[0].size();
    size_t max_states = 0;

    // Find max number of states across all characters
    for (const auto& char_post : posterior) {
        for (const auto& node_post : char_post) {
            max_states = std::max(max_states, node_post.size());
        }
    }

    // Create numpy array
    py::array_t<double> result({num_chars, num_nodes, max_states});
    auto buf = result.request();
    auto ptr = static_cast<double*>(buf.ptr);

    // Fill with zeros first
    std::fill(ptr, ptr + num_chars * num_nodes * max_states, 0.0);

    // Copy data
    for (size_t c = 0; c < num_chars; c++) {
        for (size_t n = 0; n < num_nodes; n++) {
            for (size_t s = 0; s < posterior[c][n].size(); s++) {
                ptr[c * num_nodes * max_states + n * max_states + s] = posterior[c][n][s];
            }
        }
    }

    return result;
}
