#include "numpy_conversion.h"
#include "strided_read.h"
#include <stdexcept>
#include <string>

// Warn on non-C-contiguous input. The stride-based converters read it
// correctly; the warning just surfaces an unusual layout.
static void warn_if_noncontiguous(const py::buffer_info& buf, const char* name) {
    ssize_t expected = static_cast<ssize_t>(buf.itemsize);
    for (ssize_t d = buf.ndim - 1; d >= 0; --d) {
        if (buf.strides[d] != expected) {
            py::module_::import("warnings").attr("warn")(
                std::string(name) + " is not C-contiguous; reading it via its "
                "strides (the result is correct). Pass np.ascontiguousarray(...) "
                "to silence this warning.",
                py::module_::import("pylaml").attr("LayoutWarning"));
            return;
        }
        expected *= buf.shape[d];
    }
}

std::vector<std::vector<int>> numpy_to_character_matrix(py::array_t<int32_t> arr) {
    auto buf = arr.request();

    if (buf.ndim != 2) {
        throw std::invalid_argument("Character matrix must be 2-dimensional");
    }

    warn_if_noncontiguous(buf, "Character matrix");

    // Read via the array's actual strides (see strided_read.h) so any memory
    // layout -- transposed view, slice, Fortran-ordered `.values` -- is handled
    // correctly rather than silently scrambled.
    const ssize_t s0 = buf.strides[0] / static_cast<ssize_t>(sizeof(int32_t));
    const ssize_t s1 = buf.strides[1] / static_cast<ssize_t>(sizeof(int32_t));
    return read_strided_2d(static_cast<int32_t*>(buf.ptr), buf.shape[0], buf.shape[1], s0, s1);
}

std::vector<std::vector<std::vector<double>>> numpy_to_observation_matrix(py::array_t<double> arr) {
    auto buf = arr.request();

    if (buf.ndim != 3) {
        throw std::invalid_argument("Observation matrix must be 3-dimensional (leaves, characters, states)");
    }

    warn_if_noncontiguous(buf, "Observation matrix");

    const ssize_t s0 = buf.strides[0] / static_cast<ssize_t>(sizeof(double));
    const ssize_t s1 = buf.strides[1] / static_cast<ssize_t>(sizeof(double));
    const ssize_t s2 = buf.strides[2] / static_cast<ssize_t>(sizeof(double));
    return read_strided_3d(static_cast<double*>(buf.ptr),
                           buf.shape[0], buf.shape[1], buf.shape[2], s0, s1, s2);
}

std::vector<std::vector<double>> numpy_to_mutation_priors(py::array_t<double> arr) {
    auto buf = arr.request();

    if (buf.ndim != 2) {
        throw std::invalid_argument("Mutation priors must be 2-dimensional (characters, states)");
    }

    warn_if_noncontiguous(buf, "Mutation priors");

    const ssize_t s0 = buf.strides[0] / static_cast<ssize_t>(sizeof(double));
    const ssize_t s1 = buf.strides[1] / static_cast<ssize_t>(sizeof(double));
    return read_strided_2d(static_cast<double*>(buf.ptr), buf.shape[0], buf.shape[1], s0, s1);
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
