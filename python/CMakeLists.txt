# Python bindings using pybind11

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Find Python
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Find IPOPT (needed for this subdirectory)
find_package(PkgConfig REQUIRED)
pkg_check_modules(IPOPT REQUIRED IMPORTED_TARGET ipopt>=3)

# Create the Python module
pybind11_add_module(_core
    src/bindings.cpp
    src/tree_conversion.cpp
    src/numpy_conversion.cpp
    ${CMAKE_SOURCE_DIR}/src/laml_em.cxx
    ${CMAKE_SOURCE_DIR}/src/ultrametric.cxx
    ${CMAKE_SOURCE_DIR}/src/models/laml.cxx
)

# Include directories
target_include_directories(_core PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third-party/csv-parser/include
    ${CMAKE_SOURCE_DIR}/third-party/gcem/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link libraries - use static osqp to avoid rpath issues
target_link_libraries(_core PRIVATE
    spdlog
    osqpstatic
    PkgConfig::IPOPT
)

# Install the module
install(TARGETS _core LIBRARY DESTINATION pylaml)
