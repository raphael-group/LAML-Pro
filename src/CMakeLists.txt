add_executable(lamlpro fastlaml.cxx io.cxx models/laml.cxx laml_em.cxx ultrametric.cxx)

# find ipopt
find_package(PkgConfig)
pkg_check_modules(IPOPT REQUIRED IMPORTED_TARGET ipopt>=3)

# add libraries
target_link_libraries(lamlpro PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(lamlpro PRIVATE spdlog)
target_link_libraries(lamlpro PRIVATE csv)
target_link_libraries(lamlpro PRIVATE argparse)
target_link_libraries(lamlpro PRIVATE nanobench)
target_link_libraries(lamlpro PRIVATE gcem)
target_link_libraries(lamlpro PRIVATE osqp)
target_link_libraries(lamlpro PRIVATE PkgConfig::IPOPT)

# Get the current commit hash
execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Define it as a macro in the compiler
add_compile_definitions(GIT_COMMIT_HASH="${GIT_COMMIT_HASH}")

target_compile_definitions(lamlpro PRIVATE GIT_COMMIT_HASH="${GIT_COMMIT_HASH}")

