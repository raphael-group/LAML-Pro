add_executable(fastlaml fastlaml.cxx io.cxx models/laml.cxx laml_em.cxx ultrametric.cxx)

# add libraries
target_link_libraries(fastlaml PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(fastlaml PRIVATE spdlog)
target_link_libraries(fastlaml PRIVATE csv)
target_link_libraries(fastlaml PRIVATE argparse)
target_link_libraries(fastlaml PRIVATE nanobench)
target_link_libraries(fastlaml PRIVATE gcem)
target_link_libraries(fastlaml PRIVATE osqp)
target_link_libraries(fastlaml PRIVATE nlopt)

# Get the current commit hash
execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Define it as a macro in the compiler
add_compile_definitions(GIT_COMMIT_HASH="${GIT_COMMIT_HASH}")

target_compile_definitions(fastlaml PRIVATE GIT_COMMIT_HASH="${GIT_COMMIT_HASH}")

