add_executable(
    fastlaml_tests 
    likelihood_tests.cxx 
    ../src/laml_em.cxx 
    ../src/ultrametric.cxx 
    ../src/models/laml.cxx
)

# add libraries
target_link_libraries(fastlaml_tests PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(fastlaml_tests PRIVATE spdlog)
target_link_libraries(fastlaml_tests PRIVATE csv)
target_link_libraries(fastlaml_tests PRIVATE argparse)
target_link_libraries(fastlaml_tests PRIVATE nanobench)
target_link_libraries(fastlaml_tests PRIVATE gcem)
target_link_libraries(fastlaml_tests PRIVATE Eigen3::Eigen)
target_link_libraries(fastlaml_tests PRIVATE osqp)
target_link_libraries(fastlaml_tests PRIVATE Catch2::Catch2WithMain)