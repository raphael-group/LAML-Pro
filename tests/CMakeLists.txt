add_executable(
    fastlaml_tests 
    likelihood_tests.cxx 
    test_obs_llh.cxx
    #test_parse_matrix.cxx
    ../src/laml_em.cxx 
    ../src/ultrametric.cxx 
    ../src/models/laml.cxx
    ../src/io.cxx
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g -O1")
set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")

# add libraries
target_link_libraries(fastlaml_tests PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(fastlaml_tests PRIVATE spdlog)
target_link_libraries(fastlaml_tests PRIVATE csv)
target_link_libraries(fastlaml_tests PRIVATE argparse)
target_link_libraries(fastlaml_tests PRIVATE nanobench)
target_link_libraries(fastlaml_tests PRIVATE gcem)
target_link_libraries(fastlaml_tests PRIVATE osqp)
target_link_libraries(fastlaml_tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(fastlaml_tests PRIVATE nlopt)
