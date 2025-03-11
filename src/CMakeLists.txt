add_executable(fastlaml fastlaml.cxx)

# add libraries
target_link_libraries(fastlaml PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(fastlaml PRIVATE spdlog)
target_link_libraries(fastlaml PRIVATE csv)
target_link_libraries(fastlaml PRIVATE argparse)
target_link_libraries(fastlaml PRIVATE nanobench)