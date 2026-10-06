add_library(benchmark_mnn SHARED ${CMAKE_CURRENT_LIST_DIR}/bridge.cc)
target_include_directories(benchmark_mnn PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(benchmark_mnn PRIVATE sherpa-mnn-c-api log)
target_link_options(benchmark_mnn PRIVATE -Wl,--exclude-libs,ALL -Wl,-z,max-page-size=16384)
