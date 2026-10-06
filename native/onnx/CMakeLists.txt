add_library(benchmark_sherpa SHARED ${CMAKE_CURRENT_LIST_DIR}/bridge.cc)
target_include_directories(benchmark_sherpa PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(benchmark_sherpa PRIVATE sherpa-onnx-c-api log)
target_link_options(benchmark_sherpa PRIVATE -Wl,--exclude-libs,ALL -Wl,-z,max-page-size=16384)
