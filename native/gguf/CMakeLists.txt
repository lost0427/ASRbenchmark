add_library(benchmark_gguf SHARED ${CMAKE_CURRENT_LIST_DIR}/bridge.cc)
target_include_directories(benchmark_gguf PRIVATE ${CMAKE_SOURCE_DIR}/sense-voice/csrc)
target_link_libraries(benchmark_gguf PRIVATE sense-voice-core ggml log)
target_link_options(benchmark_gguf PRIVATE -Wl,--exclude-libs,ALL -Wl,-z,max-page-size=16384)
