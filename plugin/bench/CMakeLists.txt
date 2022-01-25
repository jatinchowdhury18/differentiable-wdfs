list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/modules/cmake)
include(CPM)

# download google benchmark
CPMAddPackage(
  NAME benchmark
  GITHUB_REPOSITORY google/benchmark
  VERSION 1.5.2
  OPTIONS "BENCHMARK_ENABLE_TESTING Off"
)

juce_add_console_app(DiodeClipperBench)

juce_generate_juce_header(DiodeClipperBench)
add_custom_command(TARGET DiodeClipperBench
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E echo "copying $<TARGET_FILE:DiodeClipperBench> to ${PROJECT_BINARY_DIR}/DiodeClipperBench"
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:DiodeClipperBench> ${PROJECT_BINARY_DIR}/DiodeClipperBench)

target_sources(DiodeClipperBench PRIVATE diode_clipper_bench.cpp)

target_include_directories(DiodeClipperBench
    PRIVATE
      ../
)

target_link_libraries(DiodeClipperBench PUBLIC
    DifferentiableWDFs
    benchmark
)

set_target_properties(DiodeClipperBench PROPERTIES CXX_VISIBILITY_PRESET hidden)
