add_executable(MetricsPetTransformer main.cpp metrics.cpp wordpiece_metrics.cpp)

# tokenizer provides the public headers and C++20.
target_link_libraries(MetricsPetTransformer PRIVATE tokenizer)
