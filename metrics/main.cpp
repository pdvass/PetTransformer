#include <algorithm>
#include <array>
#include <exception>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

#include "metrics.hpp"

namespace {

struct Metric {
  std::string_view long_flag;
  std::string_view short_flag;
  std::string_view description;
  void (*run)();
};

// To add a metric: write run_<name>_metrics() in its own file, declare it in
// metrics.hpp, and add a row here.
constexpr std::array METRICS{
    Metric{"--wordpiece", "-wp", "WordPiece tokenizer (wordpiece_vocab.txt)",
           metrics::run_wordpiece_metrics},
};

void print_usage() {
  std::cout << "Usage: MetricsPetTransformer [flags]\n"
            << "Runs every metric when no flag is given.\n\n";
  for (const auto &metric : METRICS)
    std::cout << "  " << metric.long_flag << ", " << metric.short_flag << "\t"
              << metric.description << "\n";
  std::cout << "  --help, -h\tShow this message\n";
}

} // namespace

int main(int argc, char *argv[]) {
  const std::span args{argv + 1, static_cast<size_t>(argc - 1)};
  std::vector<const Metric *> selected{};

  for (const std::string_view arg : args) {
    if (arg == "--help" || arg == "-h") {
      print_usage();
      return 0;
    }
    const auto it{std::ranges::find_if(METRICS, [&](const Metric &m) {
      return arg == m.long_flag || arg == m.short_flag;
    })};
    if (it == METRICS.end()) {
      std::cerr << "Unknown flag: " << arg << "\n\n";
      print_usage();
      return 1;
    }
    const Metric *metric{&*it};
    if (std::ranges::find(selected, metric) == selected.end())
      selected.push_back(metric);
  }

  if (selected.empty())
    for (const auto &metric : METRICS)
      selected.push_back(&metric);

  try {
    for (const auto *metric : selected)
      metric->run();
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
