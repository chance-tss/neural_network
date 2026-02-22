#pragma once
#include <vector>
#include <string>
#include <utility>

namespace analyzer {

class Dataset {
public:
    // Returns a pair of vectors: input (features) and target (label)
    static std::vector<std::pair<std::vector<double>, std::vector<double>>> load(const std::string& path);
};

} // namespace analyzer
