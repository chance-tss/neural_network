#pragma once
#include <string>
#include <vector>

namespace analyzer {

class FENParser {
public:
    static std::vector<double> fenToVector(const std::string& fen);
};

} // namespace analyzer
