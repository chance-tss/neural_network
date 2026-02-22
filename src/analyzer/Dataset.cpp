#include "Dataset.hpp"
#include "FENParser.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

namespace analyzer {

std::vector<std::pair<std::vector<double>, std::vector<double>>> Dataset::load(const std::string& path) {
    std::vector<std::pair<std::vector<double>, std::vector<double>>> data;
    std::ifstream file(path);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open dataset file " << path << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::string fen, label;
        
        // Try finding semicolon first
        size_t sepPos = line.find(';');
        if (sepPos == std::string::npos) {
            // If no semicolon, find LAST space (because FEN can contain spaces)
            sepPos = line.find_last_of(' ');
        }

        if (sepPos == std::string::npos) {
            continue; // Invalid format
        }

        fen = line.substr(0, sepPos);
        label = line.substr(sepPos + 1);
        
        // Trim potential whitespace around label if needed (though find_last_of ' ' puts us right before it)
        // Ideally we trim the label just in case
        // But for now simple extraction:


        std::vector<double> input = FENParser::fenToVector(fen);
        std::vector<double> target;

        if (label == "Nothing") {
            target = {1.0, 0.0, 0.0};
        } else if (label == "Check") {
            target = {0.0, 1.0, 0.0};
        } else if (label == "Checkmate") {
            target = {0.0, 0.0, 1.0};
        } else if (label == "White" || label == "1-0") {
             target = {1.0, 0.0, 0.0};
        } else if (label == "Black" || label == "0-1") {
             target = {0.0, 1.0, 0.0};
        } else if (label == "Draw" || label == "1/2-1/2") {
             target = {0.0, 0.0, 1.0};
        } else {
            // Unknown label
            continue;
        }

        data.emplace_back(input, target);
    }

    return data;
}

} // namespace analyzer
