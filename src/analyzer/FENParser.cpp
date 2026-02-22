#include "FENParser.hpp"
#include <sstream>
#include <cctype>

namespace analyzer {

std::vector<double> FENParser::fenToVector(const std::string& fen) {
    std::vector<double> features;
    features.reserve(838);

    std::stringstream ss(fen);
    std::string board, activeColor, castling, enPassant;
    ss >> board >> activeColor >> castling >> enPassant;

    // 1. Board (832 features)
    for (char c : board) {
        if (c == '/') continue;
        
        if (isdigit(c)) {
            int emptyCount = c - '0';
            for (int i = 0; i < emptyCount; ++i) {
                // Empty square: index 0 is 1, others 0
                features.push_back(1.0); // Index 0
                for(int k=1; k<13; ++k) features.push_back(0.0);
            }
        } else {
            // Piece
            int idx = 0;
            switch(c) {
                case 'P': idx = 1; break;
                case 'N': idx = 2; break;
                case 'B': idx = 3; break;
                case 'R': idx = 4; break;
                case 'Q': idx = 5; break;
                case 'K': idx = 6; break;
                case 'p': idx = 7; break;
                case 'n': idx = 8; break;
                case 'b': idx = 9; break;
                case 'r': idx = 10; break;
                case 'q': idx = 11; break;
                case 'k': idx = 12; break;
            }
            
            for(int k=0; k<13; ++k) {
                features.push_back(k == idx ? 1.0 : 0.0);
            }
        }
    }

    // 2. Active Color (1 feature)
    features.push_back(activeColor == "w" ? 1.0 : 0.0);

    // 3. Castling (4 features)
    features.push_back(castling.find('K') != std::string::npos ? 1.0 : 0.0);
    features.push_back(castling.find('Q') != std::string::npos ? 1.0 : 0.0);
    features.push_back(castling.find('k') != std::string::npos ? 1.0 : 0.0);
    features.push_back(castling.find('q') != std::string::npos ? 1.0 : 0.0);

    // 4. En Passant (1 feature)
    features.push_back(enPassant != "-" ? 1.0 : 0.0);

    return features;
}

} // namespace analyzer
