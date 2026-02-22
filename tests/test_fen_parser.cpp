#include "unit_test.hpp"
#include "../include/FENParser.hpp"

TEST(FENParserTest) {
    // Start position
    std::string startFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    std::vector<double> vec = analyzer::FENParser::fenToVector(startFen);
    
    ASSERT_EQ(vec.size(), 838);
    
    // Check active color (White = 1)
    // Index 832 (0-indexed) is Active Color
    ASSERT_EQ(vec[832], 1.0);
    
    // Check castling (KQkq = 1,1,1,1)
    ASSERT_EQ(vec[833], 1.0);
    ASSERT_EQ(vec[834], 1.0);
    ASSERT_EQ(vec[835], 1.0);
    ASSERT_EQ(vec[836], 1.0);
    
    // Check en passant (- = 0)
    ASSERT_EQ(vec[837], 0.0);
}

TEST(FENParserEmptyTest) {
    // Empty board (kings only for validity usually, but parser doesn't check validity)
    // 8/8/8/8/8/8/8/8 b - - 0 1
    std::string emptyFen = "8/8/8/8/8/8/8/8 b - - 0 1";
    std::vector<double> vec = analyzer::FENParser::fenToVector(emptyFen);
    
    ASSERT_EQ(vec.size(), 838);
    ASSERT_EQ(vec[832], 0.0); // Black
    ASSERT_EQ(vec[837], 0.0); // No en passant
    
    // Check first square (should be empty -> index 0 = 1)
    ASSERT_EQ(vec[0], 1.0);
    ASSERT_EQ(vec[1], 0.0);
}
