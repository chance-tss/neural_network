#include "unit_test.hpp"
#include "../include/Dataset.hpp"
#include <fstream>
#include <cstdio>

TEST(DatasetRawAndCSVTest) {
    // 1. Test CSV format
    std::string csvParams = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1;Nothing\n";
    std::string csvFile = "test_csv.txt";
    std::ofstream f1(csvFile);
    f1 << csvParams;
    f1.close();
    
    auto data1 = analyzer::Dataset::load(csvFile);
    ASSERT_EQ(data1.size(), 1);
    ASSERT_EQ(data1[0].second[0], 1.0); // Nothing
    std::remove(csvFile.c_str());

    // 2. Test Raw format (Space separated)
    // FEN has spaces, so split must be on last space
    std::string rawParams = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 Nothing\n";
    std::string rawFile = "test_raw.txt";
    std::ofstream f2(rawFile);
    f2 << rawParams;
    f2.close();
    
    auto data2 = analyzer::Dataset::load(rawFile);
    ASSERT_EQ(data2.size(), 1);
    ASSERT_EQ(data2[0].second[0], 1.0); // Nothing
    std::remove(rawFile.c_str());
}
