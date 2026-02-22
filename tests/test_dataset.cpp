#include "unit_test.hpp"
#include "../include/Dataset.hpp"
#include <fstream>
#include <cstdio>

TEST(DatasetLoaderTest) {
    // Create a temporary CSV file
    std::string filename = "test_dataset.csv";
    std::ofstream file(filename);
    file << "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1;Nothing\n";
    file << "8/8/8/8/8/8/8/8 b - - 0 1;Check\n"; // Dummy FEN for check
    file << "8/8/8/8/8/8/8/8 w - - 0 1;Checkmate\n"; // Dummy FEN for checkmate
    file.close();

    auto data = analyzer::Dataset::load(filename);
    
    ASSERT_EQ(data.size(), 3);
    
    // Check first item (Nothing)
    ASSERT_EQ(data[0].second.size(), 3);
    ASSERT_EQ(data[0].second[0], 1.0);
    ASSERT_EQ(data[0].second[1], 0.0);
    ASSERT_EQ(data[0].second[2], 0.0);
    
    // Check second item (Check)
    ASSERT_EQ(data[1].second[1], 1.0);
    
    // Check third item (Checkmate)
    ASSERT_EQ(data[2].second[2], 1.0);

    // Cleanup
    std::remove(filename.c_str());
}
