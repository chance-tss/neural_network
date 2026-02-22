#include "unit_test.hpp"
#include "../include/Layer.hpp"
#include <vector>
#include <cmath>

TEST(LayerInitialization) {
    nn::Layer layer(2, 3);
    // We can't easily access private members to check weights directly without friend classes or getters.
    // For now, let's just ensure it constructs without crashing and forward returns correct size.
    // Ideally we would add getters for testing or make the test a friend.
    
    std::vector<double> input = {1.0, 2.0};
    std::vector<double> output = layer.forward(input);
    
    ASSERT_EQ(output.size(), 3);
    for (double val : output) {
        ASSERT_TRUE(val >= 0.0);
        ASSERT_TRUE(val <= 1.0);
    }
}
