#include "unit_test.hpp"
#include "../include/Network.hpp"

TEST(NetworkTest) {
    nn::Network net;
    net.addLayer(2, 3); // Input 2 -> Hidden 3
    net.addLayer(3, 1); // Hidden 3 -> Output 1
    
    std::vector<double> input = {0.5, -0.5};
    std::vector<double> output = net.forward(input);
    
    ASSERT_EQ(output.size(), 1);
    ASSERT_TRUE(output[0] >= 0.0);
    ASSERT_TRUE(output[0] <= 1.0);
}
