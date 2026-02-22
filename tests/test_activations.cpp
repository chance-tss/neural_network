#include "unit_test.hpp"
#include "../include/Activations.hpp"

TEST(SigmoidTest) {
    ASSERT_NEAR(nn::Activations::sigmoid(0.0), 0.5, 0.0001);
    ASSERT_NEAR(nn::Activations::sigmoid(100.0), 1.0, 0.0001);
    ASSERT_NEAR(nn::Activations::sigmoid(-100.0), 0.0, 0.0001);
}

TEST(ReluTest) {
    ASSERT_EQ(nn::Activations::relu(10.0), 10.0);
    ASSERT_EQ(nn::Activations::relu(-10.0), 0.0);
    ASSERT_EQ(nn::Activations::relu(0.0), 0.0);
}
