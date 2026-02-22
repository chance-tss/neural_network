#include "unit_test.hpp"
#include "../include/Network.hpp"
#include "../include/Loss.hpp"
#include "../include/Activations.hpp"
#include <iostream>

TEST(XORConvergence) {
    // XOR Problem
    // 0,0 -> 0
    // 0,1 -> 1
    // 1,0 -> 1
    // 1,1 -> 0
    std::vector<std::pair<std::vector<double>, std::vector<double>>> data = {
        {{0.0, 0.0}, {0.0}},
        {{0.0, 1.0}, {1.0}},
        {{1.0, 0.0}, {1.0}},
        {{1.0, 1.0}, {0.0}}
    };

    nn::Network net;
    net.addLayer(2, 4, nn::ActivationType::RELU);
    net.addLayer(4, 1, nn::ActivationType::SIGMOID);

    double learningRate = 0.1; 
    int epochs = 20000;

    for (int i = 0; i < epochs; ++i) {

        for (const auto& sample : data) {
            auto output = net.forward(sample.first);
            // Using MSE derivative for backward because simple Sigmoid+MSE works for XOR
            // But we implemented CrossEntropy derivative logic.
            // If output is Sigmoid, and we use MSE, deriv is 2(y-t).
            // If we use CrossEntropy logic:
            // dL/dy = -t/y + (1-t)/(1-y)
            // Let's use CrossEntropy logic since that's what we implemented.
            
            // However, our CrossEntropy assumes values in [0,1].
            
            auto grad = nn::loss::crossEntropyDerivative(output, sample.second);
            net.accumulateGradients(grad);
            net.updateWeights(learningRate, 1);
        }
    }

    // Verify
    bool correct = true;
    for (const auto& sample : data) {
        auto output = net.forward(sample.first);
        double val = output[0];
        double expected = sample.second[0];
        std::cout << "In: " << sample.first[0] << "," << sample.first[1] 
                  << " Out: " << val << " Exp: " << expected << std::endl;
        
        if (expected > 0.5) {
            if (val < 0.8) correct = false;
        } else {
            if (val > 0.2) correct = false;
        }
    }
    ASSERT_TRUE(correct);
}
