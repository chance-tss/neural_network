#include "Activations.hpp"
#include "Network.hpp"
#include "Loss.hpp"
#include <iostream>
#include <cmath>
#include <iomanip>

void test_softmax_basic() {
    std::cout << "=== Test 1: Softmax Basic ===\n";

    std::vector<double> input = {1.0, 2.0, 3.0};
    auto output = nn::Activations::softmax(input);

    std::cout << "Input:  [" << input[0] << ", " << input[1] << ", " << input[2] << "]\n";
    std::cout << "Output: [" << std::fixed << std::setprecision(4)
              << output[0] << ", " << output[1] << ", " << output[2] << "]\n";

    double sum = 0.0;
    for (double val : output) sum += val;
    std::cout << "Sum: " << sum << " (should be 1.0)\n";

    if (std::abs(sum - 1.0) < 1e-6) {
        std::cout << "✅ PASS: Sum equals 1.0\n\n";
    } else {
        std::cout << "❌ FAIL: Sum is not 1.0\n\n";
    }
}

void test_network_with_softmax() {
    std::cout << "=== Test 2: Network with Softmax ===\n";

    nn::Network net;
    net.addLayer(64, 32, nn::ActivationType::RELU);
    net.addLayer(32, 16, nn::ActivationType::RELU);
    net.addLayer(16, 3, nn::ActivationType::SOFTMAX);  // 3 classes

    std::vector<double> input(64, 0.5);
    auto output = net.forward(input);

    std::cout << "Output: [" << std::fixed << std::setprecision(4)
              << output[0] << ", " << output[1] << ", " << output[2] << "]\n";

    double sum = output[0] + output[1] + output[2];
    std::cout << "Sum: " << sum << " (should be ~1.0)\n";

    if (std::abs(sum - 1.0) < 1e-6) {
        std::cout << " PASS: Network output sums to 1.0\n\n";
    } else {
        std::cout << " FAIL: Network output doesn't sum to 1.0\n\n";
    }
}

void test_backward_with_softmax() {
    std::cout << "=== Test 3: Backward with Softmax + Cross-Entropy ===\n";

    nn::Network net;
    net.addLayer(4, 8, nn::ActivationType::RELU);
    net.addLayer(8, 3, nn::ActivationType::SOFTMAX);

    std::vector<double> input = {0.5, 0.3, 0.8, 0.2};
    std::vector<double> target = {0, 1, 0};  // Class 1 (Check)

    // Forward
    auto output = net.forward(input);

    // Calculate loss
    double loss_before = nn::loss::crossEntropy(output, target);
    std::cout << "Loss before: " << loss_before << "\n";

    // Backward
    auto grad = nn::loss::crossEntropyDerivative(output, target);
    net.backward(grad, 0.1);  // learning rate = 0.1

    // Forward again
    output = net.forward(input);
    double loss_after = nn::loss::crossEntropy(output, target);
    std::cout << "Loss after:  " << loss_after << "\n";

    if (loss_after < loss_before) {
        std::cout << " Loss decreased after training\n\n";
    } else {
        std::cout << "Loss didn't decrease (might need more iterations)\n\n";
    }
}
