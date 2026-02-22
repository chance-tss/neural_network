#include "Network.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

namespace nn {

Network::Network() {}

void Network::addLayer(int inputSize, int outputSize, ActivationType type) {
    layers.emplace_back(inputSize, outputSize, type);
}

std::vector<double> Network::forward(const std::vector<double>& input) {
    std::vector<double> current = input;
    for (auto& layer : layers) {
        current = layer.forward(current);
    }
    return current;
}

void Network::backward(const std::vector<double>& outputGradient, double learningRate) {
    std::vector<double> currentGradient = outputGradient;

    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        currentGradient = it->backward(currentGradient, learningRate);
    }
}

void Network::backward(const std::vector<double>& outputGradient) {
    std::vector<double> currentGradient = outputGradient;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        currentGradient = it->backward(currentGradient);
    }
}

void Network::accumulateGradients(const std::vector<double>& outputGradient) {
    std::vector<double> currentGradient = outputGradient;

    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        it->accumulateGradients(currentGradient);
        currentGradient = it->backward(currentGradient); // Get grad_input for next layer
    }
}

void Network::updateWeights(double learningRate, int batchSize) {
    for (auto& layer : layers) {
        layer.updateWeights(learningRate, batchSize);
    }
}

void Network::save(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot save model to " << path << std::endl;
        return;
    }

    file << layers.size() << "\n";

    for (const auto& layer : layers) {
        layer.save(file);
    }

    std::cout << "Model saved to " << path << std::endl;
}

void Network::load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot load model " << path << std::endl;
        return;
    }

    layers.clear();
    size_t numLayers;
    file >> numLayers;

    for (size_t i = 0; i < numLayers; ++i) {
        int inSize, outSize, typeInt;
        file >> inSize >> outSize >> typeInt;

        addLayer(inSize, outSize, static_cast<ActivationType>(typeInt));

        layers.back().loadWeights(file);
    }
    std::cout << "Model loaded from " << path << std::endl;
}
} // namespace nn