#pragma once
#include <vector>
#include <string>
#include "Layer.hpp"

namespace nn {

class Network {
public:
    Network();
    ~Network() = default;

    void addLayer(int inputSize, int outputSize, ActivationType type = ActivationType::SIGMOID);

    std::vector<double> forward(const std::vector<double>& input);

    void backward(const std::vector<double>& outputGradient, double learningRate); // Legacy
    void backward(const std::vector<double>& outputGradient); // Just gradients
    void accumulateGradients(const std::vector<double>& outputGradient);
    void updateWeights(double learningRate, int batchSize);

    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    std::vector<Layer> layers;
};

} // namespace nn