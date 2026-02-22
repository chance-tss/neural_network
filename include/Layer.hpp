#pragma once
#include <vector>
#include <string>
#include <iostream>

namespace nn {

enum class ActivationType {
    SIGMOID,
    RELU,
    SOFTMAX
};

class Layer {
public:
    Layer(int inputSize, int outputSize, ActivationType activationType = ActivationType::SIGMOID);
    ~Layer() = default;

    std::vector<double> forward(const std::vector<double>& input);

    std::vector<double> backward(const std::vector<double>& grad_output, double learningRate); // Legacy compatible
    std::vector<double> backward(const std::vector<double>& grad_output); // Just gradients
    void accumulateGradients(const std::vector<double>& grad_output);
    void updateWeights(double learningRate, int batchSize);
    void clearGradients();

    void save(std::ofstream& file) const;
    void loadWeights(std::ifstream& file);

    int getInputSize() const { return inputSize; }
    int getOutputSize() const { return outputSize; }

private:
    int inputSize;
    int outputSize;
    ActivationType activationType;

    std::vector<std::vector<double>> weights; // Matrice [output][input]
    std::vector<double> biases;               // Vecteur [output]

    std::vector<double> last_input;           // X
    std::vector<double> last_output;
    std::vector<double> last_pre_activation;
    std::vector<std::vector<double>> grad_weights_sum;
    std::vector<double> grad_biases_sum;  // Z = WX + B
};

} // namespace nn