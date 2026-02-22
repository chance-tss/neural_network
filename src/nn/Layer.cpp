#include "Layer.hpp"
#include "Utils.hpp"
#include "Activations.hpp"
#include <random>
#include <fstream>
#include <iostream>

namespace nn {

Layer::Layer(int inputSize, int outputSize, ActivationType type)
    : inputSize(inputSize), outputSize(outputSize), activationType(type) {

    weights.resize(outputSize, std::vector<double>(inputSize));
    biases.resize(outputSize);
    grad_weights_sum.resize(outputSize, std::vector<double>(inputSize, 0.0));
    grad_biases_sum.resize(outputSize, 0.0);

    double limit = sqrt(6.0 / (inputSize + outputSize));
    for (int i = 0; i < outputSize; ++i) {
        biases[i] = 0.1;
        for (int j = 0; j < inputSize; ++j) {
            weights[i][j] = Utils::randomWeight(-limit, limit);
        }
    }
}

std::vector<double> Layer::forward(const std::vector<double>& input) {
    last_input = input;
    last_pre_activation.resize(outputSize);
    std::vector<double> output(outputSize);

    // Calculer z = Wx + b
    for (int i = 0; i < outputSize; ++i) {
        double sum = biases[i];
        for (int j = 0; j < inputSize; ++j) {
            sum += weights[i][j] * input[j];
        }
        last_pre_activation[i] = sum;
    }

    if (activationType == ActivationType::SOFTMAX) {
        output = Activations::softmax(last_pre_activation);
    } else {
        for (int i = 0; i < outputSize; ++i) {
            output[i] = (activationType == ActivationType::RELU)
                        ? Activations::relu(last_pre_activation[i])
                        : Activations::sigmoid(last_pre_activation[i]);
        }
    }

    last_output = output;  // Stocker pour Softmax
    return output;
}

std::vector<double> Layer::backward(const std::vector<double>& grad_output, double learningRate) {
    std::vector<double> grad_input(inputSize, 0.0);
    std::vector<double> dZ(outputSize);

    if (activationType == ActivationType::SOFTMAX) {
        dZ = grad_output;
    } else {
        for (int i = 0; i < outputSize; ++i) {
            double deriv = (activationType == ActivationType::RELU)
                           ? Activations::reluDerivative(last_pre_activation[i])
                           : Activations::sigmoidDerivative(last_pre_activation[i]);
            dZ[i] = grad_output[i] * deriv;
        }
    }

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            grad_input[j] += weights[i][j] * dZ[i];
            weights[i][j] -= learningRate * dZ[i] * last_input[j];
        }
        biases[i] -= learningRate * dZ[i];
    }
    return grad_input;
}

std::vector<double> Layer::backward(const std::vector<double>& grad_output) {
    std::vector<double> grad_input(inputSize, 0.0);
    std::vector<double> dZ(outputSize);

    //  Gérer Softmax
    if (activationType == ActivationType::SOFTMAX) {
        dZ = grad_output;
    } else {
        for (int i = 0; i < outputSize; ++i) {
            double deriv = (activationType == ActivationType::RELU)
                           ? Activations::reluDerivative(last_pre_activation[i])
                           : Activations::sigmoidDerivative(last_pre_activation[i]);
            dZ[i] = grad_output[i] * deriv;
        }
    }

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            grad_input[j] += weights[i][j] * dZ[i];
        }
    }
    return grad_input;
}

void Layer::accumulateGradients(const std::vector<double>& grad_output) {
    std::vector<double> dZ(outputSize);


    if (activationType == ActivationType::SOFTMAX) {
        dZ = grad_output;
    } else {
        for (int i = 0; i < outputSize; ++i) {
            double deriv = (activationType == ActivationType::RELU)
                           ? Activations::reluDerivative(last_pre_activation[i])
                           : Activations::sigmoidDerivative(last_pre_activation[i]);
            dZ[i] = grad_output[i] * deriv;
        }
    }

    for (int i = 0; i < outputSize; ++i) {
        grad_biases_sum[i] += dZ[i];
        for (int j = 0; j < inputSize; ++j) {
            grad_weights_sum[i][j] += dZ[i] * last_input[j];
        }
    }
}

void Layer::updateWeights(double learningRate, int batchSize) {
    if (batchSize == 0) return;
    double scale = learningRate / batchSize;

    for (int i = 0; i < outputSize; ++i) {
        biases[i] -= grad_biases_sum[i] * scale;
        grad_biases_sum[i] = 0.0;
        for (int j = 0; j < inputSize; ++j) {
            weights[i][j] -= grad_weights_sum[i][j] * scale;
            grad_weights_sum[i][j] = 0.0;
        }
    }
}

void Layer::clearGradients() {
    for (int i = 0; i < outputSize; ++i) {
        grad_biases_sum[i] = 0.0;
        std::fill(grad_weights_sum[i].begin(), grad_weights_sum[i].end(), 0.0);
    }
}

void Layer::save(std::ofstream& file) const {
    file << inputSize << " " << outputSize << " " << (int)activationType << "\n";
    for(const auto& row : weights) {
        for(double w : row) file << w << " ";
        file << "\n";
    }
    for(double b : biases) file << b << " ";
    file << "\n";
}

void Layer::loadWeights(std::ifstream& file) {
    for(int i=0; i<outputSize; ++i) {
        for(int j=0; j<inputSize; ++j) {
            file >> weights[i][j];
        }
    }
    for(int i=0; i<outputSize; ++i) {
        file >> biases[i];
    }
}

} // namespace nn