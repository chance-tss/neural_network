#include "Activations.hpp"
#include <cmath>
#include <algorithm>

namespace nn {

double Activations::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double Activations::sigmoidDerivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

double Activations::relu(double x) {
    return std::max(0.0, x);
}

double Activations::reluDerivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

    std::vector<double> Activations::softmax(const std::vector<double>& x) {
    std::vector<double> result(x.size());

    // Trouver le max pour stabilité numérique (éviter overflow)
    double max_val = x[0];
    for (double val : x) {
        if (val > max_val) max_val = val;
    }

    // Calculer exp(x - max) et la somme
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::exp(x[i] - max_val);
        sum += result[i];
    }

    // Normaliser
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] /= sum;
    }

    return result;
}

} // namespace nn
