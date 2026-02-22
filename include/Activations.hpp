#pragma once
#include <vector>

namespace nn {

    class Activations {
    public:
        static double sigmoid(double x);
        static double sigmoidDerivative(double x);
        static double relu(double x);
        static double reluDerivative(double x);

        static std::vector<double> softmax(const std::vector<double>& x);
        // Note: La dérivée de Softmax est gérée directement dans la loss
    };

} // namespace nn