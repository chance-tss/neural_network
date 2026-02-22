#include "Loss.hpp"

namespace nn::loss {

    double meanSquaredError(const Vector& predicted, const Vector& expected)
    {
        double sum_squared_error = 0.0;

        for (size_t i = 0; i < predicted.size(); ++i) {
            double error = predicted[i] - expected[i];
            sum_squared_error += error * error;
        }

        return sum_squared_error / predicted.size();
    }

    Vector meanSquaredErrorDerivative(const Vector& predicted, const Vector& expected)
    {
        Vector derivative(predicted.size());

        for (size_t i = 0; i < predicted.size(); ++i) {
            derivative[i] = 2.0 * (predicted[i] - expected[i]) / predicted.size();
        }

        return derivative;
    }

    double crossEntropy(const Vector& predicted, const Vector& expected) {
        double sum = 0.0;
        double epsilon = 1e-9;

        for(size_t i = 0; i < predicted.size(); ++i) {
            // Clamp pour stabilité numérique
            double val = std::max(epsilon, std::min(1.0 - epsilon, predicted[i]));
            // Pour classification multi-classes : -Σ(y_i × log(p_i))
            sum -= expected[i] * std::log(val);
        }

        return sum;
    }

    // Dérivée simplifiée pour Softmax + Cross-Entropy
    Vector crossEntropyDerivative(const Vector& predicted, const Vector& expected) {
        Vector derivative(predicted.size());

        // Avec Softmax + Cross-Entropy, la dérivée se simplifie à :
        for(size_t i = 0; i < predicted.size(); ++i) {
            derivative[i] = predicted[i] - expected[i];
        }

        return derivative;
    }

} // namespace nn::loss