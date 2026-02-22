// include/nn/Loss.hpp

#pragma once

#include <vector>
#include <cmath>
#include <numeric>

namespace nn::loss {

    using Vector = std::vector<double>;

    double meanSquaredError(const Vector& predicted, const Vector& expected);
    Vector meanSquaredErrorDerivative(const Vector& predicted, const Vector& expected);

    double crossEntropy(const Vector& predicted, const Vector& expected);
    Vector crossEntropyDerivative(const Vector& predicted, const Vector& expected);

} // namespace nn::loss