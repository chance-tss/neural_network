#include "Utils.hpp"
#include <random>

namespace nn {

double Utils::randomWeight(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

} // namespace nn
