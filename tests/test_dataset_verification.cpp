#include "unit_test.hpp"
#include "../include/Network.hpp"
#include "../include/Dataset.hpp"
#include "../include/Loss.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

TEST(DatasetVerification) {
    // Load a subset or a small test file
    // Ideally we use one of the checked files, e.g. check_10_pieces.txt logic
    // But we need it in CSV format. 
    // Let's create a small in-memory dataset or load a small file.
    // For verification, we want to know if the network CAN learn the "Chess" patterns.
    // Let's rely on the prepare_dataset script being run or create a small temporary one.
    
    // Create a mini chess dataset
    std::string filename = "verification_dataset.csv";
    std::ofstream file(filename);
    // Simple patterns mimicking chess features (simplified)
    // Nothing: Empty board-ish, Check: King under attack
    // Let's just test if it can overfit a small batch of our REAL data format.
    // We will assume 834 inputs (Chess board size + extras? No, usually 64*12=768 + metadata)
    // Config said 838 inputs.
    
    // We need 838 floats. 
    // Let's just generate random data and see if it learns it (Sanity check)
    // Or even better, try to learn a simple mapping 838 -> 3 outputs.
    
    int inputSize = 838;
    int outputSize = 3; // Nothing, Check, Checkmate
    int samples = 50;
    
    // We will check if loss decreases.
    
    nn::Network net;
    net.addLayer(inputSize, 64, nn::ActivationType::RELU);
    net.addLayer(64, outputSize, nn::ActivationType::SIGMOID);
    
    std::vector<std::pair<std::vector<double>, std::vector<double>>> dataset;
    
    // Generate synthetic data
    for(int i=0; i<samples; ++i) {
        std::vector<double> in(inputSize, 0.0);
        std::vector<double> out(outputSize, 0.0);
        
        in[i % inputSize] = 1.0; // One-hot-ish
        out[i % outputSize] = 1.0; // Class
        dataset.push_back({in, out});
    }
    
    double initialLoss = 0.0;
    for(const auto& pair : dataset) {
        auto output = net.forward(pair.first);
        initialLoss += nn::loss::crossEntropy(output, pair.second);
    }
    
    std::cout << "Initial Loss: " << initialLoss << std::endl;
    
    // Train
    double lr = 0.01;
    for(int epoch=0; epoch<50; ++epoch) {
        for(const auto& pair : dataset) {
            auto output = net.forward(pair.first);
            auto grad = nn::loss::crossEntropyDerivative(output, pair.second);
            net.accumulateGradients(grad);
            net.updateWeights(lr, 1);
        }
    }
    
    double finalLoss = 0.0;
    int correct = 0;
    for(const auto& pair : dataset) {
        auto output = net.forward(pair.first);
        finalLoss += nn::loss::crossEntropy(output, pair.second);
        
        // Check ArgMax
        int pred = 0; double maxV = -1.0;
        int truth = 0; double maxT = -1.0;
        for(int k=0; k<outputSize; ++k) {
            if(output[k] > maxV) { maxV = output[k]; pred = k; }
            if(pair.second[k] > maxT) { maxT = pair.second[k]; truth = k; }
        }
        if(pred == truth) correct++;
    }
    
    std::cout << "Final Loss: " << finalLoss << std::endl;
    std::cout << "Accuracy: " << correct << "/" << samples << std::endl;
    
    ASSERT_TRUE(finalLoss < initialLoss);
    // It should learn this simple pattern easily
    ASSERT_TRUE(correct > samples * 0.8); 
    
    std::remove(filename.c_str());
}
