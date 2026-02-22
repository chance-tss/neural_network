#include "CLI.hpp"
#include "Network.hpp"
#include "FENParser.hpp"
#include "Dataset.hpp"
#include "Loss.hpp"
#include "Utils.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

namespace analyzer {

int CLI::run(int argc, char** argv) {
    if (argc < 2) {
        printUsage();
        return 84;
    }

    std::string mode = argv[1];

    if (mode == "--help" || mode == "-h") {
        printUsage();
        return 0;
    }

    if (mode == "train") {
        std::string datasetPath;
        std::string configPath;

        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--dataset" && i + 1 < argc) {
                datasetPath = argv[++i];
            } else if (arg == "--config" && i + 1 < argc) {
                configPath = argv[++i];
            }
        }

        if (datasetPath.empty() || configPath.empty()) {
            std::cerr << "Error: Missing arguments for train mode." << std::endl;
            printUsage();
            return 84;
        }

        try {
            Config config = loadConfig(configPath);
            trainModel(datasetPath, config);
        } catch (const std::exception& e) {
            std::cerr << "Error during training: " << e.what() << std::endl;
            return 84;
        }

    } else if (mode == "predict") {
        std::string fen;
        std::string modelPath;

        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--fen" && i + 1 < argc) {
                fen = argv[++i];
            } else if (arg == "--model" && i + 1 < argc) {
                modelPath = argv[++i];
            }
        }

        if (fen.empty() || modelPath.empty()) {
            std::cerr << "Error: Missing arguments for predict mode." << std::endl;
            printUsage();
            return 84;
        }

        std::cout << "Predicting..." << std::endl;
        std::cout << "FEN: " << fen << std::endl;
        std::cout << "Model: " << modelPath << std::endl;
        
        nn::Network net;
        net.load(modelPath);
        
        std::vector<double> input = FENParser::fenToVector(fen);
        std::vector<double> output = net.forward(input);
        
        std::cout << "Output: [";
        for (size_t i = 0; i < output.size(); ++i) {
            std::cout << output[i] << (i < output.size() - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;
        
        int maxIdx = 0;
        for(size_t i=1; i<output.size(); ++i) {
            if(output[i] > output[maxIdx]) maxIdx = i;
        }
        
        std::string result;
        if(maxIdx == 0) result = "Nothing";
        else if(maxIdx == 1) result = "Check";
        else result = "Checkmate";
        
        std::cout << "Prediction: " << result << std::endl;
    } else {
        std::cerr << "Error: Unknown mode '" << mode << "'" << std::endl;
        printUsage();
        return 84;
    }
    return 0;
}

void CLI::printUsage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "  my_torch_analyzer train --dataset <path> --config <path>" << std::endl;
    std::cout << "  my_torch_analyzer predict --fen <fen> --model <path>" << std::endl;
}

CLI::Config CLI::loadConfig(const std::string& path) {
    Config config;
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + path);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string key, value;
        if (std::getline(ss, key, '=') && std::getline(ss, value)) {
            if (key == "learning_rate") config.learningRate = std::stod(value);
            else if (key == "epochs") config.epochs = std::stoi(value);
            else if (key == "batch_size") config.batchSize = std::stoi(value);
            else if (key == "validation_ratio") config.validationSplit = std::stod(value);
            else if (key == "lr_decay") config.lrDecay = std::stod(value);
            else if (key == "decay_step") config.decayStep = std::stoi(value);
            else if (key == "layers") {
                std::stringstream lss(value);
                std::string segment;
                while (std::getline(lss, segment, ',')) {
                    config.layers.push_back(std::stoi(segment));
                }
            }
        }
    }
    return config;
}

void CLI::trainModel(const std::string& datasetPath, const Config& config) {
    std::cout << "Loading dataset..." << std::endl;
    auto data = Dataset::load(datasetPath);
    if (data.empty()) {
        throw std::runtime_error("Dataset is empty or failed to load");
    }

    size_t valSize = static_cast<size_t>(data.size() * config.validationSplit);
    size_t trainSize = data.size() - valSize;
    
    std::cout << "Training on " << trainSize << " samples, validating on " << valSize << " samples." << std::endl;

    nn::Network net;
    for (size_t i = 0; i < config.layers.size() - 1; ++i) {
        nn::ActivationType act = (i == config.layers.size() - 2) ? nn::ActivationType::SIGMOID : nn::ActivationType::RELU;
        net.addLayer(config.layers[i], config.layers[i+1], act);
    }

    std::cout << "Starting training loop..." << std::endl;
    std::cout << "epoch,train_loss,val_loss,train_acc,val_acc" << std::endl;

    double currentLr = config.learningRate;

    double bestValAcc = 0.0; // Checkpointing

    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        if (epoch > 0 && epoch % config.decayStep == 0) {
            currentLr *= config.lrDecay;
            std::cout << "Adjusting learning rate to " << currentLr << std::endl;
        }
        double totalLoss = 0.0;
        int correct = 0;

        int batchCount = 0;
        for (size_t i = 0; i < trainSize; ++i) {
            auto& sample = data[i];
            
            auto output = net.forward(sample.first);
            totalLoss += nn::loss::crossEntropy(output, sample.second);
            
            int predIdx = 0, truthIdx = 0;
            for(size_t k=1; k<output.size(); ++k) if(output[k] > output[predIdx]) predIdx = k;
            for(size_t k=1; k<sample.second.size(); ++k) if(sample.second[k] > sample.second[truthIdx]) truthIdx = k;
            if(predIdx == truthIdx) correct++;

            auto grad = nn::loss::crossEntropyDerivative(output, sample.second);
            net.accumulateGradients(grad);

            batchCount++;
            if (batchCount >= config.batchSize || i == trainSize - 1) {
                net.updateWeights(currentLr, batchCount);
                batchCount = 0;
            }
        }

        double avgTrainLoss = totalLoss / trainSize;
        double trainAcc = (double)correct / trainSize;

        double valLoss = 0.0;
        int valCorrect = 0;
        for (size_t i = trainSize; i < data.size(); ++i) {
            auto& sample = data[i];
            auto output = net.forward(sample.first);
            valLoss += nn::loss::crossEntropy(output, sample.second);
            
            int predIdx = 0, truthIdx = 0;
            for(size_t k=1; k<output.size(); ++k) if(output[k] > output[predIdx]) predIdx = k;
            for(size_t k=1; k<sample.second.size(); ++k) if(sample.second[k] > sample.second[truthIdx]) truthIdx = k;
            if(predIdx == truthIdx) valCorrect++;
        }
        double avgValLoss = (valSize > 0) ? valLoss / valSize : 0.0;
        double valAcc = (valSize > 0) ? (double)valCorrect / valSize : 0.0;

        std::cout << epoch + 1 << "," << avgTrainLoss << "," << avgValLoss << "," << trainAcc << "," << valAcc << std::endl;

        // Checkpointing
        if (valAcc > bestValAcc) {
            bestValAcc = valAcc;
            net.save("my_torch_network.nn");
            // std::cout << "New best model saved!" << std::endl; // Optional spam
        }
    }
    
    net.save("my_torch_network_final.nn");

    // Confusion Matrix (on whole dataset or just validation? Usually validation, but let's do Validation for now)
    // 3 classes: 0=Nothing/White, 1=Check/Black, 2=Checkmate/Draw
    std::vector<std::vector<int>> confusion(3, std::vector<int>(3, 0));
    
    std::cout << "\nComputing Confusion Matrix on Validation Set..." << std::endl;
    for (size_t i = trainSize; i < data.size(); ++i) {
        auto& sample = data[i];
        auto output = net.forward(sample.first);
        
        int predIdx = 0;
        for(size_t k=1; k<output.size(); ++k) if(output[k] > output[predIdx]) predIdx = k;

        int truthIdx = 0;
        for(size_t k=1; k<sample.second.size(); ++k) if(sample.second[k] > sample.second[truthIdx]) truthIdx = k;
        
        if (predIdx < 3 && truthIdx < 3)
            confusion[truthIdx][predIdx]++;
    }

    std::cout << "       Pred: 0    1    2" << std::endl;
    for(int i=0; i<3; ++i) {
        std::cout << "True " << i << ":      ";
        for(int j=0; j<3; ++j) {
            std::cout << confusion[i][j];
            if (confusion[i][j] < 10) std::cout << "    ";
            else if (confusion[i][j] < 100) std::cout << "   ";
            else std::cout << "  ";
        }
        std::cout << std::endl;
    }
    std::cout << "Legend: 0=Nothing/White, 1=Check/Black, 2=Checkmate/Draw" << std::endl;
}

} // namespace analyzer
