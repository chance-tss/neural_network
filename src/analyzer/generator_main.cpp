#include "Network.hpp"
#include "CLI.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>

void printUsage() {
    std::cout << "Usage: my_torch_generator <config_file> [seed]" << std::endl;
    std::cout << "  config_file: Path to configuration file (.txt or .ini)" << std::endl;
    std::cout << "  seed: (Optional) Random seed for weight initialization" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  ./my_torch_generator config_sample.txt 42" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        printUsage();
        return (argc < 2) ? 84 : 0;
    }

    std::string configPath = argv[1];
    int seed = (argc >= 3) ? std::atoi(argv[2]) : 0;

    try {
        // Load configuration using CLI's existing loadConfig function
        analyzer::CLI cli;
        analyzer::CLI::Config config = cli.loadConfig(configPath);

        // Set random seed if provided
        if (seed > 0) {
            std::srand(seed);
            std::cout << "Using random seed: " << seed << std::endl;
        }

        // Create network based on configuration
        nn::Network net;
        
        if (config.layers.size() < 2) {
            std::cerr << "Error: Configuration must specify at least 2 layers (input and output)" << std::endl;
            return 84;
        }

        std::cout << "Creating network with topology: ";
        for (size_t i = 0; i < config.layers.size(); ++i) {
            std::cout << config.layers[i];
            if (i < config.layers.size() - 1) std::cout << " -> ";
        }
        std::cout << std::endl;

        // Add layers to the network
        for (size_t i = 0; i < config.layers.size() - 1; ++i) {
            // Use SIGMOID for output layer, RELU for hidden layers
            nn::ActivationType act = (i == config.layers.size() - 2) 
                ? nn::ActivationType::SIGMOID 
                : nn::ActivationType::RELU;
            
            net.addLayer(config.layers[i], config.layers[i+1], act);
            
            std::cout << "  Layer " << i + 1 << ": " 
                      << config.layers[i] << " -> " << config.layers[i+1]
                      << " (Activation: " << (act == nn::ActivationType::RELU ? "ReLU" : "Sigmoid") << ")"
                      << std::endl;
        }

        // Generate output filename based on config file name
        std::string outputPath = "my_torch_network.nn";
        
        // Create models directory if it doesn't exist
        system("mkdir -p models");

        // Save the initialized network
        net.save(outputPath);

        std::cout << std::endl;
        std::cout << "✓ Network successfully generated and saved to: " << outputPath << std::endl;
        std::cout << "  Total layers: " << config.layers.size() - 1 << std::endl;
        std::cout << "  Input size: " << config.layers[0] << std::endl;
        std::cout << "  Output size: " << config.layers[config.layers.size() - 1] << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 84;
    }
}
