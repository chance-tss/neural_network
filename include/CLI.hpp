#include <string>
#include <vector>

namespace analyzer {

class CLI {
public:
    CLI() = default;
    ~CLI() = default;

    struct Config {
        std::vector<int> layers;
        double learningRate = 0.01;
        int epochs = 10;
        int batchSize = 32;
        double validationSplit = 0.2;
        double lrDecay = 1.0; // 1.0 = no decay
        int decayStep = 10;
    };

    int run(int argc, char** argv);
    Config loadConfig(const std::string& path);

private:
    void printUsage();
    void trainModel(const std::string& datasetPath, const Config& config);
};

} // namespace analyzer
