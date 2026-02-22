#include "CLI.hpp"

int main(int argc, char** argv) {
    analyzer::CLI cli;
    return cli.run(argc, argv);
}
