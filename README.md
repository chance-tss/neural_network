# MyTorch - Chess Neural Network Analyzer

![Language](https://img.shields.io/badge/language-C%2B%2B20-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)

**MyTorch** is a high-performance, dependency-free neural network framework written in C++20. It demonstrates the fundamentals of deep learning by implementing a dense neural network from scratch to analyze and predict Chess game states.

**MyTorch** est un framework de réseau de neurones haute performance, sans dépendances, écrit en C++20. Il démontre les fondamentaux du deep learning en implémentant un réseau dense *from scratch* pour analyser et prédire les états de jeu d'Échecs.

---

## Documentation

Detailed technical documentation regarding architecture, implementation details, and developer guides is available below:

*   **[English Documentation](TECHNICAL_DOCUMENTATION_EN.md)**
*   **[Documentation Française](TECHNICAL_DOCUMENTATION_FR.md)**
*   **[Benchmarks & Performance Analysis](BENCHMARKS.md)**

---

## Features / Fonctionnalités

*   **Zero Dependencies**: Pure STL C++20 implementation.
*   **Custom Neural Engine**: Dense Layers, Backpropagation, SGD Optimizer.
*   **Advanced Training**:
    *   Mini-Batch processing.
    *   Learning Rate Scheduling (Decay).
    *   Checkpointing (Saves best model automatically).
    *   Cross-Entropy Loss for Classification.
*   **Analysis Tools**:
    *   Real-time training metrics (CSV format).
    *   Confusion Matrix visualization.
    *   FEN (Forsyth-Edwards Notation) parsing engine.
*   **Network Generator**: Create untrained networks from configuration files.

## Build & Installation

### Prerequisites / Prérequis
*   G++ (supporting C++20)
*   Make
*   Python 3 (optional, for benchmark visualizations)

### Compilation
```bash
make
```

This will build both:
- `my_torch_analyzer` - Main training and prediction tool
- `my_torch_generator` - Network generator utility

To run the test suite / Pour lancer la suite de tests :
```bash
make tests
```

## Usage

### 1. Generate a Network / Générer un Réseau

Create a new untrained network from a configuration file:

```bash
./my_torch_generator <config_file> [seed]
```

**Example:**
```bash
./my_torch_generator config_sample.txt 42
```

This creates a network with randomly initialized weights saved to `models/my_torch_network_generated.nn`.

### 2. Training / Entraînement

Train a model using a dataset (CSV) and a configuration file.

```bash
./my_torch_analyzer train --dataset <dataset.csv> --config <config.txt>
```

**Dataset Format:**
```csv
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1;Nothing
...
```
Supported labels: `Nothing`, `Check`, `Checkmate`, `White`, `Black`, `Draw`.

**Config Example:**
```ini
layers=838,128,64,3     # Input -> Hidden 1 -> Hidden 2 -> Output
learning_rate=0.01
epochs=50
batch_size=32
validation_ratio=0.2
lr_decay=0.9
decay_step=10
```

### 3. Prediction / Prédiction

Load a trained model and predict the state of a specific FEN position.

```bash
./my_torch_analyzer predict --fen "<FEN_STRING>" --model <model.nn>
```

**Example:**
```bash
./my_torch_analyzer predict --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" --model models/my_torch_network_best.nn
```

### 4. Visualize Benchmarks / Visualiser les Benchmarks

Generate training performance visualizations:

```bash
# Extract metrics from training output
grep -E "^[0-9]+," training_output.log > training_metrics.csv

# Generate plots
python3 scripts/visualize_benchmarks.py training_metrics.csv
```

See [BENCHMARKS.md](BENCHMARKS.md) for detailed performance analysis.

## Project Structure / Structure du Projet

*   `src/nn/`: **Neural Network Core**. The math, layers, and network logic.
*   `src/analyzer/`: **Application Logic**. CLI, FEN parsing, Dataset handling.
*   `include/`: **Headers**.
*   `models/`: **Artifacts**. Saved neural network weights.
*   `tests/`: **Unit Tests**. validation of mathematical correctness.
*   `scripts/`: **Utilities**. Dataset preparation and visualization tools.
*   `benchmarks/`: **Performance Metrics**. Generated training visualizations.

## Authors

*   **Kenzo O'Bryan**
*   **Chance TOSSOU**
