# Technical Documentation - MyTorch

## 1. Project Overview

MyTorch is a custom neural network framework written in C++20 standard library (STL), designed specifically for analyzing Chess positions. It implements a fully functional dense neural network from scratch, without external machine learning dependencies like PyTorch or TensorFlow.

The primary use case is `my_torch_analyzer`, a CLI tool that trains a network to predict the state of a chess game (Nothing, Check, Checkmate) or the winning color based on a FEN (Forsyth-Edwards Notation) string.

## 2. System Architecture

The project is divided into two distinct logical layers:

### 2.1 Core Neural Library (src/nn)
This independent module handles the mathematical operations of the neural network.

*   **Network**: The high-level container that manages a sequence of layers. It orchestrates the forward and backward passes.
*   **Layer**: Represents a dense (fully connected) layer. It holds the weights, biases, and gradient accumulators. It performs the matrix multiplication `Y = Activation(WX + B)`.
*   **Activations**: A static utility class providing activation functions (Sigmoid, ReLU) and their derivatives.
*   **Loss**: Provides loss functions (CrossEntropy) to evaluate model performance and compute gradients.

### 2.2 Analyzer Application (src/analyzer)
This module implements the specific business logic for the chess analysis task.

*   **CLI**: The Command Line Interface entry point. It handles argument parsing, configuration loading, and drives the training/prediction workflows.
*   **FENParser**: A optimized parser that converts a FEN string into a normalized input vector of size 838.
*   **Dataset**: Handles the loading and parsing of CSV datasets into memory, including label mapping.

## 3. Implementation Details

### 3.1 Neural Network Logic

#### Forward Pass
The propagation is sequential. An input vector enters the first layer. The output of layer `N` becomes the input of layer `N+1`.
Equation: `Z = Weights * Input + Biases`
Output: `A = Activation(Z)`

#### Backward Pass (Backpropagation)
Learning is achieved via Reverse Mode Differentiation.
1.  **Loss Derivative**: We calculate the gradient of the loss function with respect to the network output.
2.  **Propagation**: Each layer calculates the gradient with respect to its inputs (to pass to the previous layer) and locally computes gradients with respect to its weights and biases.
3.  **Gradient Accumulation**: To support Mini-Batch training, gradients are not applied immediately. They are summed up in `grad_weights_sum` and `grad_biases_sum` structures within each layer.

#### Optimization
We use Stochastic Gradient Descent (SGD) with Mini-Batch support and Learning Rate Decay.
*   **Weight Update**: `NewWeight = OldWeight - (LearningRate * AccumulatedGradient / BatchSize)`
*   **LR Scheduler**: The learning rate is multiplied by an `lr_decay` factor every `decay_step` epochs to refine convergence in later stages.

### 3.2 Chess Input Encoding (FEN)

A raw FEN string is converted into a flat vector of 838 floating-point values to serve as network input.

**Structure:**
*   **Board (64 squares x 13 channels)**: 832 features.
    *   Each square uses One-Hot encoding for the piece type (Pawn, Knight, Bishop, Rook, Queen, King - White/Black) or Empty.
*   **GameState (6 features)**:
    *   1 feature for side to move (White=1, Black=0).
    *   4 features for castling rights (KQkq).
    *   1 feature for en-passant target (boolean).

## 4. Developer Guide

### 4.1 Build System
The project uses a standard Makefile.

*   **compile**: `make`
*   **clean**: `make clean`
*   **full clean**: `make fclean`
*   **execution**: `./my_torch_analyzer`

### 4.2 Configuration Format
Support is strictly `.ini` style key-value pairs.

```ini
layers=838,128,64,3     # Topology definition
learning_rate=0.01      # Initial LR
epochs=50               # Training duration
batch_size=32           # Gradient accumulation size
validation_ratio=0.2    # % of data kept for validation
lr_decay=0.9            # Factor applied to LR
decay_step=10           # Epoch interval for decay
```

### 4.3 Extending the Framework

#### Adding a New Activation Function
1.  Modify `include/Activations.hpp` to declare the function and its derivative.
2.  Implement the logic in `src/nn/Activations.cpp`.
3.  Add the new enum value to `ActivationType` in `Layer.hpp`.
4.  Update the switch cases in `Layer::forward` and `Layer::backward`.

#### modifying the Input Format
The `FENParser` class is isolated. You can replace the implementation of `fenToVector` to change how chess positions are represented without touching the neural network core.

## 5. Performance Considerations
*   **Memory**: The dataset is loaded entirely into RAM for speed. For massive datasets (>10GB), a streaming iterator approach would be required in `Dataset.cpp`.
*   **Math**: Operations are scalar loops. SIMD (Single Instruction, Multiple Data) optimizations or BLAS integration could significantly speed up matrix multiplications.

## 6. Testing
Tests are located in the `tests/` directory and use a custom minimalist unit-testing header `unit_test.hpp`.

*   **Run**: `make tests`
*   **Scope**: Tests cover component logic (Layer, Activation, Network) and integration (XOR convergence).
