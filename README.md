# Custom ML Library

A small C++ machine-learning/autograd experiment inspired by micrograd-style automatic differentiation. The project implements scalar computation graph nodes, reverse-mode backpropagation, activation functions, neurons, and simple neural-network building blocks from scratch.

## What This Demonstrates

- Reverse-mode automatic differentiation in C++.
- Manual construction of computation graphs through shared pointer relationships.
- Operator-style primitives for addition, multiplication, subtraction, division, powers, ReLU, and sigmoid.
- Backpropagation through a topologically sorted graph.
- Neural-network abstractions built on top of scalar `Value` nodes.

## Main File

```text
grad.cpp
```

The core `Value` class stores:

- `data`: scalar value
- `grad`: accumulated gradient
- `prev`: parent nodes in the computation graph
- `backward`: local gradient propagation function

## Build

```bash
g++ -std=c++17 -O2 grad.cpp -o grad
./grad
```

## Technical Focus

This project is not intended to compete with mature frameworks. It is a learning-oriented implementation of the core ideas behind neural network libraries: computation graphs, gradient flow, activation functions, and simple model composition.

## Possible Extensions

- Add a proper tensor abstraction.
- Add tests comparing gradients against finite differences.
- Implement loss functions and an optimizer interface.
- Add serialization for model weights.
