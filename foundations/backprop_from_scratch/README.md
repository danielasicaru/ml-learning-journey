# 01 — Backpropagation from Scratch

Implementation of a fully connected neural network using **only NumPy** — no frameworks — to deeply understand what happens under the hood during training.

## What's implemented

- Forward pass with sigmoid activation
- Binary cross-entropy loss
- Backpropagation using the chain rule
- Gradient descent weight updates

## Problem

The XOR classification problem — a classic non-linearly separable task that requires a hidden layer to solve.

## Network architecture

```
Input (2) → Hidden Layer (3 neurons, sigmoid) → Output (1 neuron, sigmoid)
```

## Key concepts covered

- Chain rule and gradient computation
- Vanishing gradients — why small learning rates stall training
- The difference between parameters (weights/biases) and hyperparameters (learning rate)
- Numerical stability tricks (epsilon in log to avoid log(0))

## How to run

```bash
pip install -r requirements.txt
python backprop.py
```

## Expected output

```
Epoch 0    | Loss: 0.6943
Epoch 1000 | Loss: 0.4321
...
Final predictions:
[[0.02]
 [0.97]
 [0.97]
 [0.03]]
Expected: [[0],[1],[1],[0]]
```