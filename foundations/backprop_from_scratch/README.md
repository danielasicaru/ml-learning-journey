# 01 — Backpropagation from Scratch

Implementation of a fully connected neural network using **only NumPy** — no frameworks — to deeply understand what happens under the hood during training. Built incrementally across 4 scripts, each adding a new layer of complexity.

---

## File Structure

```
backprop-from-scratch/
├── backprop_xor.py                  ← 2-layer sigmoid network, XOR problem
├── backprop_experiments.py          ← LR & weight init comparison experiments
├── backprop_relu_3layer.py          ← 3-layer Leaky ReLU network with gradient monitoring
├── backprop_validation_earlystop.py ← validation split + early stopping
├── README.md
└── requirements.txt
```

---

## Scripts

### `backprop_xor.py`
The foundation. A 2-layer fully connected network trained on the XOR problem using sigmoid activation and binary cross-entropy loss. Implements the full training pipeline from scratch: forward pass, loss computation, backpropagation via the chain rule, and gradient descent weight updates.

**Concepts:** Chain rule · Backpropagation · Binary cross-entropy · Gradient descent · Numerical stability (`1e-8` epsilon)

---

### `backprop_experiments.py`
Runs 4 training experiments in parallel and plots their loss curves on a single log-scale chart to visualize how learning rate and weight initialization interact.

| Experiment | Config | Behaviour |
|---|---|---|
| 1 | lr=0.1, ws=0.1 | Too slow — never converges |
| 2 | lr=1.1, ws=0.1 | Baseline — flat then cliff at ~epoch 2500 |
| 3 | lr=5.0, ws=0.1 | High LR — unstable but converges |
| 4 | lr=1.1, ws=0.5 | Better init — cliff disappears, learns immediately |

**Concepts:** Learning rate dynamics · Weight initialization · Loss landscape · Hyperparameter sensitivity

---

### `backprop_relu_3layer.py`
Extends the network to 3 hidden layers and swaps sigmoid activations for **Leaky ReLU** in hidden layers. Includes per-layer gradient magnitude monitoring to observe gradient flow across depth.

Key finding: vanilla ReLU caused the **Dying ReLU problem** — neurons receiving zero input (`[0,0]` in XOR) permanently died and the network failed silently at loss `0.4774`. Leaky ReLU fixed this by keeping a small gradient slope for negative values.

**Concepts:** ReLU vs Leaky ReLU · Dying ReLU problem · Gradient monitoring · Vanishing gradients across layers · Activation function choice

---

### `backprop_validation_earlystop.py`
Production-ready training loop with an 80/20 train/validation split, tracking both losses per epoch and stopping automatically when validation loss stops improving.

**Concepts:** Train/validation split · Overfitting detection · Early stopping · Patience counter · Generalisation vs memorisation

---

## Key Learnings

**Backpropagation** answers one question: *how much did each weight contribute to the error?* It uses the chain rule to propagate gradients backwards through every layer, so gradient descent knows exactly how to update each weight.

**Vanishing gradients** happen when gradients shrink as they travel backwards through layers — early layers learn far slower than later ones. Observable directly in the gradient monitoring output of `backprop_relu_3layer.py`.

**Activation functions matter architecturally** — sigmoid saturates and causes vanishing gradients in deep networks; ReLU is faster but can die; Leaky ReLU is the practical default for hidden layers.

**Hyperparameters interact** — learning rate and weight initialization are not independent. Poor initialization wastes thousands of epochs even with a good learning rate, as seen in experiment 2 vs experiment 4.

---

## How to Run

```bash
pip install -r requirements.txt

python backprop_xor.py
python backprop_experiments.py
python backprop_relu_3layer.py
python backprop_validation_earlystop.py
```

## Requirements

```
numpy
matplotlib
```
