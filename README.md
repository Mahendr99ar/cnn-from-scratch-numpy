# Trainable CNN from Scratch in NumPy 🔬

A LeNet-style convolutional neural network built **entirely in NumPy** — no deep learning frameworks. Every layer, gradient, and optimizer step is implemented manually including im2col-based convolutions and Adam.

---

## What's inside

| Component | Details |
|---|---|
| **Loss** | Numerically stable softmax, cross-entropy, accuracy |
| **Initialization** | He initialization for conv and linear layers |
| **Convolution** | im2col / col2im forward + full backward pass |
| **Pooling** | Max pooling forward + gradient scatter backward |
| **Activations** | ReLU forward + backward |
| **Linear layers** | Forward, grad w.r.t input / weights / bias |
| **Optimizer** | SGD + Adam (momentum, variance, bias correction) |
| **Architecture** | Full LeNet: conv blocks + classifier head |
| **Training** | Minibatch iteration, train/test split, full training loop |

**59 functions implemented** — from `argmax_rows` to `evaluate`.

---

## Architecture

```
Input Image (H × W × C)
     │
     ▼
Conv2D → ReLU → MaxPool2D   ← Conv Block 1
     │
     ▼
Conv2D → ReLU → MaxPool2D   ← Conv Block 2
     │
     ▼
Flatten
     │
     ▼
Linear → ReLU               ← Classifier
     │
     ▼
Linear → Softmax → Loss
```

---

## How to run

```bash
# No dependencies beyond NumPy
pip install numpy

python scaffold.py
```

---

## Key concepts implemented

- **im2col convolution** — reshapes input patches into a matrix for efficient matmul-based convolution
- **col2im** — reverses im2col for gradient propagation
- **Max pooling backward** — scatters gradients back to max positions only
- **Adam optimizer** — full implementation with bias-corrected moment estimates
- **He initialization** — proper weight scaling for ReLU networks

---

## Why NumPy only?

Implementing backprop through convolutions manually (especially the im2col trick) gives an understanding of what frameworks like PyTorch do internally. No black boxes.

---

## Project structure

```
├── model.py       # All 59 functions implemented from scratch
├── scaffold.py    # Dataset generation, training loop, evaluation
└── README.md
```

---

*Built on [Deep-ML](https://www.deep-ml.com) — Build a Trainable CNN from Scratch in NumPy project.*
