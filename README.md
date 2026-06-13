# Trainable CNN from Scratch in NumPy 🔬

I built this because I was tired of using CNNs as a black box. Every tutorial just says "call Conv2D" — nobody shows what's actually happening inside. So I implemented the whole thing in pure NumPy, no PyTorch, no autograd, nothing.

The hardest part was getting im2col and its backward pass (col2im) right. Once that clicked, the rest started making sense.

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

59 functions total — from `argmax_rows` to `evaluate`.

---

## Architecture
---
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
## How to run

```bash
pip install numpy
python scaffold.py
```

---

## Things I learned building this

- **im2col** is why convolutions are actually fast — instead of looping over every patch, you reshape the input into one big matrix and do a single matmul
- **col2im backward** was the trickiest part — gradients from overlapping patches need to be accumulated carefully, easy to get wrong
- **Max pooling backward** only sends gradients to the winning position, zeros everywhere else — simple idea but satisfying to implement
- **He initialization** made a noticeable difference — wrong init and the loss barely moves in early epochs
- Writing Adam from scratch (moments, bias correction, param update) makes you appreciate how much work goes into "just use Adam"

---

## Why NumPy only?

Frameworks hide a lot. Writing the conv backward pass by hand, deriving the gradient flow through im2col — that's the kind of thing that actually sticks. Now when something breaks in PyTorch I have a much better idea of where to look.

---

## Project structure
---
├── model.py       # All 59 functions implemented from scratch
├── scaffold.py    # Dataset generation, training loop, evaluation
└── README.md

*Built on [Deep-ML](https://www.deep-ml.com)*
