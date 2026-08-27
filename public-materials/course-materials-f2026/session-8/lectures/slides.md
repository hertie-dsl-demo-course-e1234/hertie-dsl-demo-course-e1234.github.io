% Session 8 - Neural networks from first principles
% Foundations of Machine Learning (Demo), E1234
% 27 October 2026 (after the midterm)

# One idea, stacked

A neural network is logistic regression composed with itself. Nothing in this session is new
except the composition - and the bookkeeping that composition requires.

```
layer 1:  h = g(W1 x + b1)        # p inputs  -> k hidden units
layer 2:  y_hat = sigma(W2 h + b2)  # k hidden -> 1 output
```

Drop `g` (make it the identity) and the whole thing collapses to a single linear model. **The
non-linearity is the entire point**: without it, depth buys nothing.

# Activations

| Name | `g(z)` | Note |
|---|---|---|
| Sigmoid | `1/(1+e^-z)` | saturates; gradients vanish. Output layer only. |
| Tanh | `tanh(z)` | zero-centred, still saturates |
| **ReLU** | `max(0, z)` | the default: cheap, non-saturating for `z > 0` |
| Leaky ReLU | `max(0.01z, z)` | avoids permanently dead units |

Use ReLU in hidden layers, and match the output activation to the task: none for regression,
sigmoid for binary, softmax for multi-class.

# Why depth at all

A single hidden layer of sufficient width can approximate any continuous function on a
compact set (the universal approximation theorem). That result says nothing about *how wide*.
Depth is the practical answer: composed transformations reuse intermediate structure, so a
deep narrow network can express what a shallow one needs exponentially many units for.

For the tabular data in this course, gradient-boosted trees usually still win. Networks earn
their place where structure is known and exploitable - images, sequences, graphs.

# Backpropagation is the chain rule, organised

Forward pass: compute and **store** `z1, h, z2, y_hat`, and the loss.
Backward pass: propagate the derivative of the loss back through each operation.

For our two-layer network with log-loss and sigmoid output:

```
d2 = y_hat - y                       # output error (the clean form from session 4)
dW2 = d2 h' / n ;   db2 = mean(d2)
d1 = (W2' d2) * g'(z1)               # elementwise: chain rule through the activation
dW1 = d1 x' / n ;   db1 = mean(d1)
```

Then the session-3 update, on every parameter at once:
`W <- W - alpha * dW`.

Read the middle line as the mechanism: error at the output is redistributed to hidden units
in proportion to how much each contributed, and killed off wherever the activation was flat.
That is where "vanishing gradients" comes from.

# What makes training work

- **Initialisation.** All-zero weights make every hidden unit identical forever. Use small
  random values scaled by fan-in (He for ReLU, Glorot for tanh).
- **Standardised inputs.** Same argument as session 3, now compounded across layers.
- **Mini-batches** of 32-256, and Adam rather than plain descent.
- **A learning-rate schedule.** Decay when validation loss plateaus.
- **Regularisation:** weight decay (L2 from session 5), dropout, and early stopping. Early
  stopping is the cheapest and most effective of the three.

# The debugging checklist

1. Can the network **overfit 20 rows** to near-zero loss? If not, the bug is in the code, not
   the model. Do this first, every time.
2. Does the loss decrease at all? If not, the learning rate is wrong by an order of magnitude.
3. Do gradients have sane magnitudes? Print norms; `nan` means an exploding step or a `log(0)`.
4. Is the validation loss diverging from training? That is overfitting - a modelling problem,
   and the good kind of problem to have.

# Honest expectations

Networks need more data, more compute and more tuning than anything else in this course, and
they give up the coefficient-level interpretability of sessions 2 and 4. Bring one when you
have a reason - not because it is the fashionable answer.

# Today's demo and lab

- `demo.py`: a two-layer network in ~60 lines of pure Python - forward pass, backprop,
  1,000 steps on the XOR problem, which no linear model can solve. Watch the loss fall
  through the plateau at 0.69.
- `labs/08_neural-networks`: the same network in NumPy on a real binary target; overfit 20
  rows first as the correctness check, then add early stopping and compare with session 4's
  logistic regression on the same split.

# Assignment 2 is due 3 November, 23:59

The notebook must run top to bottom from a clean kernel. Restart and run all before you push.
