"""Session 8 demo - a two-layer network in pure Python, solving XOR.

XOR is the smallest problem no linear model can solve: no single line separates
{(0,0),(1,1)} from {(0,1),(1,0)}. A 2-2-1 network with a non-linearity solves it, and the
whole of backpropagation fits in `backward()` below.

Watch the loss sit near 0.69 (= log 2, the chance-level plateau) and then break through.

Run: python3 demo.py
"""

import math
import random

random.seed(2026)

XOR = [([0.0, 0.0], 0.0), ([0.0, 1.0], 1.0), ([1.0, 0.0], 1.0), ([1.0, 1.0], 0.0)]
HIDDEN = 4  # two is enough IN PRINCIPLE; four is enough IN PRACTICE - see the closing note
ALPHA = 0.5
N_ITER = 4000


def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def init():
    """Small random weights. All-zero weights make every hidden unit identical forever -
    symmetry that gradient descent can never break."""
    scale = 1.0
    W1 = [[random.gauss(0, scale) for _ in range(2)] for _ in range(HIDDEN)]
    b1 = [0.0] * HIDDEN
    W2 = [random.gauss(0, scale) for _ in range(HIDDEN)]
    b2 = 0.0
    return W1, b1, W2, b2


def forward(x, params):
    """Store every intermediate: the backward pass needs all of them."""
    W1, b1, W2, b2 = params
    z1 = [sum(w * xi for w, xi in zip(row, x)) + b for row, b in zip(W1, b1)]
    h = [max(0.0, z) for z in z1]                    # ReLU
    z2 = sum(w * hi for w, hi in zip(W2, h)) + b2
    return z1, h, z2, sigmoid(z2)


def backward(x, y, params, cache):
    """The chain rule, organised. Returns gradients for every parameter."""
    W1, b1, W2, b2 = params
    z1, h, _z2, y_hat = cache

    d2 = y_hat - y                                   # dL/dz2 for log-loss + sigmoid
    dW2 = [d2 * hi for hi in h]
    db2 = d2

    # Redistribute the output error to the hidden units, killed where ReLU was flat.
    d1 = [d2 * w * (1.0 if z > 0 else 0.0) for w, z in zip(W2, z1)]
    dW1 = [[d * xi for xi in x] for d in d1]
    db1 = list(d1)
    return dW1, db1, dW2, db2


def log_loss(params):
    eps = 1e-12
    total = 0.0
    for x, y in XOR:
        p = min(max(forward(x, params)[3], eps), 1 - eps)
        total -= y * math.log(p) + (1 - y) * math.log(1 - p)
    return total / len(XOR)


def train():
    params = init()
    for step in range(1, N_ITER + 1):
        W1, b1, W2, b2 = params
        gW1 = [[0.0] * 2 for _ in range(HIDDEN)]
        gb1 = [0.0] * HIDDEN
        gW2 = [0.0] * HIDDEN
        gb2 = 0.0
        for x, y in XOR:                             # full-batch: only four rows
            cache = forward(x, params)
            dW1, db1, dW2, db2 = backward(x, y, params, cache)
            for i in range(HIDDEN):
                for j in range(2):
                    gW1[i][j] += dW1[i][j] / len(XOR)
                gb1[i] += db1[i] / len(XOR)
                gW2[i] += dW2[i] / len(XOR)
            gb2 += db2 / len(XOR)

        W1 = [[w - ALPHA * g for w, g in zip(row, grow)] for row, grow in zip(W1, gW1)]
        b1 = [b - ALPHA * g for b, g in zip(b1, gb1)]
        W2 = [w - ALPHA * g for w, g in zip(W2, gW2)]
        b2 = b2 - ALPHA * gb2
        params = (W1, b1, W2, b2)

        if step % 500 == 0 or step == 1:
            print(f"  iter {step:5d}   log-loss = {log_loss(params):.4f}")
    return params


def main():
    print("training a 2-2-1 ReLU network on XOR (chance-level loss is log 2 = 0.693):")
    params = train()

    print("\nlearnt function:")
    for x, y in XOR:
        p = forward(x, params)[3]
        print(f"  x = {x}  ->  p = {p:.4f}   (target {y:.0f})   {'OK' if (p >= 0.5) == (y == 1) else 'WRONG'}")

    print("\nThe correctness check to run FIRST, always: can the network drive the loss on a")
    print("handful of rows to near zero? If not, the bug is in your code, not your model.")
    print("\nNow set HIDDEN = 2 and try seeds 1 to 5. Two hidden units are enough to")
    print("REPRESENT XOR, but most initialisations stall at 0.693 - a ReLU unit that starts")
    print("with negative pre-activations everywhere has zero gradient forever and is dead.")
    print("Extra width does not add expressiveness here; it adds paths to a good basin.")
    print("That is non-convexity, not a bug, and it is why you fix and report the seed.")


if __name__ == "__main__":
    main()
