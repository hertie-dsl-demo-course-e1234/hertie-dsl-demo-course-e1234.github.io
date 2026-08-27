# Session 3 handout - gradient descent by hand

Fifteen minutes, pen only. Doing this once with a pen is worth an hour of debugging later.

## The data

Four observations of one feature (kept as small integers so the arithmetic stays doable):

| i | x | y |
|---|---|---|
| 1 | -1 | 1 |
| 2 |  0 | 2 |
| 3 |  1 | 4 |
| 4 |  2 | 5 |

Model: `y_hat = a + b*x`. Loss: `L(a, b) = (1/4) * sum_i (y_i - a - b*x_i)^2`.

## 1. Derive the gradient

Differentiate `L` with respect to each parameter. You should reach:

```
dL/da = -(2/4) * sum_i r_i
dL/db = -(2/4) * sum_i r_i * x_i        where r_i = y_i - a - b*x_i
```

Write out both derivations in full. The `x_i` in the second line is the whole reason a
feature's coefficient moves at all - be sure you see where it comes from.

## 2. Take three steps

Start at `a = 0`, `b = 0`, with learning rate `alpha = 0.1`.

| step | a | b | r = (r1, r2, r3, r4) | dL/da | dL/db | L(a, b) |
|------|---|---|----------------------|-------|-------|---------|
| 0 | 0 | 0 | (1, 2, 4, 5) | -6.0 | -6.5 | 11.50 |
| 1 | 0.60 | 0.65 | | | | |
| 2 | | | | | | |
| 3 | | | | | | |

Row 0 is worked for you; check it before continuing. Fill in rows 1-3.

## 3. Check against the closed form

The exact least-squares solution for this data is `a = 2.3`, `b = 1.4` (verify with the
formulas from session 2). How close are you after three steps? Roughly how many steps would
you need at `alpha = 0.1`?

## 4. Break it

Repeat step 1 with `alpha = 1.0`. What happens to `|a|` and `|b|` between steps 1 and 3?
Write one sentence explaining, in terms of the size of the step relative to the curvature of
the loss, why the iteration moves away from the minimum instead of towards it.

## 5. One question to bring to the lab

Suppose `x` had not been standardised - suppose it were measured in the tens of thousands.
What would happen to `dL/db` on the first step, and what would you have to do to `alpha` to
compensate? Why does that make a single learning rate for both parameters a problem?
