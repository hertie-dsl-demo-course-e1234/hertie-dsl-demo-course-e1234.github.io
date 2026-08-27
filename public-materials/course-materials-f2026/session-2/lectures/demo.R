# Session 2 demo - lm() and the normal equations agree.
#
# The library and the closed form b = (X'X)^-1 X'y are the same computation. Fit both
# on the twelve flats and compare, then look at the residuals.
#
# The same twelve rows ship as datasets/housing-mini.csv (released to data/ in the
# cohort materials repo); inlined here so the demo runs anywhere.
#
# Run: Rscript demo.R

flats <- data.frame(
  area_sqm      = c(32, 45, 52, 60, 68, 75, 80, 95, 38, 55, 110, 48),
  dist_metro_km = c(0.3, 0.9, 0.4, 1.6, 0.7, 2.1, 1.1, 0.5, 1.8, 0.6, 1.4, 2.6),
  rent_eur      = c(540, 510, 640, 545, 720, 620, 770, 860, 420, 640, 930, 400)
)

# --- 1. the library ---------------------------------------------------------------
fit <- lm(rent_eur ~ area_sqm + dist_metro_km, data = flats)
cat("--- lm() ---\n")
print(summary(fit))

# --- 2. the same thing by hand ----------------------------------------------------
# Build the design matrix with an explicit intercept column, then SOLVE the normal
# equations rather than inverting X'X (same answer, better conditioned).
X <- cbind(intercept = 1, as.matrix(flats[, c("area_sqm", "dist_metro_km")]))
y <- flats$rent_eur
b_hat <- solve(t(X) %*% X, t(X) %*% y)

cat("\n--- normal equations ---\n")
print(round(b_hat, 6))
cat("\nmax abs difference from lm():",
    format(max(abs(as.vector(b_hat) - coef(fit))), scientific = TRUE), "\n")

# --- 3. read the fit --------------------------------------------------------------
cat("\nRMSE (EUR):", round(sqrt(mean(residuals(fit)^2)), 1), "\n")
cat("R-squared   :", round(summary(fit)$r.squared, 3), "\n")
cat("\nCoefficients read as: an extra square metre is worth about",
    round(coef(fit)[["area_sqm"]], 1), "EUR/month HOLDING DISTANCE FIXED.\n")

# --- 4. residuals: look before you trust ------------------------------------------
# Structure here means the functional form is wrong, not that the fit failed.
cat("\nresiduals vs fitted (structure = a missing transform):\n")
print(round(data.frame(fitted = fitted(fit), residual = residuals(fit)), 1))

# Uncomment when running interactively:
# par(mfrow = c(1, 2)); plot(fit, which = 1:2)
