# Session 7 demo - k-means and PCA on the same twelve flats.
#
# No labels today, so no test error: the output has to be argued for. Note the two
# non-negotiables - scale() before either method, and nstart > 1 for k-means.
#
# Run: Rscript demo.R

flats <- data.frame(
  row.names     = paste0("F", sprintf("%02d", 1:12)),
  area_sqm      = c(32, 45, 52, 60, 68, 75, 80, 95, 38, 55, 110, 48),
  dist_metro_km = c(0.3, 0.9, 0.4, 1.6, 0.7, 2.1, 1.1, 0.5, 1.8, 0.6, 1.4, 2.6),
  built_year    = c(1965, 1998, 1972, 2010, 1930, 1985, 2015, 1955, 1978, 2004, 1926, 1968)
)

# Standardise: k-means minimises Euclidean distance, so built_year (~2000) would
# otherwise drown out dist_metro_km (~1) entirely.
Z <- scale(flats)

# --- k-means ----------------------------------------------------------------------
set.seed(2026)
km <- kmeans(Z, centers = 3, nstart = 25)  # nstart: 25 restarts, keep the best local optimum

cat("--- k-means, k = 3 ---\n")
cat("cluster sizes:", km$size, "\n")
cat("within-cluster SS / total SS:",
    round(km$tot.withinss / km$totss, 3), "\n\n")

cat("cluster centres, back on the ORIGINAL scale (this is what makes them communicable):\n")
centres_raw <- t(apply(km$centers, 1, function(z)
  z * attr(Z, "scaled:scale") + attr(Z, "scaled:center")))
print(round(centres_raw, 1))

cat("\nassignments:\n")
print(km$cluster)

# Choosing k: the elbow, which often is not one.
cat("\nwithin-cluster SS across k (look for a kink, and do not expect one):\n")
for (k in 2:6) {
  set.seed(2026)
  cat(sprintf("  k = %d : %.2f\n", k, kmeans(Z, centers = k, nstart = 25)$tot.withinss))
}

# --- PCA --------------------------------------------------------------------------
pca <- prcomp(flats, scale. = TRUE)   # scale. = TRUE, for the same reason as above

cat("\n--- PCA ---\n")
print(summary(pca))
cat("\nloadings (interpret these, do not guess at the component names):\n")
print(round(pca$rotation, 3))
cat("\nRead the loadings, do not assume them: here PC1 contrasts large, OLDER flats\n")
cat("against small, newer ones (area +0.68, built_year -0.71), while distance to the\n")
cat("metro barely enters it and instead dominates PC2. Signs are arbitrary; the\n")
cat("CONTRAST is what carries meaning - and it is rarely the one you guessed.\n")

# Stability check: re-run on a 10-of-12 subsample. If the profiles move, you found noise.
set.seed(7)
keep <- sample(nrow(Z), 10)
set.seed(2026)
cat("\nsizes on a 10-row subsample:", kmeans(Z[keep, ], 3, nstart = 25)$size, "\n")

# Uncomment when running interactively:
# biplot(pca); plot(pca, type = "lines")
