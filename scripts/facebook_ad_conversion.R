# =============================================================================
# Facebook Ad Conversion Analysis
# -----------------------------------------------------------------------------
# Goal : Classify ads as Low / Average / High converting and identify which ad
#        features drive conversions.
# Data : data/facebook.csv  (1,143 ads, 11 variables)
# Run  : from the repository root ->  Rscript scripts/facebook_ad_conversion.R
# Output: figures/*.png and results/*.csv
# =============================================================================

suppressPackageStartupMessages({
  library(corrplot)      # correlation plot
  library(e1071)         # Naive Bayes
  library(randomForest)  # Random Forest
  library(caret)         # stratified splits, confusion matrices
  library(mclust)        # Adjusted Rand Index
})

dir.create("figures", showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

# -----------------------------------------------------------------------------
# 1. Load and explore
# -----------------------------------------------------------------------------
facebook <- read.csv("data/facebook.csv")
str(facebook)
dim(facebook)
summary(facebook)

# Remove ID columns: ad_id, xyz_campaign_id, fb_campaign_id
facebook <- facebook[, 4:11]
facebook$gender <- as.factor(facebook$gender)

# -----------------------------------------------------------------------------
# 2. Target variable: conversion level
#    Approved_Conversion <= 2 -> Low, < 10 -> Average, otherwise High
# -----------------------------------------------------------------------------
facebook$Approved_Conversion_factor <- factor(
  ifelse(facebook$Approved_Conversion <= 2, "Low",
         ifelse(facebook$Approved_Conversion < 10, "Average", "High")),
  levels = c("Low", "Average", "High")
)
print(table(facebook$Approved_Conversion_factor))

# -----------------------------------------------------------------------------
# 3. Distribution of conversions (heavily right-skewed -> log transform)
# -----------------------------------------------------------------------------
facebook$log_Approved_Conversion <- log(facebook$Approved_Conversion + 1)

png("figures/conversion_distribution.png", width = 1600, height = 650, res = 150)
par(mfrow = c(1, 2))
hist(facebook$Approved_Conversion, col = "#2b6cb0", border = "white",
     main = "Approved conversions", xlab = "Approved conversions per ad")
hist(facebook$log_Approved_Conversion, col = "#dd6b20", border = "white",
     main = "Log-transformed", xlab = "log(Approved conversions + 1)")
dev.off()

# -----------------------------------------------------------------------------
# 4. Feature engineering: click-through rate
# -----------------------------------------------------------------------------
facebook$CTR <- facebook$Clicks / facebook$Impressions
summary(facebook$CTR)

# -----------------------------------------------------------------------------
# 5. Correlation analysis
# -----------------------------------------------------------------------------
cor_data <- data.frame(
  age                 = as.integer(factor(facebook$age)),
  gender              = as.integer(facebook$gender),
  interest            = facebook$interest,
  Impressions         = facebook$Impressions,
  Clicks              = facebook$Clicks,
  Spent               = facebook$Spent,
  CTR                 = facebook$CTR,
  Total_Conversion    = facebook$Total_Conversion,
  Approved_Conversion = facebook$Approved_Conversion
)
png("figures/correlation_matrix.png", width = 1400, height = 1400, res = 170)
corrplot(cor(cor_data), method = "ellipse", type = "upper", tl.col = "black",
         tl.cex = 0.9, addCoef.col = "black", number.cex = 0.6, diag = FALSE)
dev.off()
write.csv(round(cor(cor_data), 3), "results/correlation_matrix.csv")

# -----------------------------------------------------------------------------
# 6. Modeling setup
#    The conversion level is derived from Approved_Conversion, so
#    Approved_Conversion (and its log) must NOT be used as predictors.
# -----------------------------------------------------------------------------
target <- "Approved_Conversion_factor"
leaky  <- c("Approved_Conversion", "log_Approved_Conversion")
all_predictors <- setdiff(names(facebook), c(target, leaky))
# Sensitivity analysis: also drop Total_Conversion (enquiries), a funnel metric
ad_only_predictors <- setdiff(all_predictors, "Total_Conversion")

evaluate <- function(pred, actual, model) {
  pred <- factor(pred, levels = levels(actual))
  cm   <- confusionMatrix(pred, actual)
  data.frame(
    Model             = model,
    Accuracy          = unname(cm$overall["Accuracy"]),
    Balanced_Accuracy = mean(cm$byClass[, "Balanced Accuracy"], na.rm = TRUE),
    Adjusted_Rand_Index = adjustedRandIndex(pred, actual),
    Recall_Low        = unname(cm$byClass["Class: Low", "Sensitivity"]),
    Recall_Average    = unname(cm$byClass["Class: Average", "Sensitivity"]),
    Recall_High       = unname(cm$byClass["Class: High", "Sensitivity"]),
    row.names = NULL
  )
}

# Fit the three models on a train set and return predictions on a test set
fit_predict <- function(train, test, predictors, ntree) {
  f <- reformulate(predictors, response = target)

  nb <- naiveBayes(f, data = train)
  rf <- randomForest(f, data = train, ntree = ntree, importance = TRUE)
  # Balanced Random Forest: sample the same number of ads from every class
  n_min <- min(table(train[[target]]))
  rf_bal <- randomForest(f, data = train, ntree = ntree, importance = TRUE,
                         strata = train[[target]],
                         sampsize = rep(n_min, nlevels(train[[target]])))
  list(
    models = list(nb = nb, rf = rf, rf_bal = rf_bal),
    pred = list(
      `Naive Bayes`                 = predict(nb, test),
      `Random Forest`               = predict(rf, test),
      `Random Forest (balanced)`    = predict(rf_bal, test)
    )
  )
}

# -----------------------------------------------------------------------------
# 7. Main analysis: 65/35 stratified train-test split
# -----------------------------------------------------------------------------
set.seed(123)
idx   <- createDataPartition(facebook[[target]], p = 0.65, list = FALSE)
train <- facebook[idx, ]
test  <- facebook[-idx, ]
cat("\nTrain:", nrow(train), "ads | Test:", nrow(test), "ads\n")
print(table(train[[target]]))

run_main <- function(predictors, label, file_suffix) {
  fit <- fit_predict(train, test, predictors, ntree = 11430)
  res <- do.call(rbind, lapply(names(fit$pred), function(m)
    evaluate(fit$pred[[m]], test[[target]], m)))
  res$Features <- label
  write.csv(res, paste0("results/model_comparison", file_suffix, ".csv"), row.names = FALSE)
  cat("\n=== ", label, " ===\n", sep = "")
  print(res, digits = 3)
  for (m in names(fit$pred)) {
    cat("\nConfusion matrix:", m, "\n")
    print(table(Predicted = fit$pred[[m]], Actual = test[[target]]))
  }
  fit
}

fit_all <- run_main(all_predictors,     "All ad features (incl. Total_Conversion)", "")
fit_ad  <- run_main(ad_only_predictors, "Ad features only (excl. Total_Conversion)", "_ad_features_only")

# Variable importance (Random Forest, mean decrease in Gini)
save_importance <- function(model, file, title) {
  imp <- importance(model)[, "MeanDecreaseGini"]
  imp <- sort(imp)
  png(file, width = 1400, height = 800, res = 170)
  par(mar = c(4, 9, 3, 1))
  barplot(imp, horiz = TRUE, las = 1, col = "#2b6cb0", border = NA,
          main = title, xlab = "Mean decrease in Gini")
  dev.off()
  write.csv(data.frame(Feature = names(rev(imp)), MeanDecreaseGini = unname(rev(imp))),
            sub("\\.png$", ".csv", sub("figures/", "results/", file)), row.names = FALSE)
}
save_importance(fit_all$models$rf, "figures/importance_all_features.png",
                "Feature importance (all ad features)")
save_importance(fit_ad$models$rf,  "figures/importance_ad_features_only.png",
                "Feature importance (excluding Total_Conversion)")

# -----------------------------------------------------------------------------
# 8. Robustness: repeat the evaluation over 20 random stratified splits
# -----------------------------------------------------------------------------
repeat_eval <- function(predictors, label, n_rep = 20) {
  out <- lapply(seq_len(n_rep), function(i) {
    set.seed(1000 + i)
    id <- createDataPartition(facebook[[target]], p = 0.65, list = FALSE)
    tr <- facebook[id, ]; te <- facebook[-id, ]
    fit <- fit_predict(tr, te, predictors, ntree = 500)
    do.call(rbind, lapply(names(fit$pred), function(m)
      evaluate(fit$pred[[m]], te[[target]], m)))
  })
  all <- do.call(rbind, out)
  agg <- aggregate(cbind(Accuracy, Balanced_Accuracy, Adjusted_Rand_Index, Recall_High) ~ Model,
                   data = all, FUN = function(x) c(mean = mean(x), sd = sd(x)))
  agg <- do.call(data.frame, agg)
  agg$Features <- label
  agg
}
rob <- rbind(repeat_eval(all_predictors, "All ad features"),
             repeat_eval(ad_only_predictors, "Ad features only"))
write.csv(rob, "results/robustness_20_splits.csv", row.names = FALSE)
cat("\n=== Robustness over 20 random splits (mean, sd) ===\n")
print(rob, digits = 3)

cat("\nDone. See figures/ and results/.\n")
