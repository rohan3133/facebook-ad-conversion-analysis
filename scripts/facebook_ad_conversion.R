# Load the dataset
getwd()
facebook <- read.csv("data/facebook.csv")

# 2. Initial Exploration
str(facebook)
dim(facebook)
summary(facebook)

# 3. Remove 'ad_id', 'xyzcampaignid', 'fbcampaignid'
facebook<-facebook[,4:11]
str(facebook)

# Convert demographics 
facebook$gender <- as.factor(facebook$gender)

# 4. Create New Target Variable (Conversion Category)
facebook$Approved_Conversion_factor <- factor(
  ifelse(facebook$Approved_Conversion <= 2, "Low",
         ifelse(facebook$Approved_Conversion < 10, "Average", "High")))

table(facebook$Approved_Conversion_factor)

# 5. Visualize Approved Conversion
hist(facebook$Approved_Conversion, 
     main = "Histogram of Approved Conversions", 
     xlab = "Approved Conversions", 
     col = "darkblue")

# 6. Log Transformation
facebook$log_Approved_Conversion <- log(facebook$Approved_Conversion + 1)
hist(facebook$log_Approved_Conversion, 
     main = "Histogram of Log-Transformed Approved Conversions",
     xlab = "Log Approved Conversions", col = "red")

# 7. Add Click-Through Rate (CTR)
facebook$CTR <- facebook$Clicks / facebook$Impressions
summary(facebook$CTR)

# 8. Correlation Plot
library(corrplot)
corrplot(corr= cor(facebook),method ="ellipse")
facebook$age <- as.integer(facebook$age)
facebook$gender <- as.integer(facebook$gender)
corrplot(cor(facebook), method = "ellipse")
facebook$age <- factor(facebook$age)
facebook$gender <- factor(facebook$gender)


# 9. Train-Test Split
set.seed(123)
train.prop <- 0.65
train.cases <- sample(nrow(facebook), nrow(facebook) * train.prop)
fb.train <- facebook[train.cases, ]
fb.test <- facebook[-train.cases, ]


# 10. Naive Bayes Model
library(naivebayes)
adconversion.nb <- naive_bayes(Approved_Conversion_factor ~ ., data = fb.train)
adconversion.nb

# Model Evaluation
fb.nb.class <- predict(adconversion.nb, fb.test)
accuracy_nb <- mean(fb.test$Approved_Conversion_factor == fb.nb.class)
accuracy_nb

library(mclust)
adjustedRandIndex_nb <- adjustedRandIndex(fb.nb.class, fb.test$Approved_Conversion_factor)
adjustedRandIndex_nb

level.summary<-function(data,groups){
  aggregate(data,list(groups),function(x)mean(as.numeric(x)))
}
(level.summary(fb.test,fb.nb.class)) 
(level.summary(fb.test,fb.test$Approved_Conversion_factor)) 


# 11. Random Forest Model
library(randomForest)
fb.rf <- randomForest(Approved_Conversion_factor ~ ., data = fb.train,
                      ntree = 11430, importance = TRUE)
fb.rf


# Model Evaluation
fb.rf.class <- predict(fb.rf, fb.test)
accuracy_rf <- mean(fb.test$Approved_Conversion_factor == fb.rf.class)
accuracy_rf

library(mclust)
adjustedRandIndex_rf <- adjustedRandIndex(fb.rf.class, fb.test$Approved_Conversion_factor)
adjustedRandIndex_rf

library(RColorBrewer)
heatmap.2(t(importance(fb.rf)[,-6:-7]), key = FALSE,
          col = brewer.pal(9, "Blues"),
          dend="none", trace = "none", margins = c(10,10),
          main = "n\n\n\n\n\n\n\n\n\n\nVar. importance by Segment")

fb.rf.resize <- randomForest(Approved_Conversion_factor~., data = fb.train,
                             ntree = 11430,
                             importance = TRUE, sampsize=c(6,6,6))
#Matching original dimenions of "High"
fb.rf.resize

resize.class<-predict(fb.rf.resize,fb.test)
table(resize.class,fb.test$Approved_Conversion_factor)

importance.resize<-importance(fb.rf.resize)
importance.resize

# Evaluate performance of resized RF
accuracy_resize <- mean(fb.test$Approved_Conversion_factor == resize.class)
accuracy_resize

adjustedRandIndex_resize <- adjustedRandIndex(resize.class, fb.test$Approved_Conversion_factor)
adjustedRandIndex_resize


# Variable Importance
importance.rf <- importance(fb.rf)
importance.rf

(varImpPlot(fb.rf, main = "Variable importance by Approved_Conversion"))

(varImpPlot(fb.rf.resize, main = "Variable importance by Approved_Conversion with Resizing"))

# Compare accuracy and adjusted Rand index
accuracy_comparison <- data.frame(Method = c("Naive Bayes", "Random Forest", "Random Forest w/ Resize")Accuracy = c(accuracy_nb, accuracy_rf, accuracy_resize),
                                  Adjusted_Rand_Index = c(adjustedRandIndex_nb, adjustedRandIndex_rf, adaccuracy_comparison

