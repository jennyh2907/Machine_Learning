#' ---
#' title: "Advanced_Stats_CV"
#' output: html_document
#' date: "2023-02-08"
#' ---
#' 
## ----setup, include=FALSE--------------------------------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)
library(caret)
library(glmnet)

#' 
#' ### 1. Construct the data
#' 
## --------------------------------------------------------------------------------------------------------------------------------------------------------
# Set seed
set.seed(9295)

# Create x and error term
n <- 10000
x <- runif(n, min = 0, max = 1)
e <- rnorm(n, mean = 0, sd = sqrt(0.5))

# Compute Y from X
y <- 3 * x^5 + 2 * x^2 + e

# Put them in a dataframe
df <- data.frame(x, y)

#' 
#' ### 2. Split the 10000 points into a 80% training and 20% test split. Use a seed before randomizing to replicate results.
#' 
## --------------------------------------------------------------------------------------------------------------------------------------------------------
# Split the data
train_indices <- createDataPartition(y = df$y, p = 0.8, list = FALSE)
train <- df[train_indices, ]
test <- df[-train_indices, ]


#' 
#' ### 3. Split the training set into 5 parts and use the five folds to choose the optimal d. The loss function you would implement is the MSE error. You want to estimate the MSE error on each fold for a model that has been trained on the remaining 4 folds. The cross validation (CV) error for the training set would be the average MSE across all five folds. Plot the CV error as a function of d for d ∈[1,2,\...,10]
#' 
#' The optimal d value is 4, which has the lowest MSE of 0.50191.
#' 
## --------------------------------------------------------------------------------------------------------------------------------------------------------
# Create a vector to store the cross-validation errors
cv_errors <- rep(0, 5)
d_value <- 1:10

# Perform 5-fold cross validation
for (d in d_value){
  fold_e <- rep(0, 5)
  for (i in 1:5) {
      folds <- cut(seq(1, nrow(train)),breaks = 5,labels = FALSE)
      testIndex <- which(folds==i, arr.ind=TRUE)
      test_cv <- train[testIndex, ]
      train_cv <- train[-testIndex, ]
      model <- lm(y ~ poly(x, degree = d), data = train_cv)
      predictions <- predict(model, newdata = test_cv)
      fold_e[i] <- mean((predictions - test_cv$y)^2)
  }
  cv_errors[d] <- mean(fold_e)
  }

# Plotting
plot(d_value, cv_errors, type = "l", xlab = "d", ylab = "CV Error")


#' 
#' ### 4. In this subpart, use the entire training set for training the models. Compute the performance of the 10 models on the test set. Plot the test MSE and training MSE as a function of d. Comment on your observations.
#' 
#' As we use training set to build the model, no wonder the model fits the training data more, which results in lower MSE at all d values. In addition, the MSE values roughly remain the same when d is above 3, so if we want to have a more accurate model, we should choose those with d value \> 3.
#' 
## --------------------------------------------------------------------------------------------------------------------------------------------------------
# Use the whole training set and test it on test set
MSE_test <- rep(0, 10)
MSE_train <- rep(0, 10)

for (d in d_value){
      model <- lm(y ~ poly(x, degree = d), data = train)
      predictions_train <- predict(model, newdata = train)
      predictions_test <- predict(model, newdata = test)
      MSE_train[d] <- mean((predictions_train - train$y)^2)
      MSE_test[d] <- mean((predictions_test - test$y)^2)
  }

# Plotting
plot(d_value, MSE_test, type = "l", xlab = "d", ylab = "MSE", 
     main = "Test and Training MSE vs. d")
lines(d_value, MSE_train, col = "red")
legend("topright", c("Test MSE", "Training MSE"), col = c("black", "red"), 
       lty = 1)



