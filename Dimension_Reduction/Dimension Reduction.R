#' ---
#' title: "Advanced Stats Assignment4"
#' output: html_document
#' date: "2023-02-27"
#' ---
#' 
## ----setup, include=FALSE--------------------------------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)
library(ISLR2)
library(pls)
library(dplyr)
library(glmnet)
library(faraway)

#' 
#' ### Question 3 Dimension Reduction & Regularization
#' 
#' (1) Split the data set into a training set and a test set. Perform a 80:20 split
#' 
## --------------------------------------------------------------------------------------------------------------------------------------------------------
# Data Overview
head(Boston)

# Omit NAs
Boston <- na.omit(Boston)
summary(Boston$crim)

# Split data
set.seed(9295)
sample_data <- sample(c(TRUE, FALSE), nrow(Boston), replace = TRUE, prob=c(0.8, 0.2))
train <- as.data.frame(Boston[sample_data, ])
test <- as.data.frame(Boston[!sample_data, ])

#' 
#' (2) Fit a PCR model on the training set, with M chosen by cross-validation. Report the test error obtained, along with the value of M selected by cross-validation.
#' 
#' The test error is 42.22504. The value M selected by cross validation is 12.
#' 
## --------------------------------------------------------------------------------------------------------------------------------------------------------
# Fit a PCR model
pcr_model <- pcr(crim~., data = train, scale = TRUE, validation = "CV")
summary(pcr_model)
validationplot(pcr_model, val.type = "MSEP")

# Save x and y separately
x_train = model.matrix(crim~., train)[,-1]
x_test = model.matrix(crim~., test)[,-1]
y_train = train %>%
  select(crim) %>%
  unlist() %>%
  as.numeric()
y_test = test %>%
  select(crim) %>%
  unlist() %>%
  as.numeric()

# Compute test error
pcr_pred = predict(pcr_model, x_test, ncomp = 12)
mse <- mean((pcr_pred-y_test)^2)
mse

# Compute accuracy
accuracy <- 1 - mse/var(y_test)
accuracy

#' 
#' (3) Fit a PLS model on the training set, with M chosen by cross-validation. Report the test error obtained, along with the value of M selected by cross-validation.
#' 
#' The test error is 42.22808. The value M selected by cross validation is 9.
#' 
## --------------------------------------------------------------------------------------------------------------------------------------------------------
# Fit a PLS model
pls_model <- plsr(crim~., data = train, scale = TRUE, validation = "CV")
summary(pls_model)
validationplot(pls_model, val.type = "MSEP")

# Compute test error
pls_pred = predict(pls_model, x_test, ncomp = 9)
mse <- mean((pls_pred-y_test)^2)
mse

# Compute accuracy
accuracy <- 1 - mse/var(y_test)
accuracy

#' 
#' (4) Fit a ridge regression model on the training set, with λ chosen by cross-validation. Report the test error obtained.
#' 
#' The test error is 42.2331. The λ selected by cross validation is 0.5532618.
#' 
## --------------------------------------------------------------------------------------------------------------------------------------------------------
# Fit a ridge regression model
ridge_model <- glmnet(x_train, y_train, alpha = 0)
summary(ridge_model)

# Find optimal lambdha
cv_model <- cv.glmnet(x_train, y_train, alpha = 0)
best_lambda <- cv_model$lambda.min
best_lambda
plot(cv_model) 

# Compute the error
ridge_pred = predict(ridge_model, s = best_lambda, newx = x_test)
mse <- mean((ridge_pred - y_test)^2)
mse

# Compute accuracy
accuracy <- 1 - mse/var(y_test)
accuracy

#' 
#' (5) Fit a lasso model on the training set, with λ chosen by cross-validation. Report the test error obtained, along with the number of non-zero coefficient estimates.
#' 
#' The test error is 42.1755. The λ selected by cross validation is 0.06981372. The number of non-zero coefficient estimates is 11.
#' 
## --------------------------------------------------------------------------------------------------------------------------------------------------------
# Fit a lasso regression model
lasso_model <- glmnet(x_train, y_train, alpha = 1)
summary(lasso_model)

# Find optimal lambdha
cv_model <- cv.glmnet(x_train, y_train, alpha = 1)
best_lambda <- cv_model$lambda.min
best_lambda
plot(cv_model) 

# Compute the error
lasso_pred = predict(lasso_model, s = best_lambda, newx = x_test)
mse <- mean((lasso_pred - y_test)^2)
mse

# Number of non-zero coefficient
lasso_coef = predict(lasso_model, type = "coefficients", s = best_lambda) # Display coefficients using lambda chosen by CV
lasso_coef
length(lasso_coef[lasso_coef != 0])

# Compute accuracy
accuracy <- 1 - mse/var(y_test)
accuracy

#' 
#' (6) Comment on the results obtained. How accurately can we predict the crime rate? Is there much difference among the test errors resulting from these approaches?
#' 
#' There is no much difference among the test errors resulting from these approaches. We can not predict the crime rate accurately using these models. And since Boston data has a high degree of variability and noise, it's reasonable that we can not fit a linear model to do an accurate prediction.
