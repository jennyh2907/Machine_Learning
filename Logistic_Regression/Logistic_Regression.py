# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.factorplots import interaction_plot

# %% [markdown]
# Develop a full Logistic Regression Model using customer.csv to predict whether the customer will purchase the product. 
# Also train trimmed logistic regression models . Compute the "in-sample R2" (pseudo) for the models and compare the models based on this metric.

# %%
# Read the csv
customers = pd.read_csv("customer.csv")
customers_plot = pd.read_csv("customer.csv")

# EDA
len(customers)
customers.info()
customers.describe()
customers.isnull().sum()

# Balance or Imbalance dataset > 143 in 400, imbalanced dataset
print(customers['Purchased'].value_counts()[1])
print(customers['Gender'].value_counts()["Male"])

# Convert to dummies
customers["Gender"] = customers["Gender"].apply(lambda x: 1 if x=="Male" else 0)

# Save the values
var = pd.DataFrame(customers[["Gender", "Purchased"]])
x = customers[["Age", "EstimatedSalary"]]

# Scale the variables
scale = StandardScaler()
scale_x = scale.fit_transform(x)

# Rename the columns
scale_x = pd.DataFrame(scale_x, columns = ["Age", "EstimatedSalary"])

# Join data
customers = var.join(scale_x)

# Logistic Regression full model
formula = 'Purchased ~ Gender + Age + EstimatedSalary + Age:EstimatedSalary + Gender:EstimatedSalary + Age:Gender + Age:Gender:EstimatedSalary'
model = smf.glm(formula = formula, data=customers, family=sm.families.Binomial())
result = model.fit()
print(result.summary())

# Compute pseudo R squared for full model
full_d = result.deviance
full_d0 = result.null_deviance
print((full_d0-full_d)/full_d0)


# %%
# Train different models
formula_trim = 'Purchased ~ Gender + Age + EstimatedSalary + Age:EstimatedSalary + Gender:EstimatedSalary + Age:Gender'
model_trim = smf.glm(formula = formula_trim, data = customers, family=sm.families.Binomial())
result_trim = model_trim.fit()

trim_d = result_trim.deviance
trim_d0 = result_trim.null_deviance
print((trim_d0-trim_d)/trim_d0)

formula_trim = 'Purchased ~ Gender + Age + EstimatedSalary + Age:EstimatedSalary + Age:Gender'
model_trim = smf.glm(formula = formula_trim, data = customers, family=sm.families.Binomial())
result_trim = model_trim.fit()

trim_d = result_trim.deviance
trim_d0 = result_trim.null_deviance
print((trim_d0-trim_d)/trim_d0)

formula_trim = 'Purchased ~ Gender + Age + EstimatedSalary + Age:Gender + Gender:EstimatedSalary'
model_trim = smf.glm(formula = formula_trim, data = customers, family=sm.families.Binomial())
result_trim = model_trim.fit()

trim_d = result_trim.deviance
trim_d0 = result_trim.null_deviance
print((trim_d0-trim_d)/trim_d0)

formula_trim = 'Purchased ~ Gender + Age + EstimatedSalary + Age:EstimatedSalary + Gender:EstimatedSalary'
model_trim = smf.glm(formula = formula_trim, data = customers, family=sm.families.Binomial())
result_trim = model_trim.fit()

trim_d = result_trim.deviance
trim_d0 = result_trim.null_deviance
print((trim_d0-trim_d)/trim_d0)

formula_trim = 'Purchased ~ Age + EstimatedSalary + Age:EstimatedSalary'
model_trim = smf.glm(formula = formula_trim, data = customers, family=sm.families.Binomial())
result_trim = model_trim.fit()

trim_d = result_trim.deviance
trim_d0 = result_trim.null_deviance
print((trim_d0-trim_d)/trim_d0)

formula_trim = 'Purchased ~ Age + EstimatedSalary + Gender + Age:EstimatedSalary'
model_trim = smf.glm(formula = formula_trim, data = customers, family=sm.families.Binomial())
result_trim = model_trim.fit()

trim_d = result_trim.deviance
trim_d0 = result_trim.null_deviance
print((trim_d0-trim_d)/trim_d0)

formula_trim = 'Purchased ~ Gender + Age + EstimatedSalary'
model_trim = smf.glm(formula = formula_trim, data = customers, family=sm.families.Binomial())
result_trim = model_trim.fit()

trim_d = result_trim.deviance
trim_d0 = result_trim.null_deviance
print((trim_d0-trim_d)/trim_d0)

# Final model
formula_trim = 'Purchased ~ Gender + Age + EstimatedSalary + Age:EstimatedSalary + Gender:EstimatedSalary + Age:Gender + Age:Gender:EstimatedSalary'
model_trim = smf.glm(formula = formula_trim, data = customers, family=sm.families.Binomial())
result_trim = model_trim.fit()
print(result_trim.summary())

# Compute pseudo R squared for trimmed model
trim_d = result_trim.deviance
trim_d0 = result_trim.null_deviance
print((trim_d0-trim_d)/trim_d0)

# Odd ratio
p = (len(customers[customers['Purchased']== 1])/len(customers))
print(p/(1-p))

# %% [markdown]
# For the Logistic Regression models trained above, pick the best model wrt to the in-sample R2 

# %%
print(np.exp(3.1339))
print(np.exp(2.0412))
print(np.exp(-1.7502))
print(np.exp(3.1339 + 2.0412 - 1.7502))

# %% [markdown]
# Based on two calculated pseudo R squared values, we should pick the full model with a higher pseudo R squared value of 0.6028. Under 95% confidence level, Age, EstimatedSalary and interaction term of these two variables are significant. Holding other variables fixed, there is a 2196% increase in the odds of purchase for every standard deviation(10.46) increase in age. Holding other variables fixed, there is a 669% increase in the odds of purchase for every standard deviation(34054.31) increase in estimated salary. For every 1 standard deviation increase in age and estimated salary, there will be a 2971% increase in the odds of purchase.
# However, accuracy is not a good metric as we are looking at in-sample R squared here, which usually overestimate the real prediction accuracy. Moreover, since data of purchased is imbalanced (only 35% of data are labelled as "1"), even when model fails to preduct any purchase behavior, the accuracy is still around 65%. Some alternatives can be precision and F1 score.

# %% [markdown]
# Plot the interactions of the ‘Age’ and ‘Gender’ features with the ‘Purchased’ output. 

# %%
# Create bins for age group  
bins = [-np.inf, 30, np.inf]
customers_plot['Age_Group'] = pd.cut(customers_plot['Age'], bins)


# Plot the interaction plot
interaction_plot(customers_plot['Age_Group'], customers_plot['Gender'], customers_plot['Purchased'])
plt.show