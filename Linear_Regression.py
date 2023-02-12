# %%
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.factorplots import interaction_plot

# %% [markdown]
# 1. Using the sales.csv, write code to show effects of interactions, if any, on the linear regression model to predict the total_sales for a new area using given sales from three areas.

# %%
# Read the csv
sales = pd.read_csv("sales.csv")

# EDA
len(sales)
sales.info()
sales.describe()
sales.isnull().sum()

# Plots
sales.plot.scatter("area1_sales", "total_sales")
sales.plot.scatter("area2_sales", "total_sales")
sales.plot.scatter("area3_sales", "total_sales")

# Save the values
x = sales[["area1_sales", "area2_sales", "area3_sales"]] 
y = sales["total_sales"]

# %%
# Split the data into training group and testing group
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
train_data = x_train.join(y_train)
test_data = x_test.join(y_test)

# Call multiple linear regression
model = smf.ols(formula='total_sales ~ area1_sales + area2_sales + area3_sales', data = train_data).fit()
summary = model.summary()
print(summary)

# Including interaction term
model_interaction = smf.ols(formula='total_sales ~ area1_sales + area2_sales + area3_sales + area1_sales:area2_sales + area2_sales:area3_sales + area1_sales:area3_sales + area1_sales:area2_sales:area3_sales', data = train_data).fit()
summary_interaction = model_interaction.summary()
print(summary_interaction)

# Remove insignificant terms
model_interaction2 = smf.ols(formula='total_sales ~ area1_sales + area2_sales + area3_sales + area1_sales:area3_sales + area1_sales:area2_sales:area3_sales', data = train_data).fit()
summary_interaction2 = model_interaction2.summary()
print(summary_interaction2)

# %% [markdown]
# According to the model built, we can observe that the p-values of area1_sales:area3_sales and area1_sales:area2_sales:area3_sales is under any reasonable significance level, implying some interactions occur between the sales of them.

# %%
# Predict 
model_interaction2.predict(test_data)

# %% [markdown]
# 2. Develop a full Logistic Regression Model using customer.csv to predict whether the customer will purchase the product. Also train trimmed logistic regression models (Trimmed over features in the data). Compute the "in-sample R2" (pseudo) for the models you train and compare the models based on this metric.

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
# 3. For the Logistic Regression models trained above, pick the best model wrt to the in-sample R2 and give your interpretation of the model’s coefficients (For example, what effect does a positive or negative coefficient have on the model and so on).

# %%
print(np.exp(3.1339))
print(np.exp(2.0412))
print(np.exp(-1.7502))
print(np.exp(3.1339 + 2.0412 - 1.7502))

# %% [markdown]
# Based on two calculated pseudo R squared values, we should pick the full model with a higher pseudo R squared value of 0.6028. Under 95% confidence level, Age, EstimatedSalary and interaction term of these two variables are significant. Holding other variables fixed, there is a 2196% increase in the odds of purchase for every standard deviation(10.46) increase in age. Holding other variables fixed, there is a 669% increase in the odds of purchase for every standard deviation(34054.31) increase in estimated salary. For every 1 standard deviation increase in age and estimated salary, there will be a 2971% increase in the odds of purchase.

# %% [markdown]
# 4. Is accuracy a good metric to judge the above model? Give reasons and alternatives to support your answer.

# %% [markdown]
# Accuracy is not a good metric as we are looking at in-sample R squared here, which usually overestimate the real prediction accuracy. Moreover, since data of purchased is imbalanced (only 35% of data are labelled as "1"), even when model fails to preduct any purchase behavior, the accuracy is still around 65%. Some alternatives can be precision and F1 score.

# %% [markdown]
# 5. Plot the interactions of the ‘Age’ and ‘Gender’ features with the ‘Purchased’ output. 

# %%
# Create bins for age group  
bins = [-np.inf, 30, np.inf]
customers_plot['Age_Group'] = pd.cut(customers_plot['Age'], bins)


# Plot the interaction plot
interaction_plot(customers_plot['Age_Group'], customers_plot['Gender'], customers_plot['Purchased'])
plt.show


# %% [markdown]
# 6. The following plots show effect of Income and Change in savings on Likelihood of buying a house. Observe the three plots (a, b, c) shown and answer the following questions for each plot:
# I. Should the regression equation include the interaction terms (interaction between Income and Average Savings) or not?
# II. Give your reasoning as to why or why shouldn’t they be included.

# %% [markdown]
# (a) Obviously, the slopes of three lines in this plot are different, symbolizes some interaction may occur. According to the plot, as average saving increases, the likelihood of high income group buying a house increases, while the low income group remains almost the same. If the effect of hypothesis test is significant, interaction terms should be included.
# (b) Similarly, the slopes of two lines in this plot are not the same as well. Instead, it shows a moderate reversal interaction, so the interaction terms should be included.
# (c) In this case, two lines are parallel, showing that no interaction occurs, therefore interaction term should be removed.


