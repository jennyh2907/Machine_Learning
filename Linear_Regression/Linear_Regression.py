# %%
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf

# %% [markdown]
# Using the sales.csv, show effects of interactions, if any, on the linear regression model to predict the total_sales for a new area using given sales from three areas.

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


