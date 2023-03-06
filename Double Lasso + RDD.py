# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

# %% [markdown]
# 1. Using hotel_cancellation.csv, write code to estimate the treatment effects if a ‘different room is assigned’ as the treatment indicator and interpret its effect on the room being ‘canceled’. Use all the other columns as the covariates. Write your observations for the results.
# 
#     The treatment effect for the variable 'treatment' (whether different room is assigned to each customer) represents the difference in the odds of cancellation between those being assigned to different rooms and those who don't, controlling for all other covariates. According to the result, those who being assigned to different room are less likely to cancel by 2.518 compared to those who don't.

# %%
# Read the csv
hotel = pd.read_csv("hotel_cancellation.csv", index_col=False)

# Rename the treatment indicator and y column
hotel = hotel.rename({'different_room_assigned': 'treatment', 'is_canceled': 'outcome'}, axis=1)

# Change the treatment variable into binary variable
hotel['treatment'] = hotel['treatment'].apply(lambda x: 1 if x == True else 0)
hotel['outcome'] = hotel['outcome'].apply(lambda x: 1 if x == True else 0)

# Specify the treatment and response variables
y = hotel['outcome']
x = hotel[['treatment', 'lead_time', 'arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month', 'days_in_waiting_list']]

# Fit a logistic regression model
model = sm.Logit(y, x)
result = model.fit()

# Print the treatment effect estimates
print(result.params)


# %% [markdown]
# 2. For hotel_cancellation.csv, now use double now use double LOGISTIC regression to measure the effect of ‘different room is assigned’ on the room being ‘canceled’.
# 
#     The treatment effect for the variable 'x1' (whether different room is assigned to each customer) represents the difference in the odds of cancellation between those being assigned to different rooms and those who don't, controlling for all other covariates. According to the result, those who being assigned to different room are less likely to cancel by 1.981 compared to those who don't.

# %%
model1 = sm.Logit(y, x).fit()
y_hat = np.array(model1.predict(x)).reshape(len(x), 1)
x_new = np.hstack((x, y_hat))
model2 = sm.Logit(y, x_new).fit()
print(model2.params)

# %% [markdown]
# 3. Use bootstrap to estimate the standard error of the treatment effects measured in (2).

# %%
# Define the number of bootstrap resamples
n_resamples = 1000

# Initialize a matrix to store the treatment effect estimates
treat_effects = np.zeros((n_resamples, model2.params.shape[0] - 1))

# Use bootstrapping to estimate the standard error of the treatment effects
i = 0
while i < n_resamples:
    resample_index = np.random.choice(hotel.index, size = hotel.index.size, replace = True)
    resample = hotel.iloc[resample_index]
    x_resample = x.iloc[resample_index]
    y_resample = y.iloc[resample_index]
    model1 = sm.Logit(y_resample, x_resample).fit()
    y_hat = np.array(model1.predict(x_resample)).reshape(len(y_hat), 1)
    x_new = np.hstack((x_resample, y_hat))
    model2 = sm.Logit(y_resample, x_new).fit()
    treat_effects[i, :] = model2.params[:-1]
    i += 1

# Calculate the standard error of the treatment effects
treat_effects_se = treat_effects.std(axis=0)

# Print the standard errors of the treatment effect estimates
print('Standard errors of the treatment effects:')
print(treat_effects_se)

# %% [markdown]
# 4. Use the drinking.csv dataset for this question. Keeping 21 as the threshold for age, explore the data with an RDD by writing very simple code (no package needed, just average to one side of the threshold minus average to the other side) to determine if alcohol increases the chances of death by accident, suicide and/or others (the three given columns) and comment on the question “Should the legal age for drinking be reduced from 21?” based on the results. Plot graphs to show the discontinuity (if any) and to show results for the change in chances of death with all the three features (i.e., accident vs age, suicide vs age and others vs age). For this problem, choose the bandwidth to be 1 year (i.e., 21 +- 1). What might be the effect of choosing a smaller bandwidth?  What if we chose the maximum bandwidth?
# 
#     In the first stage, as we keep 21 as threshold and set the maximum bandwidth, the average treament effect indicates that those above 21 have a higher chance to died in suicide and other factors, but not in accident. Accordingly, we can not determine whether the legal drinking age should be reduced or not. However, as we choose a smaller bandwidth (21+-1) and redo RDD, the result flips. The treament effect after controlling badwidth shows that we underestimate the treatment effect of being 21 in previous stage. In fact, the treatment effect of being 21 and above leads to a higher chance to died in suicide, other factors, as well as accident. Therefore, we have no reason to reduce the legal drinking age.
# 
#     Because it's hard to control for all covariates for far end people(19-year-old people) and those being close to threshold(20.5-year-old people), we usually choose a smaller bandwidthavoid to avoid underestimating treatment effect. On the contrary, if we choose the maximum bandwidth, there is high probability that the treatment effect will be inaccurate, or even suggesting an misleading result. 

# %%
# Read the csv and drop NA
drink = pd.read_csv("drinking.csv")
drink = drink.dropna()

# Filter for data that's under 21
Under_21 = drink[drink['age'] < 21]
Above_21 = drink[drink['age'] >= 21]

# Take average of data that's under 21
Others_u21 = Under_21["others"].mean()
Accident_u21 = Under_21["accident"].mean()
Suicide_u21 = Under_21["suicide"].mean()

# Take average of data that's above 21
Others_a21 = Above_21["others"].mean()
Accident_a21 = Above_21["accident"].mean()
Suicide_a21 = Above_21["suicide"].mean()

# Print the result
print("[Under_21] ","Others:", Others_u21, " Accident:", Accident_u21, " Suicide:", Suicide_u21)
print("[Above_21] ", "Others:", Others_a21, " Accident:", Accident_a21, " Suicide:", Suicide_a21)
print("[Above_21-Under_21 Difference] ", "Others:", Others_a21 - Others_u21, " Accident:", Accident_a21 - Accident_u21, " Suicide:", Suicide_a21 - Suicide_u21)

# Plot
drink.plot.scatter(x="age", y="others")
plt.axvline(x=21, color='r', linestyle='-')
plt.title("Other Death Cause vs Age")

drink.plot.scatter(x="age", y="accident")
plt.axvline(x=21, color='r', linestyle='-')
plt.title("Accident vs Age")

drink.plot.scatter(x="age", y="suicide");
plt.axvline(x=21, color='r', linestyle='-')
plt.title("Suicide vs Age")

# Set the bandwidth to 20-21 and 21-22
S20_21 = Under_21[Under_21['age'] > 20]
S21_22 = Above_21[Above_21['age'] < 22]

# Take average of data that's under 21
Others_20_21 = S20_21["others"].mean()
Accident_20_21 = S20_21["accident"].mean()
Suicide_20_21 = S20_21["suicide"].mean()

# Take average of data that's above 21
Others_21_22 = S21_22["others"].mean()
Accident_21_22 = S21_22["accident"].mean()
Suicide_21_22 = S21_22["suicide"].mean()

# Print out
print("-------------------")
print("[20_21] ","Others:", Others_20_21, " Accident:", Accident_20_21, " Suicide:", Suicide_20_21)
print("[21_22] ", "Others:", Others_21_22, " Accident:", Accident_21_22, " Suicide:", Suicide_21_22)
print("[21_22-20_21 Difference] ", "Others:", Others_21_22 - Others_20_21, " Accident:", Accident_21_22 - Accident_20_21, " Suicide:", Suicide_21_22 - Suicide_20_21)


# %% [markdown]
# 5. Use the iris.csv dataset for this question. How does the performance of k-nearest neighbors change as k takes on the following values: 1, 3, 5, 7? Which of these is the optimal value of k? Which distance/similarity metric did you choose to use and why?
# 
#     The accuracy score of k = 1 and 7 is 0.933. For k = 3 and 5 is 0.967. The optimal k can either be 3 or 5, as they both have the highest accuracy score of 0.967. With respect to different distance/similarity metric, Euclidean distance is well suited for continuous, numerical data. Compared to Euclidean distance, Manhattan distance is less sensitive to outliers. In addition, it can fit categorical data well. Cosine similarity is commonly used in text mining where the data is represented as TF-IDF vectors. To conclude, as Manhattan distance is more stable to data contains outlier and the data actually include a categorical variable, I choose to use it as the metric for this question.

# %%
# Read the csv
iris = pd.read_csv("iris.csv")

# Identify x and y
x = iris[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
y = iris['variety']

# Split the data to training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 66)

# KNN
k = [1, 3, 5, 7]
for i in k:
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(x_train, y_train)
    predicted = knn.predict(x_test)
# Evaluate accuracy
    print(accuracy_score(predicted, y_test))
print("------------")
# Pick three kind of distance and provide the result
# Euclidean
k_optimal = 3
knn = KNeighborsClassifier(n_neighbors = k_optimal, metric = "euclidean")
knn.fit(x_train, y_train)
predicted = knn.predict(x_test)
print("Euclidean:", accuracy_score(predicted, y_test))

knn = KNeighborsClassifier(n_neighbors = k_optimal, metric = "manhattan")
knn.fit(x_train, y_train)
predicted = knn.predict(x_test)
print("Manhattan:", accuracy_score(predicted, y_test))

knn = KNeighborsClassifier(n_neighbors = k_optimal, metric = "cosine")
knn.fit(x_train, y_train)
predicted = knn.predict(x_test)
print("Cosine:", accuracy_score(predicted, y_test))



