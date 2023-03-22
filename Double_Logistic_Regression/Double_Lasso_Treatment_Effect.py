# %%
import pandas as pd
import numpy as np
import statsmodels.api as sm

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


