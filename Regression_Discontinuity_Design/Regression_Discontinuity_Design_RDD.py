# %%
import pandas as pd
from matplotlib import pyplot as plt


# %% [markdown]
# - Use the drinking.csv dataset for this question. Keeping 21 as the threshold for age, explore the data with an RDD by writing very simple code (no package needed, just average to one side of the threshold minus average to the other side) to determine if alcohol increases the chances of death by accident, suicide and/or others (the three given columns) and comment on the question “Should the legal age for drinking be reduced from 21?” based on the results. Plot graphs to show the discontinuity (if any) and to show results for the change in chances of death with all the three features (i.e., accident vs age, suicide vs age and others vs age). For this problem, choose the bandwidth to be 1 year (i.e., 21 +- 1). What might be the effect of choosing a smaller bandwidth?  What if we chose the maximum bandwidth?
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



