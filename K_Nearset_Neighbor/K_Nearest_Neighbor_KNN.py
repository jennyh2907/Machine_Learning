# %%
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %% [markdown]
# - Use the iris.csv dataset for this question. How does the performance of k-nearest neighbors change as k takes on the following values: 1, 3, 5, 7? Which of these is the optimal value of k? Which distance/similarity metric did you choose to use and why?
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



