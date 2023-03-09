# %%
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# %% [markdown]
# 1. Before performing any dimensionality reduction, write a program to use k-means clustering on the Madelon dataset. Try the following k values: 4, 8, 16, 32, 64. 

# %%
# Read in the data
data = pd.read_csv("madelon.csv", index_col = False)

# Check if there is na
data.isnull().values.any()

# Scale the data and convert to dataframe
data_scaled = StandardScaler().fit_transform(data.iloc[:, 1:501])
data_scaled = pd.DataFrame(data_scaled)

# K-means
k = [4, 8, 16, 32, 64]
sse = {}

for i in k:
    kmeans = KMeans(n_clusters = i, random_state = 10)
    y = kmeans.fit_predict(data_scaled)
    sse[i] = kmeans.inertia_
    data['Cluster_' + str(i)] = y.tolist()


# %% [markdown]
# 1-1. What preprocessing techniques did you apply, if any?
# 
# Since the units of features may differ from each other, I apply StandardScaler to scale the variables. It standardizes features by removing the mean and scale them to unit variance.

# %% [markdown]
# 1-2. Describe qualitatively: how does the number of clusters affect the performance of the clustering algorithm on the dataset? 
# 
# If the number of clusters (k) is too small, the algorithm has a higher risk of not capturing the underlying structure of data, which leads to a poor performance of clustering. On the contrary, if k is too large, the algorithm may overfit the data and generate too many meaningless clusters. And it indicates a poor performance as well. Therefore, choosing an optimal k that can keep a good balance in betweeen is important while performing k-means clustering.

# %% [markdown]
# 1-3. Generate a plot of the number of clusters k (x-axis) versus the sum of squared distance (SSE) between data points and their assigned centroids (y-axis). What appears to be the optimal k from the list of values you used, and why? 
# 
# According to elbow rule, the optimal k value appears to be 8. Although the angle is not very clear on the plot, we can see the decrease in within-cluster sum of square(WCSS) becomes smaller when k goes above 8. 

# %%
# Generate a plot
plt.figure()
plt.plot(k, list(sse.values()))
plt.xticks(k, k)
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

# %% [markdown]
# 1-4. For k = 8, how did you initialize the set of cluster centroids before running k-means? Rerun k-means again, this time ensuring that the centroids are all different. Does this – and if so, to what extent – affect the final clusters created?
# 
# I initialize the centroids randomly before running k-means. By randomly assigning the starting position, we can ensure that the algorithm converges to a global optimum. For example, if the initial centroids are chosen based on a specific sub-pattern in the data, k-means may generate a result following that pattern, which leads to a poor performance in clustering. We can also observe significant improvement in the plot, the angles of line are not only more clear but also indicates that 16 should be the optimal k instead of 8.

# %%
# Rerun K-means
k = [4, 8, 16, 32, 64]
sse = {}
for i in k:
    kmeans = KMeans(n_clusters = i, init = "random", random_state = 10)
    y = kmeans.fit_predict(data_scaled)
    sse[i] = kmeans.inertia_
    data_scaled['Cluster_' + str(i)] = y.tolist()

# Generate a plot
plt.figure()
plt.plot(k, list(sse.values()))
plt.xticks(k, k)
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

# %% [markdown]
# 1-5. More generally for any dataset, what evaluation metrics can be used to assess the quality of the clusters produced?
# 
# As we demonstrate in previous questions, within-cluster sum of squares is a common metric to measure the quality of the clusters produced, the lower the better. Silhouette score is another metric that measures the similarity of each point to its own cluster compared to other clusters, a higher score indicates the datapoints are well-separated. There is also Calinski-Harabasz Index. This index measures the ratio of the between-cluster variance to the within-cluster variance, a higher index value symbolizes better clustering results.

# %% [markdown]
# 2-1. Fit the standardized data with PCA. Then, create a cumulative variance plot – showing the number of components included (x-axis) versus the amount of variance captured (y-axis). Generally, we want to retain at least 75% of the variance. How many components would you decide to keep?
# 
# According to the screen print, 281 components should be kept.

# %%
# Scale the data and convert to dataframe
data = pd.read_csv("madelon.csv", index_col = False)
data_scaled = StandardScaler().fit_transform(data.iloc[:, 1:501])
data_scaled = pd.DataFrame(data_scaled)

# PCA
x = list(range(1, 501))
cum_var = {}
for i in range(1, 501):
    pca = PCA(n_components=i, random_state = 10)
    pca_result = pca.fit_transform(data_scaled)
    cum_var[i] = np.sum(pca.explained_variance_ratio_)

# Generate a cumulative variance plot
plt.figure()
plt.plot(x, list(cum_var.values()))
plt.xticks(x, x)
plt.xlabel("PCA")
plt.ylabel("Variance")
plt.show()

# Keep at least 75% variance
result = next(i for i, value in enumerate(list(cum_var.values()))
if value > 0.75)
print("Number of components to keep: ", result)

# %% [markdown]
# 2-2. Perform PCA with your selected principal components. 
# 
# (i) Plot the transformed data on a graph with the first two principal components as the axes i.e. x = PC 1, y = PC 2.
# 
# (ii) Plot the original data on a graph with the two original variables that have the highest absolute combined loading for PC 1 and PC 2 i.e. maximizing |loading PC1| + |loading PC2|.

# %%
# PCA with selected features
pca = PCA(n_components = 281, random_state = 30)
pca_result = pca.fit_transform(data_scaled)

# Plot the transformed data on a graph with the first two principal components
plot = plt.scatter(pca_result[:,0], pca_result[:,1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("First two principal components after scaling")
plt.show()

# Plot the original data with the two variables that have the highest absolute combined loading
loadings = pd.DataFrame(pca.components_.T, columns=['PC'+ str(i+1) for i in range(281)])
loadings['combined_loading'] = np.abs(loadings).sum(axis=1)

# Get the indices of the two variables with the highest absolute combined loadings
id1, id2 = loadings['combined_loading'].nlargest(2).index

# Plot the original data with variables
plt.scatter(data["V" + str(id1+1)], data["V" + str(id2+1)])
plt.xlabel('V' + str(id1+1))
plt.ylabel('V'+ str(id2+1))
plt.title("Original data with the two selected variables with highest absolute loading")
plt.show()

# %% [markdown]
# 2-3. Examine the scatter plot of PC 1 (x-axis) versus PC 2 (y-axis) for all data points that you created in the previous part. Qualitatively, can you identify visible clusters? Why or why not might this be the case with this particular dataset?
# 
# No, I can not identify visible clusters. That may because that there are still too many features included after performing PCA.

# %% [markdown]
# 3-1. Now, we will run k-means clustering on the transformed data from the previous problem. Why is it a good idea to do this, especially for high-dimensional datasets? Name a number of reasons.
# 
# The first reason is dimensionality reduction. As too many dimensions result in a higher risk of overfitting the data, PCA helps elimating unnecessary dimensions while preserving most of the variance. The second reason is higher interpretability. Instead of getting confused by hundreds of feature in original data, PCA provides insights into which of them are most important for clustering. The third reason is speed/cost. Perform k-means clustering in high dimensional data is quite expensive, and PCA can significantly reduces the cost and time to run the algorithm.

# %% [markdown]
# 3-2. Use the same k values again (4, 8, 16, 32, 64) to again generate an elbow plot. What is the optimal k? Is it different from the one you found in (1)? Compare the SSE values plotted in this exercise to the previous plot you generated in (1c) before performing PCA.
# 
# The optimal k should be 16. It is different from what I found in (1). The SSE of k-means after performing PCA is significantly smaller than (1) since PCA helps us do dimension reduction by removing irrelevant features. Also, PCA identifies the directions of greatest variance in the data and projects the data onto these directions. This can help to amplify the differences between data points that belong to different clusters, and reduce the variance within clusters. Therefore, it will be easier for k-means algorithm to find distinct clusters.

# %%
# K-means
k = [4, 8, 16, 32, 64]
sse = {}
for i in k:
    kmeans = KMeans(n_clusters = i, random_state = 30)
    y = kmeans.fit_predict(pca_result)
    sse[i] = kmeans.inertia_
    data_scaled['Cluster_' + str(i)] = y.tolist()

# Generate a plot
plt.figure()
plt.plot(k, list(sse.values()))
plt.xticks(k, k)
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

# %% [markdown]
# 3-3. Again, create a scatter plot of PC 1 (x-axis) versus PC 2 (y-axis) for all of the transformed data points. Label the cluster centers and color-code by cluster assignment for the first 5 iterations of k = 32. Can you see the algorithm begin to converge to optimal assignments?
# 
# I can roughly see the algorithm begins to converge to optimal assignments as the iteration goes, although it's not so obvious.

# %%
for i in range(5):
    kmeans = KMeans(n_clusters=32, max_iter = i + 1)
    kmeans.fit(pca_result)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Plot the transformed data with color-coded cluster assignments and labeled centers
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c = labels)
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', color='r')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title("Iteration " + str(i+1))
    plt.show()


