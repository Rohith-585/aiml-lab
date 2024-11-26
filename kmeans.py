import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd
import numpy as np

# Load the Iris dataset
iris = datasets.load_iris()

# Convert data to DataFrame
X = pd.DataFrame(iris.data)
X.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_length', 'Petal_Width']
y = pd.DataFrame(iris.target)
y.columns = ['target']

# Set up the figure
plt.figure(figsize=(14, 7))
colormap = np.array(['red', 'lime', 'black'])

# Plot real classifications using Sepal features
plt.subplot(1, 2, 1)
plt.scatter(X.Sepal_Length, X.Sepal_Width, c=colormap[y.target], s=40)
plt.title('Sepal')

# Plot real classifications using Petal features
plt.subplot(1, 2, 2)
plt.scatter(X.Petal_length, X.Petal_Width, c=colormap[y.target], s=40)
plt.title('Petal')

# Apply K-Means clustering
model = KMeans(n_clusters=3)
model.fit(X)

# Print the predicted cluster labels
print(model.labels_)

# Plot real classification vs. K-Means classification
plt.subplot(1, 2, 1)
plt.scatter(X.Petal_length, X.Petal_Width, c=colormap[y.target], s=40)
plt.title('Real Classification')

plt.subplot(1, 2, 2)
plt.scatter(X.Petal_length, X.Petal_Width, c=colormap[model.labels_], s=40)
plt.title('KMEANS Classification')
