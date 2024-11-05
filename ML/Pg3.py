import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\\Users\Lenovo\Desktop\Study\Engineering\BE SEM 1\LP 3\ML\sales_data_sample.csv", encoding='ISO-8859-1')

print(df.head())

print(df.isna().sum())

#Selecting relevant features for clustering
features = df.select_dtypes(include=[np.number])

#Handling missing values
features.fillna(features.mean(), inplace=True)

#Feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

#Elbow method to determine the optimal number of clusters:
inertia=[]
K = range(1,11)
for k in K:
  kmeans = KMeans(n_clusters=k, random_state=42)
  kmeans.fit(scaled_features)
  inertia.append(kmeans.inertia_)

#Plotting the elbow graph
plt.plot(K, inertia, marker = 'o')
plt.title("Elbow method for optimal K")
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.xticks(K)
plt.grid()
plt.show()

optimal_k = 3
#Applying KMeans clustering
kmeans = KMeans(n_clusters = optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_features)

#Visualizing the clusters:
plt.scatter(scaled_features[:,0], scaled_features[:,1], c=df['Cluster'])
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid()
plt.show()