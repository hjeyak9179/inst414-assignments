from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Prepare dataset by mapping values and normalizing 
plants_df = pd.read_csv("Indoor_Plant_Health_and_Growth_Factors.csv")[["Height_cm", "New_Growth_Count", "Pest_Severity"]].dropna()

severity_map = {"None": 3, "Low": 2, "Moderate": 1, "High": 0}
plants_df["Pest_Severity"] = plants_df["Pest_Severity"].map(severity_map)

scaler = MinMaxScaler()

normal_plants_df = pd.DataFrame(scaler.fit_transform(plants_df), columns=plants_df.columns)

# Elbow method to determine number of clusters
elbow = []
for k in range(1, 20):
    kmeans = KMeans(n_clusters = k, init = 'k-means++').fit(normal_plants_df)
    elbow.append(kmeans.inertia_)
    
plt.plot(range(1, 20), elbow)
# shows that the ideal k is 5

# using k groups, executing kmeans
plant_kmeans = KMeans(n_clusters = 5, init = 'k-means++').fit(normal_plants_df)
normal_plants_df.insert(0, "Plant_ID", pd.read_csv("Indoor_Plant_Health_and_Growth_Factors.csv")[["Plant_ID"]])
normal_plants_df['Cluster'] = plant_kmeans.labels_
normal_plants_df.sort_values(by='Cluster').to_csv('output.csv', index=False)