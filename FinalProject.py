from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Prepare dataset
house_df = pd.read_csv("RealEstate_California.csv")[["price", "yearBuilt", "buildingArea", "homeType", "levels"]].dropna()

house_map = {"SINGLE_FAMILY": 0, "LOT": 1, "CONDO": 2, "MULTI_FAMILY": 3, "TOWNHOUSE": 4}
house_df["homeType"] = house_df["homeType"].map(house_map)

level_map = {"0": 0, "One": 1, "Two": 2, "Three Or More": 3, "One Story": 4}
house_df["levels"] = house_df["levels"].map(level_map)

scaler = MinMaxScaler()

normal_house_df = pd.DataFrame(scaler.fit_transform(house_df), columns=house_df.columns).dropna()

# Elbow method to determine number of clusters
elbow = []
for k in range(1, 20):
    kmeans = KMeans(n_clusters = k, init = 'k-means++').fit(normal_house_df)
    elbow.append(kmeans.inertia_)
    
# optimal number is 5
plt.plot(range(1, 20), elbow)
plt.show()

# using k groups, executing kmeans
house_kmeans = KMeans(n_clusters = 5, init = 'k-means++').fit(normal_house_df)
normal_house_df.insert(0, "id", pd.read_csv("RealEstate_California.csv")[["id"]])
normal_house_df.insert(1, "streetAddress", pd.read_csv("RealEstate_California.csv")[["streetAddress"]])
normal_house_df.insert(2, "city", pd.read_csv("RealEstate_California.csv")[["city"]])
normal_house_df.insert(3, "county", pd.read_csv("RealEstate_California.csv")[["county"]])
normal_house_df['Cluster'] = house_kmeans.labels_
normal_house_df.sort_values(by='Cluster').to_csv('output.csv', index=False)