import pandas as pd
import json 
import matplotlib.pyplot as plt 
from sklearn.metrics import DistanceMetric

event_map = {}
event_features_map = {}
similiarity_map = {}

df = pd.read_csv("earthquake_data_tsunami.csv")

features = ['magnitude', 'depth', 'mmi']
df_features = df[features + ['tsunami']].dropna()

for index, row in df_features.iterrows():
    label = ""
    if row["tsunami"] == 1:
        label = "Tsunami"
    else:
        label = "No Tsunami"
    event_map[index] = label
    event_features_map[index] = row[features].to_dict()


target_event_index = df[df['tsunami'] == 1].index[0]
euclid = DistanceMetric.get_metric("euclidean")

for feature in features:
    print(f"\nFeature: {feature.upper()}")

    # Extract just this feature
    df_feature = df[[feature]].copy()
    
    index = list(event_map.keys())
    rows = [event_features_map[k] for k in index]
    df_feat = pd.DataFrame(rows, index=index).fillna(0).astype(float)

    row_sums = df_feature.sum(axis=1)
    safe_mask=row_sums>0
    df = df.loc[safe_mask]
    row_sums = row_sums.loc[safe_mask]
    df_norm = df.divide(row_sums, axis = 0)

    all_vecs = df_norm.to_numpy()
    target_vec = df_norm.loc[target_event_index].to_numpy().reshape(1, -1)
    distances = euclid.pairwise(all_vecs, target_vec)[:, 0]

    pairs = [(eid, dist) for eid, dist in zip(df_norm.index, distances) if eid != target_event_index]
    top10_euclid = sorted(pairs, key=lambda x: x[1])[:10]

    print("\nTop 10 most similar earthquakes based on Euclidean distance:")
    for eid, dist in top10_euclid:
        label = event_map[eid]
        print(f"Index: {eid} | Label: {label} | Distance: {dist:.4f}")

