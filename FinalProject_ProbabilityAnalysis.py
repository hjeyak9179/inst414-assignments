from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

house_df = pd.read_csv("RealEstate_California.csv")[[
    "price",
    "yearBuilt",
    "buildingArea",
    "homeType",
    "levels",
    "city",
    "county"
]].dropna()

house_map = {
    "SINGLE_FAMILY": 0,
    "LOT": 1,
    "CONDO": 2,
    "MULTI_FAMILY": 3,
    "TOWNHOUSE": 4
}

level_map = {
    "0": 0,
    "One": 1,
    "Two": 2,
    "Three Or More": 3,
    "One Story": 4
}

house_df["homeType"] = house_df["homeType"].map(house_map)
house_df["levels"] = house_df["levels"].map(level_map)

house_df = house_df.dropna()

house_df["price_category"] = pd.qcut(
    house_df["price"],
    q=3,
    labels=["Low", "Medium", "High"]
)

print("Price Category Distribution:")
print(house_df["price_category"].value_counts(normalize=True))

features = ["yearBuilt", "buildingArea", "homeType", "levels"]

scaler = MinMaxScaler()
X = scaler.fit_transform(house_df[features])
y = house_df["price_category"]

indices = np.arange(len(house_df))

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X,
    y,
    indices,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

probabilities = model.predict_proba(X_test)

prob_df = pd.DataFrame(
    probabilities,
    columns=model.classes_
)

results = house_df.iloc[idx_test].copy()

results["Prob_Low"] = prob_df["Low"].values
results["Prob_Medium"] = prob_df["Medium"].values
results["Prob_High"] = prob_df["High"].values

city_probabilities = (
    results
    .groupby("city")[["Prob_High"]]
    .mean()
    .sort_values(by="Prob_High", ascending=False)
)

print("\nTop Cities by Probability of High-Priced Listings:")
print(city_probabilities.head(10))

results.to_csv("probability_output.csv", index=False)

print("\nProbability analysis complete. Output saved as 'probability_output.csv'.")

results.groupby("price_category")[["Prob_Low", "Prob_Medium", "Prob_High"]].mean().plot.bar()
plt.title("Average Predicted Price Probabilities by Category")
plt.show()