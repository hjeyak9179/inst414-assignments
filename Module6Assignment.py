import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Compile fields for analysis
real_estate_df = pd.read_csv("RealEstate_California.csv")[["id", "streetAddress", "city", "buildingArea", "price"]].dropna()

# Defining features and prediction values
features = real_estate_df[["city", "buildingArea"]]
features = pd.get_dummies(features, columns=["city"], drop_first=True)

prediction = real_estate_df["price"]

x_train, x_test, y_train, y_test = train_test_split(features, prediction, test_size = 0.3, random_state = 100)

# Develop model
model = LinearRegression().fit(x_train, y_train)

y_pred = model.predict(x_test)

errors = abs(y_pred - y_test)
worst_indices = np.argsort(errors)[-5:]

worst_samples = real_estate_df.iloc[y_test.index[worst_indices]]
worst_samples["predicted_price"] = y_pred[worst_indices]
worst_samples["absolute_error"] = errors.iloc[worst_indices]

print("\nWorst 5 prediction samples:")
print(worst_samples)






