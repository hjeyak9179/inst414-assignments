import pandas as pd

# Dataset with data about people from different countries and how much
# coffee they drink
df = pd.read_csv("synthetic_coffee_health_10000.csv")

# Want to find the average caffeine intake by country in mg
avg_by_country = df.groupby('Country')['Caffeine_mg'].mean()

country_df = pd.DataFrame(avg_by_country)

# Caffeine intake sorted from most to least
print(country_df.sort_values(by="Caffeine_mg", ascending=False).to_string())
