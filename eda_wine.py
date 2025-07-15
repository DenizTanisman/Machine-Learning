import pandas as pd
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# 1. Read Data
df = pd.read_csv('data/winemag-data-130k-v2.csv', index_col=0)

# 2. General Overview
print("📌 First 5 Rows:")
print(df.head())

# 3. Basic Info
print("\n📌 Dataset Info:")
print(df.info())

print("\n📌 Column Names:")
print(df.columns)

# 4. Separate Numeric and Categorical Columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])
object_df = df.select_dtypes(include=['object'])

print("\n📌 Numeric Summary:")
print(numeric_df.describe().T)

print("\n📌 Categorical Summary:")
print(object_df.describe().T)

# 5. Missing Value Analysis
print("\n📌 Missing Values in Numeric Columns:")
print(numeric_df.isnull().sum())

print("\n📌 Percentage of Missing Values in Categorical Columns:")
print((object_df.isnull().sum() / len(df)) * 100)

# 6. Fill Missing Values
price_median = df['price'].median()
df['price'].fillna(price_median, inplace=True)

object_cols = [col for col in df.columns if df[col].dtype == 'object' and df[col].isnull().any()]
for col in object_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# 7. Filtering Examples
print("\n📌 Countries excluding Italy and France:")
not_europe = df[~df['country'].isin(['Italy', 'France'])]
print(not_europe['country'].unique())

print("\n📌 Wines from Australia or New Zealand with points >= 95:")
print(df[(df['country'].isin(['Australia', 'New Zealand'])) & (df['points'] >= 95)])

# 8. Map & Apply Example
points_mean = df['points'].mean()
df['points_centered'] = df['points'].map(lambda p: p - points_mean)

df[['points', 'price']] = df[['points', 'price']].apply(lambda col: col - points_mean)

# 9. Rename Columns for Clarity
df.rename(columns={
    'points': 'scores',
    'region_1': 'normal_address',
    'region_2': 'detail_address'
}, inplace=True)

# 10. Show First 50 Descriptions
print("\n📌 First 50 Descriptions:")
print(df['description'].iloc[:50])

# Environment and running the project
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python eda_wine.py
