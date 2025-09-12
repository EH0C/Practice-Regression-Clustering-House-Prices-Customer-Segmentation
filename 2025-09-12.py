# -----------------------------
# Day 1: Load House Prices & Handle Missing Values
# -----------------------------
import pandas as pd
import numpy as np

# Load dataset (Kaggle House Prices or any similar CSV)
df = pd.read_csv("Housing.csv")

# Explore
print(df.head())
print(df.info())
print(df.describe())

# Target
target = "price"

# Handle missing values
# Numeric → fill with median
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Categorical → fill with mode
cat_cols = df.select_dtypes(include="object").columns
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# -----------------------------
# Day 2: Preprocess & Linear Regression
# -----------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Select features (drop ID and target)
X = df.drop(columns=["Id", target], errors="ignore")
y = df[target]

# One-hot encode categorical
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predictions
y_pred = lr.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")
