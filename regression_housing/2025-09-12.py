# -----------------------------
# Day 1: Load House Prices & Handle Missing Values
# -----------------------------
import pandas as pd
import numpy as np

# Load dataset (Kaggle House Prices or any similar CSV)
df = pd.read_csv("Housing.csv")

# Explore
# print(df.head())
# print(df.info())
# print(df.describe())

# Target
target = "price"

# Handle missing values
# Count total rows
total_rows = len(df)
# Count missing values per column
null_counts = df.isnull().sum()
# Compare with total rows (percentage of missing values)
null_percentage = (null_counts / total_rows) * 100
# Combine into one DataFrame for clarity
missing_report = pd.DataFrame({
    "Missing Values": null_counts,
    "Total Rows": total_rows,
    "Percentage (%)": null_percentage.round(2)
})
# print(missing_report)

# # Numeric → fill with median
# num_cols = df.select_dtypes(include=np.number).columns
# df[num_cols] = df[num_cols].fillna(df[num_cols].median())
# # Categorical → fill with mode
# cat_cols = df.select_dtypes(include="object").columns
# df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# -----------------------------
# Day 2: Preprocess & Linear Regression
# -----------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# -----------------------------
# 1. Features & Target
# -----------------------------
X = df.drop(columns=[target], errors="ignore")
y = df[target]

# One-hot encode categorical
X = pd.get_dummies(X, drop_first=True)

# -----------------------------
# 2. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Scale numeric features
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# -----------------------------
# 4. Multicollinearity check (VIF) with auto-drop
# -----------------------------
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]
    return vif_data.sort_values(by="VIF", ascending=False)

def drop_high_vif(X, threshold=10):
    dropped = []
    while True:
        vif_df = calculate_vif(X)
        max_vif = vif_df["VIF"].max()
        if max_vif > threshold:
            drop_feature = vif_df.loc[vif_df["VIF"].idxmax(), "Feature"]
            print(f"Dropping '{drop_feature}' (VIF={max_vif:.2f})")
            dropped.append(drop_feature)
            X = X.drop(columns=[drop_feature])
        else:
            break
    return X, vif_df, dropped

X_train_vif, final_vif, dropped_features = drop_high_vif(X_train_scaled, threshold=10)
X_test_vif = X_test_scaled[X_train_vif.columns]  # keep same features

# # -----------------------------
# # 5. Show feature summary
# # -----------------------------
print("\n=== Dropped Features (VIF > 10) ===")
print(dropped_features if dropped_features else "None")

print("\n=== Final Features Kept ===")
print(list(X_train_vif.columns))

print("\n=== Final VIF values ===")
print(final_vif)

# # -----------------------------
# # 6. Train Linear Regression
# # -----------------------------
lr = LinearRegression()
lr.fit(X_train_vif, y_train)

# # -----------------------------
# # 7. Predictions & Evaluation
# # -----------------------------
y_pred = lr.predict(X_test_vif)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n=== Model Performance ===")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# # # -----------------------------
# # # 8 OLS Regression Summary
# # # -----------------------------
# import statsmodels.api as sm

# # Assume df is your DataFrame
# X = df[["area", "bedrooms", "bathrooms", "stories", "parking"]]
# y = df["price"]

# # Add intercept
# X = sm.add_constant(X)

# # Fit model
# model = sm.OLS(y, X).fit()

# # Summary
# print(model.summary())

# -----------------------------
# Day 3: Random Forest Regression + Feature Importance
# -----------------------------
y_pred_lr = lr.predict(X_test_vif)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_test, y_pred_lr)


from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_vif, y_train)
y_pred_rf = rf.predict(X_test_vif)

mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# =============================
# Model Comparison
# =============================
results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MSE": [mse_lr, mse_rf],
    "RMSE": [rmse_lr, rmse_rf],
    "R²": [r2_lr, r2_rf]
})
print("\n=== Model Comparison ===")
print(results)

# Feature importance (Random Forest)
importances = rf.feature_importances_
indices = np.argsort(importances)[-10:]  # Top 10
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [X_train_vif.columns[i] for i in indices])
plt.xlabel("Feature Importance")
plt.title("Top 10 Important Features (Random Forest)")
plt.show()