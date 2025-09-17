import pandas as pd

# -----------------------------
# Day 11 – Load & Explore Data
# -----------------------------
df = pd.read_csv("Mall_Customers.csv")
df.head()

df.info()
df.describe()
df.isnull().sum()

import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['Age'], bins=20, kde=True)
# plt.title("Age Distribution")
# plt.show()

sns.histplot(df['Annual Income (k$)'], bins=20, kde=True)
# plt.title("Annual Income Distribution")
# plt.show()

sns.histplot(df['Spending Score (1-100)'], bins=20, kde=True)
# plt.title("Spending Score Distribution")
# plt.show()

sns.countplot(x='Gender', data=df)
# plt.title("Gender Count")
# plt.show()

from sklearn.preprocessing import StandardScaler

features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
