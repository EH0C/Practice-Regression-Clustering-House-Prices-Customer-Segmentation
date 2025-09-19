import pandas as pd

# -----------------------------
# Day 11 – Load & Explore Data
# -----------------------------

# Load dataset
df = pd.read_csv("Mall_Customers.csv")
df.head()

# Explore features
df.info()
df.describe()
df.isnull().sum()

# Feature exploration
import seaborn as sns
import matplotlib.pyplot as plt

# sns.histplot(df['Age'], bins=20, kde=True)
# # plt.title("Age Distribution")
# # plt.show()

# sns.histplot(df['Annual Income (k$)'], bins=20, kde=True)
# # plt.title("Annual Income Distribution")
# # plt.show()

# sns.histplot(df['Spending Score (1-100)'], bins=20, kde=True)
# # plt.title("Spending Score Distribution")
# # plt.show()

# sns.countplot(x='Gender', data=df)
# # plt.title("Gender Count")
# # plt.show()

# cale data (prepare for clustering)
from sklearn.preprocessing import StandardScaler

features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# -----------------------------
# Day 12 – K-Means & Elbow Method
# -----------------------------

# Apply K-Means
from sklearn.cluster import KMeans

inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method
# plt.plot(K, inertia, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Inertia')
# plt.title('Elbow Method')
# plt.show()

# Visualize clusters
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
sns.scatterplot(
    x=df['Annual Income (k$)'], 
    y=df['Spending Score (1-100)'], 
    hue=df['Cluster'], palette='tab10'
)
plt.show()