import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

iris = load_iris()

df = pd.DataFrame(data = iris.data, columns = iris.feature_names)

df['target'] = iris.target

print("Dataset Preview:")
print(df.head())
print(df.columns)

x = df.drop("target", axis = 1)
y = df["target"]

scalar = StandardScaler()

X_scaled = scalar.fit_transform(x)

print("\nStandardized Data Preview:")
print(X_scaled[:5])

pca = PCA(n_components=None)

X_pca = pca.fit_transform(X_scaled)

pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
df_pca = pd.DataFrame(data = X_pca, columns = pca_columns)

print("\nPrincipal Components:")
print(df_pca.head())

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, 5), pca.explained_variance_ratio_, 'ro-', linewidth=2)

# Adding labels and titles
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.xticks(range(1, 5)) # Ensure x-axis shows 1, 2, 3, 4
plt.grid(True)
plt.show()

# Print the exact variance values
print("\nExplained Variance Ratio per component:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio:.4f}")