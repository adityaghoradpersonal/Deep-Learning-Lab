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