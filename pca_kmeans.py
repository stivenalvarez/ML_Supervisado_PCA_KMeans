# Implementación PCA + K-Means para Iris y Wine
# Autor: [Tu Nombre]
# Proyecto de ML Supervisado - Reducción de Dimensionalidad

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ============================
# Dataset 1: IRIS
# ============================

iris = load_iris()
X = iris.data

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# K-Means y Silhouette
ks = [2, 4, 5]
print("Resultados IRIS")
for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    print(f"k={k}, silhouette={score:.4f}")

# ============================
# Dataset 2: WINE
# ============================

wine = load_wine()
X2 = wine.data

# Escalado
scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(X2)

# PCA
pca2 = PCA(n_components=2)
X2_pca = pca2.fit_transform(X2_scaled)

# K-Means y Silhouette
print("\nResultados WINE")
for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels2 = kmeans.fit_predict(X2_pca)
    score2 = silhouette_score(X2_pca, labels2)
    print(f"k={k}, silhouette={score2:.4f}")
