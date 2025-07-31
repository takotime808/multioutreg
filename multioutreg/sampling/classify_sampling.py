# # Copyright (c) 2025 takotime808

# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.neighbors import KDTree
# from sklearn.metrics import silhouette_score
# from scipy.stats import entropy
# import umap


# def classify_sampling_method(
#     X: np.ndarray | pd.DataFrame,
#     n_neighbors: int = 15,
#     min_dist: float = 0.1,
#     random_state: int = 0
# ) -> dict:
#     """
#     Classify the sampling method (Grid, Random, Sobol, LHS, or Uncertain) of an input dataset
#     using UMAP projection and structural heuristics.

#     Parameters
#     ----------
#     X : np.ndarray or pd.DataFrame
#         Input data. Must be numeric and shape (n_samples, n_features).
#     n_neighbors : int
#         UMAP neighborhood size.
#     min_dist : float
#         UMAP minimum distance.
#     random_state : int
#         Random seed for UMAP and KMeans.

#     Returns
#     -------
#     dict
#         A dictionary with the keys:
#         - 'method': str, one of 'Grid', 'Random', 'Sobol', 'LHS', 'Uncertain'
#         - 'explanation': str, rule-based reasoning
#         - 'std': float, std of neighbor distances
#         - 'silhouette': float
#         - 'entropy': float
#     """
#     if isinstance(X, pd.DataFrame):
#         X = X.select_dtypes(include=[np.number]).dropna().to_numpy()

#     if X.shape[0] < 10 or X.shape[1] < 1:
#         return {
#             "method": "Uncertain",
#             "explanation": "Insufficient data",
#             "std": np.nan,
#             "silhouette": np.nan,
#             "entropy": np.nan
#         }

#     # UMAP projection
#     X_scaled = StandardScaler().fit_transform(X)
#     reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=random_state)
#     X_emb = reducer.fit_transform(X_scaled)

#     # Clustering & structure metrics
#     kmeans = KMeans(n_clusters=2, n_init="auto", random_state=random_state)
#     labels = kmeans.fit_predict(X_emb)
#     dists = KDTree(X_emb).query(X_emb, k=2)[0][:, 1]
#     sil = silhouette_score(X_emb, labels)
#     std = np.std(dists)
#     ent = entropy(np.histogram(dists, bins=30, density=True)[0])

#     # Rule-based classification
#     if std < 0.05 and sil > 0.6:
#         method = "Grid"
#         explanation = "Low std and high silhouette -> Grid"
#     elif std > 0.2 and sil < 0.3:
#         method = "Random"
#         explanation = "High std and low silhouette -> Random"
#     elif 0.05 <= std <= 0.15:
#         if ent < 2.0:
#             method = "Sobol"
#             explanation = "Moderate spread, low entropy -> Sobol"
#         else:
#             method = "LHS"
#             explanation = "Moderate spread, higher entropy -> LHS"
#     else:
#         method = "Uncertain"
#         explanation = "Pattern unclear"

#     return {
#         "method": method,
#         "explanation": explanation,
#         "std": float(std),
#         "silhouette": float(sil),
#         "entropy": float(ent)
#     }
