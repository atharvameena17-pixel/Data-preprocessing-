from sklearn.datasets import make_classification
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

dimensions = [2, 10, 50, 200]

plt.figure(figsize=(12, 8))

for dim in dimensions:
    X, _ = make_classification(
        n_samples=500,
        n_features=dim,
        random_state=42
    )
    
    distances = pdist(X, metric='euclidean')
    
    plt.hist(distances, bins=30, alpha=0.5, label=f"{dim}D")

plt.legend()
plt.title("Distance Distribution vs Dimensions")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.show()