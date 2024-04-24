# kmeans-- & Bayes Optimizatin
## KMeans-- K-Means minus minus Pytorch implementation with Bayesian Optimizatin to improve hyperparameter discovery and resistance to outliers in clustering tasks. 

| Refs:  |
| ------ |
| ["k-means--: A unified approach to clustering and outlier detection"](http://users.ics.aalto.fi/gionis/kmmm.pdf) |


K-Means-- is an extesion of k-means that performs simultaneously both clustering and outliers detection. It takes as input the number of clusters (k) and the number of outliers (l).

If CUDA is enabled, the model can run on GPU.


```python
from kmeansmm import KMeansMM
```

```python
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import torch

n_samples = 1000
X, y = make_blobs(n_samples=n_samples, random_state=1232)

y_pred = KMeansMM(n_clusters=3, l=2, max_iter=1000).fit_predict(torch.FloatTensor(X))

plt.scatter(X[:, 0], X[:, 1], c=y_pred.cpu())
```   

