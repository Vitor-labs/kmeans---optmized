# kmeans-- for new cluster detection
## K-Means minus minus Pytorch implementation with custom cluster metrics n to improve hyperparameter discovery and resistance to outliers in clustering tasks to detect new clusters over data distribution drift. 

| Refs:                                                                                                            |
| ---------------------------------------------------------------------------------------------------------------- |
| ["k-means--: A unified approach to clustering and outlier detection"](http://users.ics.aalto.fi/gionis/kmmm.pdf) |
| ["Pytorch Silhuete Score"](https://github.com/maxschelski/pytorch-cluster-metrics)                               |

* Construção da base de dados experimental.
* Definição dos hiperparâmetros que serão otimizados.
* Implementação da otimização bayesiana dos hiperparâmetros do kmeans--.
* Análise dos resultados.

K-Means-- is an extesion of k-means that performs simultaneously both clustering and outliers detection. It takes as input the number of clusters (k) and the number of outliers (l).

If CUDA is enabled, the model can run on GPU.

```python
from kmeansmm import KMeansMM
```

```python
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import torch

X, y = make_blobs(n_samples=1000, random_state=42)

y_pred = KMeansMM(n_clusters=3, l=2, max_iter=1000).fit_predict(torch.FloatTensor(X))

plt.scatter(X[:, 0], X[:, 1], c=y_pred.cpu())
```   

| Features to add:          |
| ------------------------- |
| Stop condition to train   |
| Hyper-params to add       |
| Custom metric to evaluate |
