"""
This module implements the KMeans-- algorithm with a bayesian optimiation
from https://scikit-optimize.github.io/stable/auto_examples/sklearn-gridsearchcv-replacement.html


"""

import torch
from bayes_opt import BayesianOptimization
from sklearn.metrics import silhouette_score

from .kmeansmm import KMeansMM


class KM3Opt:
    def __init__(self, X_train):
        self.X_train = X_train

    def evaluate_kmeans(self, n_clusters, l, max_iter, tol):
        model = KMeansMM(
            n_clusters=int(n_clusters), l=int(l), max_iter=int(max_iter), tol=float(tol)
        )
        model.fit(self.X_train)
        # Example evaluation metric: return negative silhouette score
        return -silhouette_score(self.X_train, model.predict(self.X_train))

    def optimize_hyperparameters(self, init_points=5, n_iter=10, **kwargs):
        # Define parameter space bounds
        pbounds = {
            "n_clusters": (2, 20),
            "l": (1, 5),
            "max_iter": (500, 2000),
            "tol": (0.0001, 0.01),
        }
        optimizer = BayesianOptimization(
            f=self.evaluate_kmeans, pbounds=pbounds, random_state=42, verbose=2
        )
        optimizer.maximize(init_points=init_points, n_iter=n_iter, **kwargs)

        return optimizer
