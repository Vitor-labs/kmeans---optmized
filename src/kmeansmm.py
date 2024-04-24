"""
This module implements the KMeans-- algorithm with torch.
Based on: "k-means--: A unified approach to clustering and outlier detection"
http://users.ics.aalto.fi/gionis/kmmm.pdf by Sanjay Chawla and Aristides Gionis.

K-Means-- is an extesion of k-means that performs simultaneously both clustering
and outliers detection. It takes as input the number of clusters (k) and the
number of outliers (l).
"""

from typing_extensions import Self

import torch


class KMeansMM:
    """
    KMeans-- clustering with torch.

    Parameters
    ----------
    n_clusters: int
        number of clusters
    l: int
        number of outliers
    max_iter: int
        maximum number of iterations
    tol: float
        tolerance
    device: str
        'cpu' or 'cuda'
    """

    def __init__(
        self,
        n_clusters: int = 10,
        l: int = 2,
        max_iter: int = 1000,
        tol: float = 0.0001,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.k = n_clusters
        self.l = l
        self.device = torch.device(device)
        self.centroids = None
        self.max_iter = max_iter
        self.tol = tol

    def euclidean(self, X: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Euclidean distance between each point in X and each centroid
        in centroids.

        Parameters:
            X (torch.Tensor): Input tensor of shape (n_samples, n_features).
            centroids (torch.Tensor): Tensor of centroids of shape (n_centroids, n_features).

        Returns:
            torch.Tensor: The tensor of shape (n_samples, n_centroids) containing
            the Euclidean distances.
        """
        return ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(dim=-1)

    def fit(self, X: torch.Tensor) -> Self:
        """
        Fits the model to the input data.

        Parameters:
            X (torch.Tensor): The input data of shape (n_samples, n_features).

        Returns:
            Self: The fitted model.
        """
        X = X.to(self.device)
        # initialize centroids as random points from X
        perm = torch.randperm(X.shape[0])
        idx = perm[: self.k]
        centroids = X[idx, :]
        previous_state = centroids.clone()

        for _ in range(self.max_iter):
            # Compute distances between points and centroids
            distances = self.euclidean(X, centroids)

            # Select closest centroid for each point
            mins = distances.min(dim=1)
            labels = mins[1]  # argmin
            mindists = mins[0]  # min
            # take l points with largest distance from centroids
            L = torch.topk(mindists, k=self.l)[1]
            # remove outliers from the assigned cluster (assign special cluster/label -1)
            labels[L] = -1
            # Compute new centroids
            for cl in range(self.k):
                centroids[cl] = X[labels == cl, :].mean(dim=0)
            # Check for convergence
            centroids_shift = (
                ((previous_state - centroids) ** 2).sum(dim=1).sqrt().sum()
            )
            if centroids_shift < self.tol:
                break

        self.centroids = centroids

        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predicts the labels for the input data based on the fitted model.

        Parameters:
            X (torch.Tensor): The input data of shape (n_samples, n_features).

        Returns:
            torch.Tensor: The predicted labels for the input data.
        """
        X = X.to(self.device)
        assert self.centroids is not None, "Model must be fitted before predicting"

        distances = self.euclidean(X, self.centroids)
        mins = distances.min(dim=1)
        labels = mins[1]  # argmin
        mindists = mins[0]  # min

        L = torch.topk(mindists, k=self.l)[
            1
        ]  # take l points with largest distance from centroids
        labels[L] = -1  # remove outliers from the assigned cluster
        return labels

    def fit_predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Fits the model to the input data and then predicts the labels for the input data.

        Parameters:
            X (torch.Tensor): The input data of shape (n_samples, n_features).

        Returns:
            torch.Tensor: The predicted labels for the input data.
        """
        X = X.to(self.device)
        self.fit(X)
        return self.predict(X)
