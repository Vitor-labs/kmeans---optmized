import torch

class KMeansMM:
    def __init__(self, n_clusters=10, l=2, max_iter=1000, tol=0.0001, device = 'cuda'):
        self.k = n_clusters
        self.l = l
        self.device = torch.device(device)
        self.centroids = None
        self.max_iter = max_iter
        self.tol = tol
        
    def euclidean(self,X,centroids):
        return ((X[:,None,:] - centroids[None,:,:]) ** 2).sum(dim=-1)
    
    def fit(self, X):
        X = X.to(self.device)
        #initialize centroids as random points from X
        perm = torch.randperm(X.shape[0])
        idx = perm[:self.k]
        centroids = X[idx,:]
        previous_state = centroids.clone()
        
        for i in range(self.max_iter):
            #Compute distances between points and centroids
            distances = self.euclidean(X,centroids)
            
            #Select closest centroid for each point
            mins = distances.min(dim=1)
            labels = mins[1] #argmin
            mindists = mins[0] #min
            
            L = torch.topk(mindists,k=self.l)[1] #take l points with largest distance from centroids
            labels[L] = -1 #remove outliers from the assigned cluster (assign special cluster/label -1)
            
            #Compute new centroids
            for cl in range(self.k):
                centroids[cl] = X[labels == cl,:].mean(dim=0)
            
            #Check for convergence
            centroids_shift = ((previous_state - centroids) ** 2).sum(dim=1).sqrt().sum()
            if centroids_shift < self.tol:
                break
        
        self.centroids = centroids
        
        return self

    def predict(self, X):
        X = X.to(self.device)
        distances = self.euclidean(X,self.centroids)
        mins = distances.min(dim=1)
        labels = mins[1] #argmin
        mindists = mins[0] #min

        L = torch.topk(mindists,k=self.l)[1] #take l points with largest distance from centroids
        labels[L] = -1 #remove outliers from the assigned cluster
        return labels
    
    def fit_predict(self, X):
        X = X.to(self.device)
        self.fit(X)
        return self.predict(X)