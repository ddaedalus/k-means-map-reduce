from numpy import array, zeros, sin, cos, arctan2, sqrt, reshape, argmin, sum
from numpy import linalg as LA

class KMeans:
    '''
        KMeans classic clustering algorithm 
        as it is described in "Pattern Recognition and Machine Learning", Bishop, 2006.
    '''
    def __init__(self, K=5, centroids=None, maxiter=3, solver="euclidean"):
        '''
            K:           int (default 5), number of clusters
            centroids:   K X M array (default None), cluster prototypes
            maxiter:     int (default 3), maximum number of iterations
            solver:      string (default "euclidean"), distance metric
        '''
        self.K = K
        self.centroids = centroids  # prototypes
        self.maxiter = maxiter
        self.r = None    # labels
        self.solver = solver
        self.metric = None
        
        
    @staticmethod
    def hav(h):
        return (sin(h / 2))** 2
    
        
    def haversine(self, x1, x2):
        '''
            x1: (2,) array
            x2: (2,) array
        '''
        
        phi1 = x1[0]
        phi2 = x2[0]
        l1 = x1[1]
        l2 = x2[1]
        dphi = phi1 - phi2
        dl = l1 - l2
        R = 6371   # radius of earth (km)
        
        a = self.hav(dphi) + cos(phi1) * cos(phi2) * self.hav(dl)
        c = 2 * arctan2(sqrt(a), sqrt(1-a))
        
        # haversine distance
        d = R * c
        
        return d
    
    
    @staticmethod
    def euclidean(x1, x2):
        '''
            x1: array
            x2: array
        '''
        return LA.norm(x1 - x2)
    
    
        
    def fit(self, X):
        '''
            X: N x M array, data points to be clustered
        '''
        
        K = self.K
        
        N = X.shape[0] # num_samples
        M = X.shape[1] # num_features
        
        # Initialize centroids
        if self.centroids is None:
            self.centroids = X[0:K, :]
        
        if self.solver == "euclidean":
            self.metric = lambda x1,x2: self.euclidean(x1, x2)
            
        elif self.solver == "haversine":
            self.metric = lambda x1,x2: self.haversine(x1, x2)
        
        i = 0
        while (i < self.maxiter):
            # Expectation step
            R = zeros((N,K))   # distances with all centroids
            for n in range(N):
                R[n,:] = reshape([self.metric(X[n,:], self.centroids[k,:]) for k in range(K)], (1,K))
                
            self.r = reshape(argmin(R, axis=1), (1,N)) # (1,N)
            
            # Maximization step
            for k in range(K):
                self.centroids[k,:] = sum(X[self.r[0] == k], axis=0) / X[self.r[0] == k].shape[0]

            # while step
            i += 1
            
        