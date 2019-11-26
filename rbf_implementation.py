# Based on http://www.rueckstiess.net/research/snippets/show/72d2363e implementation

import scipy
import sklearn.gaussian_process.kernels
import sklearn.cluster


def gaussian_kernel(x, center, s):
    return np.exp(-1/2*s**2) * (x-center)**2)
    
class RBF():
    def __init__(self):
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.number_of_centers = number_of_centers
        self.beta = 8
        
        self.weights = scipy.random.random((self.number_of_centers, self.number_of_outputs))        
        
        self.centers = None
        self.functions = None

    def fit(self, X, y):
        # 1) Find centers
        k_means = cluster.Kmeans(n_clusters = self.number_of_centers)
        k_means.fit(X)
        k_centers = k_means.cluster_centers_.squeeze()
        
        # 2) Get centers and dispersions
        
        # 3) Define the centroids and the kernels
        
    def predict(self, X, y):
        pass()
    