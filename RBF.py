import pandas as pd
import numpy as np

import sklearn.cluster
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances

import sklearn.metrics
from sklearn.linear_model import LogisticRegression


class RBFClassifier():
    """TESTE"""
    def __init__(self, number_of_centers, algorithm, random_state=42):
        """[summary]
        
        Arguments:
            number_of_centers {integer} -- The number of the centers
        
        Keyword Arguments:
            algorithm {sklearn classification algorithm} -- When None, it will create a LogisticRegression, without balanced class_weights. 
            random_state {int} -- Random state, to be used with the internal algorithms (default: {42})
        """        

        self.number_of_centers = number_of_centers
        self.centers = None
        self.centers_std = None
        self.kernels = None
        self.random_state = random_state
        self.algorithm = algorithm

    @staticmethod
    def make_gaussian_kernel(center, sigma):
        ''' Creates a Gaussian Kernel function that takes X and calculate the
        distance from the center with the sigma deviation. 
        '''
        variance = sigma**2
        gamma = 2*(variance)
        reshaped_center = np.reshape(center, newshape=(1, -1))

        def gaussian(X):
            dist = euclidean_distances(X, reshaped_center, squared=True)
            normalization_constant = 1/(2*np.pi*variance)
            return normalization_constant * np.exp(-(dist/gamma))
        return gaussian

    def fit(self, X, y):
        self._fit_centers(X, y)
        self._generate_radial_functions()

        transformed_X = self._transformed_inputs(X)
        self._linear_fit(transformed_X, y)

    def predict(self, X):
        transformed_X = self._transformed_inputs(X)
        return (self.algorithm.predict(transformed_X))

    def _fit_centers(self, X, y=None):
        kmeans = sklearn.cluster.KMeans(n_clusters=self.number_of_centers,
                                        random_state=self.random_state)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_
        groups = kmeans.predict(X)

        center_distances = euclidean_distances(X, self.centers)
        center_distances_df = pd.DataFrame(center_distances)
        center_distances_df['classe'] = groups

        # Calculate the std from the center
        # Note that the center_distances_df has the distance from each center
        center_distances_std = center_distances_df.groupby('classe').std()
        self.centers_std = np.diag(center_distances_std)

    def _generate_radial_functions(self):
        self.kernels = list()
        for cluster_center, cluster_deviance in zip(self.centers, self.centers_std):
            kernel = RBFClassifier.make_gaussian_kernel(
                cluster_center, cluster_deviance)
            self.kernels.append(kernel)

    def _transformed_inputs(self, X):
        features = [kernel(X) for kernel in self.kernels]
        features_array = (np.concatenate(features, axis=1))
        features_array = np.nan_to_num(x = features_array, nan = 0)
        return features_array

    def _linear_fit(self, X, y):
        self.algorithm.fit(X, y)
