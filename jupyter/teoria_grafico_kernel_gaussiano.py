# -*- coding: utf-8 -*-
"""Plota o gráfico do Kernel Gaussiano"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sn
import pandas as pd
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


    
    def plota_kernel(intervalo = 100):  
        x = np.array(range(-intervalo, intervalo, 1))/100
        x = x.reshape(-1, 1)
        
        kernel01 = make_gaussian_kernel(0, 0.8)
        kernel10 = make_gaussian_kernel(0, 1)
        kernel100 = make_gaussian_kernel(0, 1.2)
        
        
        y01 = kernel01(x)
        y10 = kernel10(x)
        y100 = kernel100(x)
        
        plt.title("Kernel Gaussiano")
        plt.xlabel("$\chi$")
        plt.ylabel("$h_{m}(\chi)$")
        plt.grid(color="lightgray")
        
        plt.plot(x,y01, color = "black", linestyle = "dotted", label = "0.8")
        plt.plot(x,y10, color = "black", linestyle = "solid", label = "1")
        plt.plot(x,y100, color = "black", linestyle = "dashed", label = "1.2")
        plt.legend(title = "Dispersão $(\sigma)$")
           
    plota_kernel(350)