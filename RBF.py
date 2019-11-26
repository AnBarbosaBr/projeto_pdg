#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import numpy as np

import sklearn.cluster
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances


# In[25]:


seed = 42
k = 6


# In[26]:


from sklearn import datasets


# In[27]:


iris = datasets.load_iris()


# In[28]:


X = iris.data
y = iris.target


# In[31]:


X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 50, random_state = seed)


# In[36]:


kmeans = sklearn.cluster.KMeans(n_clusters = k, random_state = seed)


# In[37]:


kmeans.fit(X_treino)


# In[40]:


centros = kmeans.cluster_centers_ 
# pd.DataFrame(centros)


# In[47]:


classificados = kmeans.predict(X_treino)
# classificados


# In[76]:


distancias_para_cada_centro = euclidean_distances(X_treino, centros)
distancias_para_cada_centro


# In[235]:


df_distancias = pd.DataFrame(distancias_para_cada_centro)
df_distancias['classe'] = classificados


# In[236]:


distancias_medias = df_distancias.groupby('classe').mean()
medias_por_grupo = np.diag(medias)


# In[237]:


desvios = df_distancias.groupby('classe').std()
desvios_por_grupo = np.diag(desvios)


# In[238]:


def make_gaussian_kernel(center, sigma):
    variance = sigma**2
    gamma = 2*(variance)
    reshaped_center = np.reshape(center, newshape = (1,-1))
    def gaussian(x):
        dist = euclidean_distances(x, reshaped_center, squared=True)
        normalization_constant = 1/(2*np.pi*variance)
        return  normalization_constant * np.exp(-(dist/gamma))
    return gaussian


# In[244]:


kernels = list()
for cluster_alvo in range(k):
    alvos = X_treino[classificados==cluster_alvo]
    kernel = make_gaussian_kernel(centros[cluster_alvo], desvios_por_grupo[cluster_alvo])
    kernels.append(kernel)


# In[256]:


features = list()
for kernel in kernels:
    features.append(kernel(X_treino))



    
    


# In[261]:


transformed_input = pd.DataFrame(np.concatenate(features, axis = 1))


# In[268]:


regression = LogisticRegression()


# In[269]:


regression.fit(transformed_input, y_treino)


# In[275]:


predicoes = regression.predict(transformed_input)


# In[279]:


import sklearn.metrics
from sklearn.linear_model import LogisticRegression


# In[280]:


sklearn.metrics.multilabel_confusion_matrix(y_true = y_treino, y_pred = predicoes)


# In[283]:


sklearn.metrics.accuracy_score(y_treino, predicoes)


# In[ ]:




