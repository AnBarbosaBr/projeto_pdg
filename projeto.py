#!/usr/bin/env python
# coding: utf-8

# # Projeto PDG



import pandas as pd;
import numpy as np; 
import sklearn.model_selection;
import sklearn.metrics;


from sklearn.tree import DecisionTreeClassifier; # Tree

# OBS: Scikit-Learn
# scikit-learn uses an optimised version of the CART algorithm; however, scikit-learn implementation does not support categorical variables for now.
# https://scikit-learn.org/stable/modules/tree.html



import statsmodels.api as sm; # GLM


# In[4]:


seed = 42


# In[37]:


data = pd.read_csv("data/car_insurance_claim.csv")


# In[17]:


# Linhas / Colunas
forma = data.shape
print(f"{data.shape[0]} linhas e {data.shape[1]} colunas")
print(pd.DataFrame(data.columns))
print(data.describe())
print(data.describe().shape[1]) # Colunas Númericas


# In[114]:


def score_results(y_real, y_predito, label):
    matriz_de_confusao = sklearn.metrics.confusion_matrix(y_true = y_real, y_pred = y_predito)
    tn, fp, fn, tp = matriz_de_confusao.ravel()
    
    print(f"--- {label} ---")
    print("Matrix de Confusão")
    print(matriz_de_confusao)
    print("Balanced Accuracy: ", end=" ")
    print(f"{100*sklearn.metrics.balanced_accuracy_score(y_true = y_real, y_pred = y_predito):.2f}%")
    print(f"Falsos Positivos: {fp}, Falsos Negativos: {fn}\n"+
            f"Verdadeiros Positivos: {tp}, Verdadeiros Negativos: {tn}")
    print(f"Precisao (tp/(tp+fp)): {100*tp/(tp+fp) :.2f}%")
    print(f"Recall (tp/(tp+fn)): {100*tp/(tp+fn) :.2f}%")
    
    print("-"*50)


# In[125]:


## Decision Tree
independent_variables = list(range(1,25))
dependent_variables = 25
cartTree = DecisionTreeClassifier()

X_data = data.iloc[ : , independent_variables]
y_data = data.iloc[ : , dependent_variables]

stratified_k_fold = sklearn.model_selection.StratifiedKFold(n_splits = 5, random_state = seed )

pastas = stratified_k_fold.split(X_data, y_data)

for i, (train, test) in enumerate(pastas):
    print("-"*80)
    print(f"K Fold - Rodada {i}\n")
    X_train = X_data.iloc[train, : ]
    y_train = y_data.iloc[train]

    X_test = X_data.iloc[test, : ]
    y_test = y_data.iloc[test]
    
    # Improve this part
    numerical_x_train = X_train._get_numeric_data()
    numerical_x_train.fillna(numerical_x_train.mean(), inplace=True)
    
    fit_result = cartTree.fit(numerical_x_train, y_train)
    print(fit_result)
    print("-"*50)
    
    predicted_train = cartTree.predict(numerical_x_train)
    score_results(y_train, predicted_train, "CART Tree - Treinamento")
    
    # Improve this part
    numerical_x_test = X_test._get_numeric_data()
    numerical_x_test.fillna(numerical_x_test.mean(), inplace=True)
    
    predicted_test = cartTree.predict(numerical_x_test)
    score_results(y_test, predicted_test, "CART Tree - Teste")


# In[146]:


## GLM
independent_variables = list(range(1,25))
dependent_variables = 25

X_data = data.iloc[ : , independent_variables]
y_data = data.iloc[ : , dependent_variables]

stratified_k_fold = sklearn.model_selection.StratifiedKFold(n_splits = 5, random_state = seed )

pastas = stratified_k_fold.split(X_data, y_data)

for i, (train, test) in enumerate(pastas):
    print("-"*80)
    print(f"K Fold - Rodada {i}\n")
    X_train = X_data.iloc[train, : ]
    y_train = y_data.iloc[train]

    X_test = X_data.iloc[test, : ]
    y_test = y_data.iloc[test]
    
    # Improve this part
    numerical_x_train = X_train._get_numeric_data()
    numerical_x_train.fillna(numerical_x_train.mean(), inplace=True)    
    #################################################
    
    glm = sm.GLM(exog = numerical_x_train, endog =  y_train, family=sm.families.Binomial())
    
    predictor_glm = glm.fit()
    print(predictor_glm.summary())
    print("-"*50)

 
    treshold = 0.5
    predicted_train_probs = predictor_glm.predict(numerical_x_train)
    
    predicted_train = (predicted_train_probs > treshold)
    score_results(y_train, predicted_train, "GLM - Treinamento")

    # Improve this part
    numerical_x_test = X_test._get_numeric_data()
    numerical_x_test.fillna(numerical_x_test.mean(), inplace=True)
    ######################################################
    predicted_test_probs = predictor_glm.predict(numerical_x_test)
    predicted_test =  (predicted_test_probs > treshold )
    
    score_results(y_test, predicted_test, "GLM - Teste")
    




# In[ ]:


# RBFN
## Setup

## Train
## Test


# Compare results


# In[134]:


#get_ipython().run_line_magic('pinfo', 'sm.GLM')

