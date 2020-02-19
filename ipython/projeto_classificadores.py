#!/usr/bin/env python
# coding: utf-8

# In[]

#################
# # Projeto PDG
#################
import pandas as pd
import numpy as np
import datetime
import sklearn.model_selection
import funcoes_auxiliares
import analise_classificacao
import data_pipeline
import RBF


# In[]:

###############
# # Preparacao
###############

SEED = 42
TRAIN_SIZE = 0.7
N_SPLITS = 10
GLM_TRESHOLD = 0.5
RBF_TRESHOLD = 0.27
NUM_CENTERS = 40
data = pd.read_csv("../data/car_insurance_claim.csv")

###############
# # Pré Tratamento dos Dados
###############
data_pre, tratamentos = data_pipeline.pre_process(data)
################
# # Separa Treino e Teste
################
train_data, test_data = sklearn.model_selection.train_test_split(data_pre, train_size = TRAIN_SIZE, random_state = SEED)

############
# # Primeiras analises - Usando apenas o Treino e K-Fold
############
kfold = sklearn.model_selection.KFold(n_splits=N_SPLITS, shuffle = True, random_state = SEED)


# In[]
# Análise Classificadores

#########################
# # Modelo - Árvore
#########################
pastas = kfold.split(train_data)
matrizes_arvore = analise_classificacao.analisa_arvore(train_data, pastas)

# print(matrizes_arvore[0].teste)
#########################
# # Modelo - GLM
#########################
pastas = kfold.split(train_data)
matrizes_glm = analise_classificacao.analisa_glm(
                                        train_data,
                                        pastas, 
                                        treshold = GLM_TRESHOLD)

# print(matrizes_glm[0].teste)
#########################
# # Modelo - RBF
#########################
pastas = kfold.split(train_data)
matrizes_rbf = analise_classificacao.analisa_rbf(
                                train_data, 
                                pastas, 
                                treshold = RBF_TRESHOLD,
                                number_of_centers = NUM_CENTERS)
#print(matrizes_rbf[0].teste)


# In[]
#########################
# # Avalia Modelos
#########################
hoje = datetime.datetime.today()
identificacao = hoje.strftime('%Y%m%d_%Hh%Mm%Ss')

tratamento_df = pd.DataFrame(tratamentos)
tratamento_df.to_csv(f"outputs/tratamento_{identificacao}.csv")

avaliacao_arvore = funcoes_auxiliares.avaliacoes_para_tabela(matrizes_arvore, "Arvore")
avaliacao_glm = funcoes_auxiliares.avaliacoes_para_tabela(matrizes_glm, "GLM")
avaliacao_rbf = funcoes_auxiliares.avaliacoes_para_tabela(matrizes_rbf, "RBF")

avaliacao_arvore.to_csv(f"outputs/avaliacao_arvore_{identificacao}.csv")
avaliacao_glm.to_csv(f"outputs/avaliacao_glm_{identificacao}.csv")
avaliacao_rbf.to_csv(f"outputs/avaliacao_rbf_{identificacao}.csv")
print(f"Salvo às {identificacao}.")
print(f"GLM_TRESHOLD = {GLM_TRESHOLD}")
print(f"RBF_TRESHOLD = {RBF_TRESHOLD}")
print(f"NUM_CENTERS = {NUM_CENTERS}")




# %%
