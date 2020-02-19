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
from funcoes_auxiliares.funcoes_auxiliares_regressao import avalia_modelo, avaliacoes_para_tabela
from funcoes_auxiliares.analise_regressao_freq import analisa_arvore, analisa_glm, analisa_rbf
import data_pipeline
import RBF

import statsmodels.api as sm

# In[]:

###############
# # Preparacao
###############

SEED = 42
TRAIN_SIZE = 0.7
N_SPLITS = 10

LINK = sm.genmod.families.links.log
FAMILIA = sm.families.Tweedie(var_power = 1.0)
familia = FAMILIA

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
# Análise Arvore

#########################
# # Modelo - Árvore
#########################
pastas = kfold.split(train_data)
matrizes_arvore = analisa_arvore(
                                            dados = train_data, 
                                            folds = pastas)

# print(matrizes_arvore[0].teste)

# In[]
# Analise GLM
#########################
# # Modelo - GLM
#########################
pastas = kfold.split(train_data)
matrizes_glm = analisa_glm(
                                        dados = train_data,
                                        folds = pastas,
                                        familia = FAMILIA)

print(matrizes_glm[0].teste)
# In[]
# Analise RBF
#########################
# # Modelo - RBF
#########################
pastas = kfold.split(train_data)
matrizes_rbf = analisa_rbf(
                                dados = train_data, 
                                folds = pastas, 
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

avaliacao_arvore = avaliacoes_para_tabela(matrizes_arvore, "Arvore")
avaliacao_glm = avaliacoes_para_tabela(matrizes_glm, "GLM")
avaliacao_rbf = avaliacoes_para_tabela(matrizes_rbf, "RBF")

avaliacao_arvore.to_csv(f"outputs/avaliacao_arvore_{identificacao}.csv")
avaliacao_glm.to_csv(f"outputs/avaliacao_glm_{identificacao}.csv")
avaliacao_rbf.to_csv(f"outputs/avaliacao_rbf_{identificacao}.csv")
print(f"Salvo às {identificacao}.")
print(f"NUM_CENTERS = {NUM_CENTERS}")


# %%
