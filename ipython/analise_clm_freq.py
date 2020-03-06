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
from funcoes_auxiliares import funcoes_auxiliares_regressao_freq
from funcoes_auxiliares import analise_regressao_freq
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

TAMANHO_ARVORE = None
LINK = sm.genmod.families.links.log
#FAMILIA = sm.families.Tweedie(var_power = 1.0)
FAMILIA = sm.families.Tweedie(var_power = 1.0)
familia = FAMILIA

NUM_CENTERS = 40



FUNCOES_DE_PREPROCESSAMENTO = {"Simples": data_pipeline.pre_simples,
                                "Normalizado": data_pipeline.pre_normalizado,
                                "LogDinheiro": data_pipeline.pre_logDinheiro}

TAREFA = "REGRESSAO_FREQ"

# In[]: Análise
data = pd.read_csv("../data/car_insurance_claim.csv")
train_data, test_data = sklearn.model_selection.train_test_split(data_pre, train_size = TRAIN_SIZE, random_state = SEED)

for nome_preprocessamento, funcao_de_preprocessamento in FUNCOES_DE_PREPROCESSAMENTO.items():

    ###############
    # # Pré Tratamento dos Dados
    ###############
    data_pre, tratamentos = data_pipeline.pre_process(data)
    ################
    # # Separa Treino e Teste
    ################

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
    matrizes_arvore = analise_regressao_freq.analisa_arvore(
                                                dados = train_data, 
                                                folds = pastas,
                                                depth = TAMANHO_ARVORE)

    # print(matrizes_arvore[0].teste)

    # In[]
    # Analise GLM
    #########################
    # # Modelo - GLM
    #########################
    pastas = kfold.split(train_data)
    matrizes_glm = analise_regressao_freq.analisa_glm(
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
    matrizes_rbf = analise_regressao_freq.analisa_rbf(
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

    avaliacao_arvore = funcoes_auxiliares_regressao_freq.avaliacoes_para_tabela(matrizes_arvore, "Arvore")
    avaliacao_glm = funcoes_auxiliares_regressao_freq.avaliacoes_para_tabela(matrizes_glm, "GLM")
    avaliacao_rbf = funcoes_auxiliares_regressao_freq.avaliacoes_para_tabela(matrizes_rbf, "RBF")

    # avaliacao_arvore.to_csv(f"outputs/avaliacao_arvore_{identificacao}.csv")
    # avaliacao_glm.to_csv(f"outputs/avaliacao_glm_{identificacao}.csv")
    # avaliacao_rbf.to_csv(f"outputs/avaliacao_rbf_{identificacao}.csv")
    # print(f"Salvo às {identificacao}.")
    # print(f"NUM_CENTERS = {NUM_CENTERS}")

    columns_details = ["Nome","Tipo","Iteracao", "PoissonDeviance"]
    columns_summary = ["Nome","Tipo", "PoissonDeviance"]
    avaliacoes = pd.concat([avaliacao_arvore, avaliacao_glm, avaliacao_rbf], axis = "index")[columns_details].reset_index().drop("index",axis="columns")
    sumarios = avaliacoes.groupby(["Nome","Tipo"]).sum().drop("Iteracao", axis="columns").reset_index()[columns_summary]

    # avaliacao_arvore.to_csv(f"outputs/avaliacao_arvore_{identificacao}.csv")
    # avaliacao_glm.to_csv(f"outputs/avaliacao_glm_{identificacao}.csv")
    # avaliacao_rbf.to_csv(f"outputs/avaliacao_rbf_{identificacao}.csv")

    avaliacoes.to_csv(f"outputs/detalhes_{TAREFA}_{nome_preprocessamento}_{identificacao}.csv")
    sumarios.to_csv(f"outputs/sumarios_{TAREFA}_{nome_preprocessamento}_{identificacao}.csv")

    print(f"Salvo às {identificacao}.")
    print(f"NUM_CENTERS = {NUM_CENTERS}")


# %%
