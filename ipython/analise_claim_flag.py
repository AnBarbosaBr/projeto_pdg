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
from funcoes_auxiliares import analise_classificacao
from funcoes_auxiliares import funcoes_auxiliares_classificacao
import data_pipeline
import RBF

import statsmodels.api as sm

# In[]: Preparação

SEED = 42
TRAIN_SIZE = 0.7
N_SPLITS = 10
GLM_TRESHOLD = 0.5
RBF_TRESHOLD = 0.27
NUM_CENTERS = 40

TAMANHO_ARVORE = None
FUNCOES_DE_PREPROCESSAMENTO = {"Simples": data_pipeline.pre_simples,
                                "Normalizado": data_pipeline.pre_normalizado,
                                "LogDinheiro": data_pipeline.pre_logDinheiro}

TAREFA = "CLASSIFICACAO"



# In[]: Pré Análise
data = pd.read_csv("../data/car_insurance_claim.csv")

for nome_preprocessamento, funcao_de_preprocessamento in FUNCOES_DE_PREPROCESSAMENTO.items():
    
    print(f"Iniciando análise de classificação. Tratamento Atributos: {nome_preprocessamento}")
    
    ###############
    # # Pré Tratamento dos Dados
    ###############
    data_pre, tratamentos = funcao_de_preprocessamento(data)
    ################
    # # Separa Treino e Teste
    ################
    train_data, test_data = sklearn.model_selection.train_test_split(data_pre, train_size = TRAIN_SIZE, random_state = SEED)

    ############
    # # Primeiras analises - Usando apenas o Treino e K-Fold
    ############
    kfold = sklearn.model_selection.KFold(n_splits=N_SPLITS, shuffle = True, random_state = SEED)


    # In[]: Análise CLAIM_FLAG - Classificação

    #########################
    # # Modelo - Árvore
    #########################
    print("\tAnalisando Árvores")
    pastas = kfold.split(train_data)

    matrizes_arvore = analise_classificacao.analisa_arvore(train_data, pastas)

    # print(matrizes_arvore[0].teste)
    #########################
    # # Modelo - GLM
    #########################
    print("\tAnalisando GLM")
    pastas = kfold.split(train_data)
    matrizes_glm = analise_classificacao.analisa_glm(
                                            train_data,
                                            pastas, 
                                            treshold = GLM_TRESHOLD)

    # print(matrizes_glm[0].teste)
    #########################
    # # Modelo - RBF
    #########################
    print("\tAnalisando RBF")
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
    tratamento_df.to_csv(f"outputs/{TAREFA}_{nome_preprocessamento}_{identificacao}.csv")

    avaliacao_arvore = funcoes_auxiliares_classificacao.avaliacoes_para_tabela(matrizes_arvore, "Arvore")
    avaliacao_glm = funcoes_auxiliares_classificacao.avaliacoes_para_tabela(matrizes_glm, "GLM")
    avaliacao_rbf = funcoes_auxiliares_classificacao.avaliacoes_para_tabela(matrizes_rbf, "RBF")

    columns_details = ["Nome","Tipo","Iteracao", "True_Positive","True_Negative","False_Positive","False_Negative"]
    columns_summary = ["Nome","Tipo", "True_Positive","True_Negative","False_Positive","False_Negative"]
    avaliacoes = pd.concat([avaliacao_arvore, avaliacao_glm, avaliacao_rbf], axis = "index")[columns_details].reset_index().drop("index",axis="columns")
    sumarios = avaliacoes.groupby(["Nome","Tipo"]).sum().drop("Iteracao", axis="columns").reset_index()[columns_summary]

    avaliacoes.to_csv(f"outputs/detalhes_{TAREFA}_{nome_preprocessamento}_{identificacao}.csv")
    sumarios.to_csv(f"outputs/sumarios_{TAREFA}_{nome_preprocessamento}_{identificacao}.csv")
    print(f"Salvo às {identificacao}.")
    print(f"GLM_TRESHOLD = {GLM_TRESHOLD}")
    print(f"RBF_TRESHOLD = {RBF_TRESHOLD}")
    print(f"NUM_CENTERS = {NUM_CENTERS}")


# %%
