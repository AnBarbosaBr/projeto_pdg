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

from funcoes_auxiliares.report_maker import ReportFlag

# In[]: Preparação
OUTPUT_PATH = "outputs/2020_03_08/FLAG"
SEED = 42
TRAIN_SIZE = 0.7
N_SPLITS = 10
FUNCOES_DE_PREPROCESSAMENTO = {"Simples": data_pipeline.pre_simples,
                                "Normalizado": data_pipeline.pre_normalizado,
                                "LogDinheiro": data_pipeline.pre_logDinheiro}

TAREFA = "CLASSIFICACAO"

parametros = {"Tree": {"MaxDepth": None},
              "GLM": {"Família": sm.families.Binomial,
                      "Var Power": None,
                      "Link": None, 
                      "Limiar": 0.5},
              "RBFN": {"NCentroides": 40,
                       "Limiar": 0.27}}


# In[]: Pré Análise
data = pd.read_csv("../data/car_insurance_clain_train.csv")

for nome_preprocessamento, funcao_de_preprocessamento in FUNCOES_DE_PREPROCESSAMENTO.items():
    
    print(f"Iniciando análise de classificação. Tratamento Atributos: {nome_preprocessamento}")
    
    ###############
    # # Pré Tratamento dos Dados
    ###############
    train_data, tratamentos = funcao_de_preprocessamento(data)
    
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
    matrizes_arvore = analise_classificacao.analisa_arvore(
                                            train_data, 
                                            pastas, 
                                            parametros)

    #########################
    # # Modelo - GLM
    #########################
    print("\tAnalisando GLM")
    pastas = kfold.split(train_data)
    matrizes_glm = analise_classificacao.analisa_glm(
                                            train_data,
                                            pastas, 
                                            parametros)

    #########################
    # # Modelo - RBF
    #########################
    print("\tAnalisando RBF")
    pastas = kfold.split(train_data)
    matrizes_rbf = analise_classificacao.analisa_rbf(
                                    train_data, 
                                    pastas, 
                                    parametros)

    ########################
    # # Gerar Relatório
    ########################
    reportGenerator = ReportFlag({"Árvore" : matrizes_arvore, 
                                 "GLM" : matrizes_glm, 
                                 "RBFN" : matrizes_rbf},
                                preprocessamentos = tratamentos, 
                                parametros = parametros, 
                                nome_preprocessamento = nome_preprocessamento)

    reportGenerator.generate_report(output_path = OUTPUT_PATH)


# %%
