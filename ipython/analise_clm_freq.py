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
from funcoes_auxiliares.report_maker import ReportFrequencia

# In[]:

###############
# # Preparacao
###############
OUTPUT_PATH = "outputs/2020_03_08/CLM_FREQ"
SEED = 42
TRAIN_SIZE = 0.7
N_SPLITS = 10
FUNCOES_DE_PREPROCESSAMENTO = {"Simples": data_pipeline.pre_simples,
                                "Normalizado": data_pipeline.pre_normalizado,
                                "LogDinheiro": data_pipeline.pre_logDinheiro}

TAREFA = "REGRESSAO_FREQ"


parametros = {"Tree": {"MaxDepth": None},
              "GLM": {"Família": sm.families.Tweedie,
                      "Var Power": 1.0,
                      # Verificar se o link está sendo usado.
                      "Link": sm.genmod.families.links.log,  
                      "Limiar": "Irrelevante"},
              "RBFN": {"NCentroides": 40,
                       "Limiar": "Irrelevante"}}

# In[]: Análise
data = pd.read_csv("../data/car_insurance_clain_train.csv")

for nome_preprocessamento, funcao_de_preprocessamento in FUNCOES_DE_PREPROCESSAMENTO.items():

    ###############
    # # Pré Tratamento dos Dados
    ###############
    train_data, tratamentos = data_pipeline.pre_process(data)
    kfold = sklearn.model_selection.KFold(n_splits=N_SPLITS, shuffle = True, random_state = SEED)
    
    ############
    # # Primeiras analises - Usando apenas o Treino e K-Fold
    ############

    #########################
    # # Modelo - Árvore
    #########################
    pastas = kfold.split(train_data)
    matrizes_arvore = analise_regressao_freq.analisa_arvore(
                                                dados = train_data, 
                                                folds = pastas, 
                                    parametros = parametros)

    #########################
    # # Modelo - GLM
    #########################
    pastas = kfold.split(train_data)
    matrizes_glm = analise_regressao_freq.analisa_glm(
                                            dados = train_data,
                                            folds = pastas, 
                                    parametros = parametros)

    #########################
    # # Modelo - RBF
    #########################
    pastas = kfold.split(train_data)
    matrizes_rbf = analise_regressao_freq.analisa_rbf(
                                    dados = train_data, 
                                    folds = pastas, 
                                    parametros = parametros)

    ########################
    # # Gerar Relatório
    ########################
    reportGenerator = ReportFrequencia({"Árvore" : matrizes_arvore, 
                                 "GLM" : matrizes_glm, 
                                 "RBFN" : matrizes_rbf},
                                preprocessamentos = tratamentos, 
                                parametros = parametros, 
                                nome_preprocessamento = nome_preprocessamento)

    reportGenerator.generate_report(output_path = OUTPUT_PATH)


# %%
