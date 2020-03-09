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
from funcoes_auxiliares import report_maker
from funcoes_auxiliares import analise_regressao_valor
from funcoes_auxiliares import funcoes_auxiliares_regressao_valor
import data_pipeline
import RBF

import statsmodels.api as sm
from funcoes_auxiliares.report_maker import ReportValor




# In[]:

###############
# # Preparacao
###############
OUTPUT_PATH = "outputs/2020_03_08/CLM_AMT"
SEED = 42
TRAIN_SIZE = 0.7
N_SPLITS = 10

FUNCOES_DE_PREPROCESSAMENTO = {"Simples": data_pipeline.pre_simples,
                                "Normalizado": data_pipeline.pre_normalizado,
                                "LogDinheiro": data_pipeline.pre_logDinheiro}

TAREFA = "REGRESSAO_AMT"

parametros = {"Tree": {"MaxDepth": None},
              "GLM": {"Família": sm.families.Tweedie,
                      "Var Power": 1.0,
                      # Verificar se o link está sendo usado.
                      "Link": sm.genmod.families.links.log,  
                      "Limiar": "Irrelevante"},
              "RBFN": {"NCentroides": 40,
                       "Limiar": "Irrelevante"}}

# In[]: Pré Análise
raw_data = pd.read_csv("../data/car_insurance_clain_train.csv", index_col=0)

for nome_preprocessamento, funcao_de_preprocessamento in FUNCOES_DE_PREPROCESSAMENTO.items():
    print(f"Iniciando análise de valor. Tratamento Atributos: {nome_preprocessamento}")
    
    ###############
    # # Pré Tratamento dos Dados
    ###############
    train_data, tratamentos = funcao_de_preprocessamento(raw_data)


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
    matrizes_arvore = analise_regressao_valor.analisa_arvore(
                                                dados = train_data, 
                                                folds = pastas,
                                                parametros = parametros)

    # print(matrizes_arvore[0].teste)

    # In[]
    # Analise GLM
    #########################
    # # Modelo - GLM
    #########################
    pastas = kfold.split(train_data)
    matrizes_glm = analise_regressao_valor.analisa_glm(
                                            dados = train_data,
                                            folds = pastas,
                                            parametros = parametros)

        # In[]
    # Analise RBF
    #########################
    # # Modelo - RBF
    #########################
    pastas = kfold.split(train_data)
    matrizes_rbf = analise_regressao_valor.analisa_rbf(
                                    dados = train_data, 
                                    folds = pastas,
                                    parametros = parametros)

    reportGenerator = ReportValor({"Árvore" : matrizes_arvore, 
                                 "GLM" : matrizes_glm, 
                                 "RBFN" : matrizes_rbf},
                                preprocessamentos = tratamentos, 
                                parametros = parametros, 
                                nome_preprocessamento = nome_preprocessamento)

    reportGenerator.generate_report(output_path = OUTPUT_PATH)