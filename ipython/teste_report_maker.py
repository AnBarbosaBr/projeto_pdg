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
import RBF
from funcoes_auxiliares import analise_classificacao
from funcoes_auxiliares import funcoes_auxiliares_classificacao
import data_pipeline
from funcoes_auxiliares.report_maker import ReportFlag
import statsmodels.api as sm

# In[]: Preparação
OUTPUT = "outputs/2020_03_08"
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


TAREFA = "CLASSIFICACAO_FLAG"
CRITERIO_MSE = "MSE"
CRITERIO_POISSON_DEVIACNE = "PoissonDeviance"


criterio = CRITERIO_MSE

parametros = {"Tree": {"MaxDepth": TAMANHO_ARVORE},
              "GLM": {"Família": None,
                      "Var Power": None,
                      "Link": None, 
                      "Limiar": GLM_TRESHOLD},
              "RBFN": {"NCentroides": NUM_CENTERS,
                       "Limiar":RBF_TRESHOLD}}


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
    # Apenas uma amostra 
    train_data = train_data[0:1000]
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

    matrizes_arvore = analise_classificacao.analisa_arvore(train_data, pastas, parametros)

    # print(matrizes_arvore[0].teste)
    #########################
    # # Modelo - GLM
    #########################
    print("\tAnalisando GLM")
    pastas = kfold.split(train_data)
    matrizes_glm = analise_classificacao.analisa_glm(
                                            train_data,
                                            pastas,
                                            parametros)

    # print(matrizes_glm[0].teste)
    #########################
    # # Modelo - RBF
    #########################
    print("\tAnalisando RBF")
    pastas = kfold.split(train_data)
    matrizes_rbf = analise_classificacao.analisa_rbf(
                                    train_data, 
                                    pastas, 
                                    parametros)
    #print(matrizes_rbf[0].teste)


    reportGenerator = ReportFlag({"TREE" : matrizes_arvore, 
                                 "GLM" : matrizes_glm, 
                                 "RBFN" : matrizes_rbf},
                                preprocessamentos = tratamentos, 
                                parametros = parametros, 
                                nome_preprocessamento = nome_preprocessamento)

    reportGenerator.generate_report(identificador = "TesteReportMaker", output_path = OUTPUT)
