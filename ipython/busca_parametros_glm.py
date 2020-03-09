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
from funcoes_auxiliares import funcoes_auxiliares_classificacao
from funcoes_auxiliares import funcoes_auxiliares_regressao_freq
from funcoes_auxiliares import funcoes_auxiliares_regressao_valor
import data_pipeline
import RBF

import statsmodels.api as sm

from funcoes_auxiliares.report_maker import ReportFlag, ReportValor, ReportFrequencia

# In[]: Preparação
OUTPUT_PATH = "outputs/2020_03_08/BuscaParametros/GLM"
SEED = 42
TRAIN_SIZE = 0.7
N_SPLITS = 10
FUNCOES_DE_PREPROCESSAMENTO = {"Simples": data_pipeline.pre_simples,
                                "Normalizado": data_pipeline.pre_normalizado,
                                "LogDinheiro": data_pipeline.pre_logDinheiro}

TAREFA = {"CLASSIFICACAO": "CLAIM_FLAG", 
          "REGRESSAO_AMT": "CLM_AMT",
          "REGRESSAO_FREQ": "CLM_FREQ"}

TARGETS = ["CLM_FREQ","CLM_AMT","CLAIM_FLAG"]


parametros = {"GLM": {"Var Power": [1.0, 1.2, 1.5, 1.8, 2.0],
                      "Limiar": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}}

# In[]: Pré Análise
raw_data = pd.read_csv("../data/car_insurance_clain_train.csv", index_col=0)

for var_power in parametros["GLM"]["Var Power"]:
    resultados_classificacao = {limiar: [] for limiar in parametros["GLM"]["Limiar"]}
    resultados_valor = list()
    resultados_frequencia = list()
    for nome_preprocessamento, funcao_de_preprocessamento in FUNCOES_DE_PREPROCESSAMENTO.items():
        print(f"Preparando Dados.\nTratamento Dado: {nome_preprocessamento}")
        train_data, tratamentos = funcao_de_preprocessamento(raw_data)

        X_data = train_data.drop(TARGETS, axis = 1)
        y_data = train_data.loc[ : , TARGETS]
        
        kfold = sklearn.model_selection.KFold(n_splits=N_SPLITS, shuffle = True, random_state = SEED)
        pastas = kfold.split(train_data)

        
        for i, (train_index, test_index) in enumerate(pastas):
            X_treino = X_data.iloc[train_index, : ]
            y_treinoDF = y_data.iloc[train_index, : ]

            X_teste = X_data.iloc[test_index, : ]
            y_testeDF = y_data.iloc[test_index, : ]

            print("Classificacao")
            y_teste = y_testeDF.loc[ : , "CLAIM_FLAG"]
            y_treino = y_treinoDF.loc[ : , "CLAIM_FLAG"]

            familia = sm.families.Tweedie(var_power = var_power)
            
            model = sm.GLM(exog = X_treino,
                           endog = y_treino,
                           family = familia)
            
            glm = model.fit()

            probabilidade_treino = glm.predict(X_treino)
            probabilidade_teste = glm.predict(X_teste)

            for treshold in parametros["GLM"]["Limiar"]:
                treino_previsto = probabilidade_treino >= treshold
                teste_previsto = probabilidade_teste >= treshold

                avaliacao = funcoes_auxiliares_classificacao.avalia_modelo(y_treino, y_teste, treino_previsto, teste_previsto)
                resultados_classificacao[treshold].append(avaliacao)
            

            print("Valor")
            y_teste = y_testeDF.loc[ : , "CLM_AMT"]
            y_treino = y_treinoDF.loc[ : , "CLM_AMT"]
            
            model = sm.GLM(exog = X_treino,
                           endog = y_treino,
                           family = familia)
            
            glm = model.fit()

            treino_previsto = glm.predict(X_treino)
            teste_previsto = glm.predict(X_teste)

            avaliacao = funcoes_auxiliares_regressao_valor.avalia_modelo(y_treino, y_teste, treino_previsto, teste_previsto)
            resultados_valor.append(avaliacao)


            print("Frequencia")
            y_teste = y_testeDF.loc[ : , "CLM_FREQ"]
            y_treino = y_treinoDF.loc[ : , "CLM_FREQ"]
            
            model = sm.GLM(exog = X_treino,
                           endog = y_treino,
                           family = familia)
            
            glm = model.fit()

            treino_previsto = glm.predict(X_treino)
            teste_previsto = glm.predict(X_teste)

            avaliacao = funcoes_auxiliares_regressao_freq.avalia_modelo(y_treino, y_teste, treino_previsto, teste_previsto)
            resultados_frequencia.append(avaliacao)

 
            
        ReportFlag(resultados_classificacao,
                                            preprocessamentos = tratamentos,
                                            parametros = {"GLM": {"VarPower": var_power, "Limiar": parametros["GLM"]["Limiar"]}},
                                            nome_preprocessamento = nome_preprocessamento,
                                            ).generate_report(output_path = OUTPUT_PATH, identificador = "BuscaParamGLM")
        
        ReportValor({var_power: resultados_valor},
                                        preprocessamentos = tratamentos,
                                        parametros = {"GLM": {"VarPower": var_power}},
                                        nome_preprocessamento = nome_preprocessamento,
                                        ).generate_report(output_path = OUTPUT_PATH, identificador = "BuscaParamGLM")

        ReportFrequencia({var_power: resultados_valor},
                                        preprocessamentos = tratamentos,
                                        parametros = {"GLM": {"VarPower": var_power}},
                                        nome_preprocessamento = nome_preprocessamento,
                                        ).generate_report(output_path = OUTPUT_PATH, identificador = "BuscaParamGLM")


# %%
