#!/usr/bin/env python
# coding: utf-8
# In[]
# Inicializacao
import os
os.chdir("e:\\andre\\documentos\\ufabc\\pdg\\projeto_ipython")

#################
# # Projeto PDG
#################
import datetime
import pandas as pd
import numpy as np
import statsmodels.api as sm
import sklearn.model_selection
import funcoes_auxiliares.funcoes_auxiliares_regressao as auxiliares
import funcoes_auxiliares.analise_regressao_valor as regressao
import data_pipeline
import RBF


# In[]:

###############
# # Preparacao
###############

SEED = 42
TRAIN_SIZE = 0.7
N_SPLITS = 10


data = pd.read_csv("../data/car_insurance_claim.csv")

###############
# # Pré Tratamento dos Dados
###############
data_pre, tratamentos = data_pipeline.pre_process(data)

# Separa Treino e Teste - A calibração é feita sobre o treino, nós não usamos os testes
train_data, test_data = sklearn.model_selection.train_test_split(data_pre, train_size = TRAIN_SIZE, random_state = SEED)

# Evitando usar o teste por acaso nessa etapa
test_data = None 

# In[]
# # Análise

hoje = datetime.datetime.today()
horario_str = hoje.strftime('%Y%m%d_%Hh%Mm%Ss')
tratamentod_df = pd.DataFrame(tratamentos)
tratamentod_df.to_csv(f"outputs/tratamentos_{horario_str}.csv")

# Analise 0: Grosso modo, para ver o treshold
N_SPLITS = 10
TWEEDIE_VALUES = [value/100 for value in range (166, 201)]

kfold = sklearn.model_selection.KFold(n_splits=N_SPLITS, shuffle = True, random_state = SEED)
for i, value in enumerate(TWEEDIE_VALUES):
    print(f"Iteração [{1 + i}]")
    print(f"Avaliando {value}.")
    pastas = kfold.split(train_data)
    familia = sm.families.Tweedie(var_power = value)
    try:
        matrizes_glm = regressao.analisa_glm(
                                        dados = train_data, 
                                        folds = pastas, 
                                        familia = familia)
            
        identificacao = f"GLM_Tweedie_Valor_{value}"
        avaliacao = auxiliares.avaliacoes_para_tabela(matrizes_glm, identificacao)
        avaliacao.to_csv(f"outputs/avaliacao_{identificacao}_{horario_str}.csv")
        print("Resultado salvo.")
    except ValueError as err:
        mensagem = f"ERRO NO VALOR {value}: {str(err)}"
        with open(f"outputs/erros_{identificacao}_{horario_str}.txt", "a") as f:
            f.write(mensagem)
            f.write("\n")
        print(mensagem)

print(f"Fim analise Tweedies. Id: {horario_str}")

