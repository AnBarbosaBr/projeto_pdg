#!/usr/bin/env python
# coding: utf-8

#################
# # Projeto PDG
#################
import datetime
import pandas as pd
import numpy as np
import sklearn.model_selection
import funcoes_auxiliares.funcoes_auxiliares
import funcoes_auxiliares.analise_classificacao
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

data = pd.read_csv("../data/car_insurance_claim.csv")

###############
# # Pré Tratamento dos Dados
###############
data_pre, tratamentos = data_pipeline.pre_process(data)


################
# # Preparacao
################

data_pre = data_pre


################
# # Separa Treino e Teste - A calibração é feita sobre o treino, nós não usamos os testes
################
train_data, test_data = sklearn.model_selection.train_test_split(data_pre, train_size = TRAIN_SIZE, random_state = SEED)
# Evitando usar o teste por acaso nessa etapa
test_data = None 

################
# # Análise
################
hoje = datetime.datetime.today()
horario_str = hoje.strftime('%Y%m%d_%Hh%Mm%Ss')
tratamentod_df = pd.DataFrame(tratamentos)
tratamentod_df.to_csv(f"outputs/tratamentos_{horario_str}.csv")

# Analise 0: Grosso modo, para ver o treshold

RBF_TRESHOLD = [-1 + 2*value/10 for value in range(0, 11)]
NUM_CENTERS = [40]
N_SPLITS = 10
kfold = sklearn.model_selection.KFold(n_splits=N_SPLITS, shuffle = True, random_state = SEED)
for i, n_center in enumerate(NUM_CENTERS):
    for i2, treshold in enumerate(RBF_TRESHOLD):
        print(f"Iteração [{1 + i*len(RBF_TRESHOLD) + i2} de {len(RBF_TRESHOLD * len(NUM_CENTERS))}]")
        print(f"Avaliando {n_center} e {treshold}")
        pastas = kfold.split(train_data)
        matrizes_rbf = analise_classificacao.analisa_rbf(
                                        train_data, 
                                        pastas, 
                                        treshold = treshold,
                                        number_of_centers = n_center)
        
        identificacao = f"RBF_{n_center}_{treshold}"
        avaliacao = funcoes_auxiliares.avaliacoes_para_tabela(matrizes_rbf, identificacao)
        avaliacao.to_csv(f"outputs/avaliacao_{identificacao}_{horario_str}.csv")
        print("Resultado salvo.")

    print("Fim parte 0")

# Analise 2 - Treshold Ajuste Fino
RBF_TRESHOLD = [0.2 + value/100 for value in range(0, 21)]
NUM_CENTERS = [40]
N_SPLITS = 10
kfold = sklearn.model_selection.KFold(n_splits=N_SPLITS, shuffle = True, random_state = SEED)
for i, n_center in enumerate(NUM_CENTERS):
    for i2, treshold in enumerate(RBF_TRESHOLD):
        print(f"Iteração [{1 + i*len(RBF_TRESHOLD) + i2} de {len(RBF_TRESHOLD * len(NUM_CENTERS))}]")
        print(f"Avaliando {n_center} e {treshold}")
        pastas = kfold.split(train_data)
        matrizes_rbf = analise_classificacao.analisa_rbf(
                                        train_data, 
                                        pastas, 
                                        treshold = treshold,
                                        number_of_centers = n_center)
        
        identificacao = f"RBF_{n_center}_{treshold}"
        avaliacao = funcoes_auxiliares.avaliacoes_para_tabela(matrizes_rbf, identificacao)
        avaliacao.to_csv(f"outputs/avaliacao_{identificacao}_{horario_str}.csv")
        print("Resultado salvo.")

    print("Fim parte 1")

    
# Analise 3: Número de centros:
RBF_TRESHOLD = [0.3]
NUM_CENTERS = [40, 100, 500, 1000, 2000, 4000]
N_SPLITS = 5
kfold = sklearn.model_selection.KFold(n_splits=N_SPLITS, shuffle = True, random_state = SEED)
for i, n_center in enumerate(NUM_CENTERS):
    for i2, treshold in enumerate(RBF_TRESHOLD):
        print(f"Iteração [{1 + i*len(RBF_TRESHOLD) + i2} de {len(RBF_TRESHOLD * len(NUM_CENTERS))}]")
        print(f"Avaliando {n_center} e {treshold}")
        pastas = kfold.split(train_data)
        matrizes_rbf = analise_classificacao.analisa_rbf(
                                        train_data, 
                                        pastas, 
                                        treshold = treshold,
                                        number_of_centers = n_center)
        
        identificacao = f"RBF_{n_center}_{treshold}"
        avaliacao = funcoes_auxiliares.avaliacoes_para_tabela(matrizes_rbf, identificacao)
        avaliacao.to_csv(f"outputs/avaliacao_{identificacao}_{horario_str}.csv")
        print("Resultado salvo.")

    print("Fim parte 0")


    
# Analise 4: Número de centros com o Treshold mais fino
RBF_TRESHOLD = [0.27]
NUM_CENTERS = [40, 400, 450, 500, 550, 600, 4000]
N_SPLITS = 5
kfold = sklearn.model_selection.KFold(n_splits=N_SPLITS, shuffle = True, random_state = SEED)
for i, n_center in enumerate(NUM_CENTERS):
    for i2, treshold in enumerate(RBF_TRESHOLD):
        print(f"Iteração [{1 + i*len(RBF_TRESHOLD) + i2} de {len(RBF_TRESHOLD * len(NUM_CENTERS))}]")
        print(f"Avaliando {n_center} e {treshold}")
        pastas = kfold.split(train_data)
        matrizes_rbf = analise_classificacao.analisa_rbf(
                                        train_data, 
                                        pastas, 
                                        treshold = treshold,
                                        number_of_centers = n_center)
        
        identificacao = f"RBF_{n_center}_{treshold}"
        avaliacao = funcoes_auxiliares.avaliacoes_para_tabela(matrizes_rbf, identificacao)
        avaliacao.to_csv(f"outputs/avaliacao_{identificacao}_{horario_str}.csv")
        print("Resultado salvo.")

    print("Fim parte 4")
    print(f"Horário: {horario_str}")
