#!/usr/bin/env python
# coding: utf-8

#################
# # Projeto PDG
#################
import pandas as pd;
import numpy as np; 
import matplotlib.pyplot as plt;


import funcoes_auxiliares
import data_pipeline
###############
# # Preparacao
###############

SEED = 42

data = pd.read_csv("../../data/car_insurance_claim.csv")



# Linhas / Colunas
forma = data.shape
print(f"{data.shape[0]} linhas e {data.shape[1]} colunas")
print(pd.DataFrame(data.columns))
print(data.describe())
print(data.describe().shape[1]) # Colunas Númericas


# Imprime tipos das colunas
[print(f"{col}: {data[col].dtype}") for col in data.columns]

# Imprime "Enuns"
[print(f"{col}: {data[col].unique()}") for col in data.columns if len(data[col].unique()) < 10]


# Imprime amostra para valores que não são "Enuns"
[print(f"{col}: [{data[col].iloc[0]}, {data[col].iloc[-1]}]") for col in data.columns if len(data[col].unique()) >= 10]

# Descrição dos dados numericos
data.describe().to_csv("describe.csv")

# Verificando Variáveis Nulas
data.isnull().sum()

# Quantidade de Linhas nulas: 2645
linhas_nulas = data.shape[0] - data.dropna().shape[0]
linhas_nulas/data.shape[0]

# Variaveis Alvo - Distribuição
data.CLAIM_FLAG.value_counts()
data.REVOKED.value_counts()
7556-1261 # 
data.CLM_FREQ.value_counts()
data.CLM_AMT = data.CLM_AMT.replace('[\$,]', '', regex=True).astype(float)
data.CLM_AMT

sum(data.CLM_FREQ > 0) # 4010
4010 - 1261 # = 2749, vs 2746 CLAIM_FLAG = TRUE

data.query("CLAIM_FLAG == 0 & CLM_AMT > 0")
data[["CLM_FREQ", "CLM_AMT", "REVOKED", "CLAIM_FLAG"]].query("REVOKED == 'Yes'")