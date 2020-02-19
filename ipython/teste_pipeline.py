import pandas as pd;
import numpy as np; 

import funcoes_auxiliares
import data_pipeline
###############
# # Preparacao
###############

SEED = 42

data = pd.read_csv("../projeto_pdg/data/car_insurance_claim.csv")


# Análise do Processamento
data.shape
data._get_numeric_data().shape
data_pre = data_pipeline.pre_000_remove_colunas_desnecessarias(data)
data_pre.shape # Removida 1 coluna

data_pre = data_pipeline.pre_001_dinheiro_para_numerico(data_pre)
data_pre._get_numeric_data().shape # Adicionadas 5 colunas numericas

data_pre = data_pipeline.pre_002_remove_missing(data_pre)
data_pre.isna().sum() # Sem valores missings

data_pre = data_pipeline.pre_003_transforma_flags(data_pre)
colunas_flags = ["PARENT1", "MSTATUS","RED_CAR", "REVOKED", "CLAIM_FLAG"]
[print(f"{col}: {data_pre[col].unique()}") for col in colunas_flags]

data_pre = data_pipeline.pre_004_one_hot_encoding(data_pre)

data_pre.isna().sum()
data_pre.columns
data_pre.shape