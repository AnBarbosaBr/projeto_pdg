# In[]
# Importacoes
import pandas as pd
import os
from funcoes_auxiliares.funcoes_concatenacao import *

# In[] Constantes
GRID_SEARH = "gridsearch"
MODELO = "modelos"
REGRESSAO = "regressao"
CLASSIFICACAO_FLAG = "classificacao"

GLM = "glm"
RBF = "rbf"

# In[] Parametros

tarefa = REGRESSAO
modo = MODELO
algoritmo = None # GLM # RBF

pasta_original = "outputs"
output_name = "avaliacao_"
data_hora_desejada = "14h25m54s"

# In[] Executando rotina
full_df = concatena_arquivos(data_hora_desejada, pasta = pasta_original)
full_df.drop("Iteracao", axis="columns", inplace = True, errors = "ignore")

if(full_df.shape[0] == 0):
    raise Exception(f"Não foi possível encontrar nenhum arquivo para a hora desejada: {data_hora_desejada}.")
resumo = agrupa_modelos(full_df)

if tarefa == CLASSIFICACAO_FLAG:
    resumo = enriquece_matriz_de_confusao(resumo)

try:
    os.mkdir(f"outputs/{data_hora_desejada}")
except OSError:
    print("Assumindo que o diretorio de saida já existe...")


if modo == GRID_SEARH:
    print("Modo Grid Search")
    if(algoritmo == RBF):
        relatorio_grid_search_rbf(resumo, f"outputs/{data_hora_desejada}/grid_search_resumo")
    elif(algoritmo == GLM):
        relatorio_grid_search_glm(resumo, f"outputs/{data_hora_desejada}/grid_search_resumo")

elif modo == MODELO:
    print("Modo análise de modelos")
    resumo.to_csv(f"outputs/{data_hora_desejada}/{output_name}_{modo}_{tarefa}.csv", sep = ";", decimal = ",", encoding = "utf-8-sig")
print("Fim")

# %%
