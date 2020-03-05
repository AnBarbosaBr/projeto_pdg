import pandas as pd 
import numpy as np

import sklearn.preprocessing
import sklearn.model_selection

# In[]

VARIAVEIS_NUMERICAS =  ["KIDSDRIV","AGE","HOMEKIDS","YOJ","INCOME","HOME_VAL","TRAVTIME", "BLUEBOOK","TIF","OLDCLAIM","MVR_PTS","CAR_AGE"]
VARIAVEIS_MULTI_CATEGORICAS = ["GENDER", "EDUCATION", "OCCUPATION", "CAR_USE", "CAR_TYPE", "URBANICITY"]
VARIAVEIS_BINARIAS = ["PARENT1", "MSTATUS", "RED_CAR", "REVOKED"]
VARIAVEIS_ALVO = ["CLM_FREQ","CLM_AMT","CLAIM_FLAG"]

VARIAVEIS_DINHEIRO = ["INCOME","HOME_VAL","BLUEBOOK","OLDCLAIM", "CLM_AMT"]
VARIAVEIS_NUMERICAS_COM_MISSING =  ["AGE", "YOJ", "INCOME","HOME_VAL", "BLUEBOOK","OLDCLAIM", "CAR_AGE"]
VARIAVEIS_CATEGORICAS_COM_MISSING = ["OCCUPATION"]

# In[]
def pre_000_remove_colunas_desnecessarias(dados):
    data = dados.copy()

    colunas_a_remover = ["ID", "BIRTH"]
    return data.drop(colunas_a_remover, axis = 1)

def pre_001_dinheiro_para_numerico(dados):
    data = dados.copy()
    # Converte Dinheiro para numérico        
    data.loc[ : , VARIAVEIS_DINHEIRO] = data.loc[ : , VARIAVEIS_DINHEIRO].replace('[\$,]', '', regex=True).astype(float)

    return data

def pre_002_remove_missing(dados):
    data = dados.copy()
    numericas = data.loc[: , VARIAVEIS_NUMERICAS_COM_MISSING]
    numericas.fillna(numericas.mean(), inplace = True)
    
    categoricas = data.loc[: , VARIAVEIS_CATEGORICAS_COM_MISSING].fillna("Desconhecido")
    
    data.loc[ : , VARIAVEIS_NUMERICAS_COM_MISSING] = numericas
    data.loc[ : , VARIAVEIS_CATEGORICAS_COM_MISSING] = categoricas
    return data

def pre_003_transforma_flags(dados):
    data = dados.copy()
    data.loc[: , "PARENT1"].replace("No", 0, inplace = True)
    data.loc[: , "PARENT1"].replace("Yes", 1, inplace = True)

    data.loc[: , "MSTATUS"].replace("z_No", 0, inplace = True)
    data.loc[: , "MSTATUS"].replace("Yes", 1, inplace = True)

    data.loc[: , "RED_CAR"].replace("no", 0, inplace = True)
    data.loc[: , "RED_CAR"].replace("yes", 1, inplace = True)

    data.loc[: , "REVOKED"].replace("No", 0, inplace = True)
    data.loc[: , "REVOKED"].replace("Yes", 1, inplace = True)

    return data
    
def pre_004_one_hot_encoding(dados):
    data = dados.copy()
    categoricas = data.loc[: , VARIAVEIS_MULTI_CATEGORICAS]
    encoder = sklearn.preprocessing.OneHotEncoder(
        categories = "auto", 
        drop = "first", 
        handle_unknown = "error", 
        sparse = False)
    
    encoder.fit(categoricas)

    
    categoricas_encodadas = encoder.transform(categoricas)
    df_categoricas = pd.DataFrame(categoricas_encodadas)
    df_categoricas.columns = encoder.get_feature_names()
    
    data.drop(VARIAVEIS_MULTI_CATEGORICAS, axis = 1, inplace=True)
    data = data.join(df_categoricas)
    
    return data

def pre_005_normalizando_dados(dados):
    '''Precisa ser testada'''
    data = dados.copy()
    reescaladas = sklearn.preprocessing.scale(data.loc[ : , VARIAVEIS_NUMERICAS])
    data.loc[ : , VARIAVEIS_NUMERICAS] = reescaladas
    return data

def pre_010_log_dinheiro(dados):
    data = dados.copy()
    data.loc[ : , VARIAVEIS_DINHEIRO] = np.log10(data.loc[ : , VARIAVEIS_DINHEIRO]+0.001)
    return data

def pre_process(data):
    etapas = list()
    data_pre = pre_000_remove_colunas_desnecessarias(data)
    etapas.append("remove_colunas")
    data_pre = pre_001_dinheiro_para_numerico(data_pre)
    etapas.append("dinheiro_para_numericas")
    data_pre = pre_002_remove_missing(data_pre)
    etapas.append("substitui_missings")
    data_pre = pre_003_transforma_flags(data_pre)
    etapas.append("transforma_flags_para_1_0")
    data_pre = pre_004_one_hot_encoding(data_pre)
    etapas.append("one_hot_encoding")
    data_pre = pre_005_normalizando_dados(data_pre)
    etapas.append("normalizacao_com_desvios_padrao")
    return data_pre, etapas
    

def pre_simples(data):
    etapas = list()
    data_pre = pre_000_remove_colunas_desnecessarias(data)
    etapas.append("remove_colunas")
    data_pre = pre_001_dinheiro_para_numerico(data_pre)
    etapas.append("dinheiro_para_numericas")
    data_pre = pre_002_remove_missing(data_pre)
    etapas.append("substitui_missings")
    data_pre = pre_003_transforma_flags(data_pre)
    etapas.append("transforma_flags_para_1_0")
    data_pre = pre_004_one_hot_encoding(data_pre)
    etapas.append("one_hot_encoding")
    return data_pre, etapas
        
def pre_normalizado(data):
    data_pre, etapas = pre_simples(data)    
    etapas.append("Normalização Com Desvios Padrão")
    data_pre = pre_005_normalizando_dados(data_pre)
    return data_pre, etapas

def pre_logDinheiro(data):
    data_pre, etapas = pre_simples(data)    
    etapas.append("Log Dinheiro")
    data_pre = pre_010_log_dinheiro(data_pre)
    return data_pre, etapas

def pre_pca(data):
    '''TODO'''
    raise NotImplementedError
