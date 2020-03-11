import pandas as pd

class DataPipeline(object):  

    VARIAVEIS_DESNECESSARIAS = ["ID", "BIRTH"]
    VARIAVEIS_NUMERICAS =  ["KIDSDRIV","AGE","HOMEKIDS","YOJ","INCOME","HOME_VAL","TRAVTIME", "BLUEBOOK","TIF","OLDCLAIM","MVR_PTS","CAR_AGE"]
    VARIAVEIS_MULTI_CATEGORICAS = ["GENDER", "EDUCATION", "OCCUPATION", "CAR_USE", "CAR_TYPE", "URBANICITY"]
    VARIAVEIS_BINARIAS = ["PARENT1", "MSTATUS", "RED_CAR", "REVOKED"]
    VARIAVEIS_ALVO = ["CLM_FREQ","CLM_AMT","CLAIM_FLAG"]


    VARIAVEIS_DINHEIRO = ["INCOME","HOME_VAL","BLUEBOOK","OLDCLAIM", "CLM_AMT"]
    VARIAVEIS_NUMERICAS_COM_MISSING =  ["AGE", "YOJ", "INCOME","HOME_VAL", "BLUEBOOK","OLDCLAIM", "CAR_AGE"]
    VARIAVEIS_CATEGORICAS_COM_MISSING = ["OCCUPATION"]


    def __init__(self, datapath):
        self.datapath = datapath
        self.data_out = None
        self.last_procedure = "None"
        self.procedures = list()
        self.raw_data = None
        
        self.read_data(datapath)


    def clean_procedures(self):
        self.last_procedure = None
        self.procedures = list()

    def record_procedure(self, procedureID : str, descricao : str) -> None:
        self.last_procedure = procedureID
        self.procedures.append(descricao)

    def read_data(self, raw_path):
        self.clean_procedures()
        self.record_procedure("Leitura da Base", f"Leitura do Arquivo: {raw_path}")
        self.raw_data = pd.read_csv(self.datapath)
        self.data_out = self.raw_data
        return self.raw_data

    def step00_remove_columns(self):
        self.record_procedure(f"Remove Colunas", "Removendo Colunas : {self.VARIAVEIS_DESNECESSARIAS}")
        self.data_out = self.data_out.drop(self.VARIAVEIS_DESNECESSARIAS, axis = 1)

    def step01_money_to_number(self):
        self.record_procedure("Transformando Dinheiro em Número", f"Removendo '$' das variáveis monetárias e convertendo em numéricas: {self.VARIAVEIS_DINHEIRO}")
        data = self.data_out.copy()
        # Converte Dinheiro para numérico        
        data.loc[ : , self.VARIAVEIS_DINHEIRO] = data.loc[ : , self.VARIAVEIS_DINHEIRO].replace('[\$,]', '', regex=True).astype(float)
        self.data_out = data

    def step02_remove_missing(self):
        self.record_procedure(f"Substituindo Missing - Categóricas", f"Substituindo valores missing colunas categóricas. Estratégia: Adicionada categoria 'Desconhecido': {self.VARIAVEIS_CATEGORICAS_COM_MISSING}")
        self.record_procedure(f"Substituindo Missing - Numéricas", f"Substituindo valores missing colunas numéricas. Estratégia: Média: {self.VARIAVEIS_NUMERICAS_COM_MISSING}")
        data = self.data_out.copy()
        numericas = data.loc[: , self.VARIAVEIS_NUMERICAS_COM_MISSING]
        numericas.fillna(numericas.mean(), inplace = True)
        
        categoricas = data.loc[: , self.VARIAVEIS_CATEGORICAS_COM_MISSING].fillna("Desconhecido")
        
        data.loc[ : , self.VARIAVEIS_NUMERICAS_COM_MISSING] = numericas
        data.loc[ : , self.VARIAVEIS_CATEGORICAS_COM_MISSING] = categoricas
        self.data_out = data

    def step03_flags_to_number(self):
        self.record_procedure(f"Homogenizando flags", "Transformando todas as categorias yes/no em 1/0.")
        data = self.data_out.copy()
        data.loc[: , "PARENT1"].replace("No", 0, inplace = True)
        data.loc[: , "PARENT1"].replace("Yes", 1, inplace = True)

        data.loc[: , "MSTATUS"].replace("z_No", 0, inplace = True)
        data.loc[: , "MSTATUS"].replace("Yes", 1, inplace = True)

        data.loc[: , "RED_CAR"].replace("no", 0, inplace = True)
        data.loc[: , "RED_CAR"].replace("yes", 1, inplace = True)

        data.loc[: , "REVOKED"].replace("No", 0, inplace = True)
        data.loc[: , "REVOKED"].replace("Yes", 1, inplace = True)

        self.data_out = data
        
    def step04_one_hot_encoding(self):
        self.record_procedure("OneHotEncoding", f"Realizando OneHotEncoding, removendo uma categoria para ser a 'base': {self.VARIAVEIS_MULTI_CATEGORICAS}.")
        data = self.data_out.copy()
        categoricas = data.loc[: , self.VARIAVEIS_MULTI_CATEGORICAS]
        encoder = sklearn.preprocessing.OneHotEncoder(
            categories = "auto", 
            drop = "first", 
            handle_unknown = "error", 
            sparse = False)
        
        encoder.fit(categoricas)

        
        categoricas_encodadas = encoder.transform(categoricas)
        df_categoricas = pd.DataFrame(categoricas_encodadas)
        df_categoricas.columns = encoder.get_feature_names()
        df_categoricas.index = categoricas.index
        
        data.drop(self.VARIAVEIS_MULTI_CATEGORICAS, axis = 1, inplace=True)
        self.data_out = data.join(df_categoricas)

    
    def step05_normalizando_dados(self):
        self.record_procedure("Normalizando Dados", "Normalizando com sklearn -> scale: {self.VARIAVEIS_NUMERICAS}")
        data = self.data_out.copy()
        reescaladas = sklearn.preprocessing.scale(data.loc[ : , self.VARIAVEIS_NUMERICAS])
        data.loc[ : , self.VARIAVEIS_NUMERICAS] = reescaladas
        self.data_out = data

    
    def step05b_log_dinheiro(self):
        data = self.data_out.copy()
        data.loc[ : , self.VARIAVEIS_DINHEIRO] = np.log10(data.loc[ : , self.VARIAVEIS_DINHEIRO]+0.001)
        self.data_out = data


    def pre_simples(self):
        etapas = list()
        data_pre = step00_remove_colunas_desnecessarias(data)
        etapas.append("remove_colunas")
        data_pre = step01_dinheiro_para_numerico(data_pre)
        etapas.append("dinheiro_para_numericas")
        data_pre = step02_remove_missing(data_pre)
        etapas.append("substitui_missings")
        data_pre = step03_transforma_flags(data_pre)
        etapas.append("transforma_flags_para_1_0")
        data_pre = step04_one_hot_encoding(data_pre)
        etapas.append("one_hot_encoding")
        return data_pre
    
    def pre_normalizado(self):
        data_pre, etapas = pre_simples(data)    
        etapas.append("Normalização Com Desvios Padrão")
        data_pre = step05_normalizando_dados(data_pre)
        return data_pre

    def pre_logDinheiro(self):
        data_pre, etapas = pre_simples(data)    
        etapas.append("Log Dinheiro")
        data_pre = step10_log_dinheiro(data_pre)
        return data_pre
