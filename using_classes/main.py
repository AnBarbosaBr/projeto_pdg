import pandas as pd
import numpy as np
import pipelines
from collections import defaultdict, namedtuple
import sklearn.model_selection

teste = pipelines.DataPipeline("../data/car_insurance_clain_train.csv")
folds = sklearn.model_selection.KFold(10, shuffle = True, random_state=42)
treino, teste = next(folds.split(teste.getXdata(), teste.getYdata(teste.Y_AMT)))





Avaliacao = namedtuple("Avaliacao", ["criterio", "treino","teste"])
SaidaAnalise = namedtuple("SaidaAnalise", ["values_actual", "values_predicted"])

class Resultado(object):
    def __init__(self, model, parameter):
        self.model = model
        self.parameter = parameter
        
        self.values_predicted = {"teste" : [], "treino": []}
        self.values_real = {"teste":[], "treino": []}
        
    
    def addValorY(self, tipo, values):
        if tipo not in ["teste","treino"]:
            raise ValueError(f"Tipo deve ser 'teste' ou 'treino'. Fornecido {tipo}")
            self.value_real[tipo].append(values)

    def addPrevisao(self, tipo, values):
        if tipo not in ["teste","treino"]:
            raise ValueError(f"Tipo deve ser 'teste' ou 'treino'. Fornecido {tipo}")
        self.values_predicted[tipo].append(values) 
    
class Resultados(object):
    def __init__(self):
        self.resultados = defaultdict(lambda : [])
    
    def addResultado(iteracao, modelo, resultado):
        self.resultados[iteracao].append(resultado)

class Model(object):
    def __init__(self, nome):
        self.name = nome
        self.parameters = {}
        
    def fit_predict(self, X_train, y_train, X_test):
        raise NotImplementedError
    

class AbstractAnalysis(object):
    TREE_ALGORITHM = "Árvore"
    GLM_ALGORITHM = "GLM"
    RBFN_ALGORITHM = "RBFN"
    def __init__(self, models, scorers, dataset = "../data/car_insurance_clain_train.csv", seed = 42):
        # If dataset is None, it will be the default value.
        self.dataset = dataset or "../data/car_insurance_clain_train.csv" 
        self.models = models,
        self.scorers = scorers,

        self.datapipeline = pipelines.DataPipeline(dataset)
        self.seed = 42

        self.preprocessings = ["Simples", "Normalizado", "LogDinheiro"]
        
        self.tarefa = None
        self.target_variable = None

            
    def preprocess(self, tipo : str):
        if(tipo == "Simples"):
            datapipeline.pre_simples()
        elif(tipo == "Normalizado"):
            datapipeline.pre_normalizado()
        elif(tipo == "LogDinheiro"):
            datapipeline.pre_logDinheiro()
    
    def run_tests_with_until(self, with_preprocessing : str, until_folds : int):
        '''
        Faz os testes usando diferentes números de folds.
        Parameters
        ----------
        preprocessing: Nome do tipo de preprocessamento usado.
        folds: Rode os testes Número de folds a usar: "Simples","Normalizado","LogDinheiro".
        '''

        results = list()
        for k in range(until_folds):
            analysis_output = self.do_analysis(n_folds = k, preprocessing = with_preprocessing)
            results.append(analysis_output)

        return results
    
    def do_analysis(self, n_folds, preprocessing):
        self.preprocess(preprocessing)
        data = self.pipeline.data_out
        X = self.pipeline.getXdata()
        y = self.pipeline.getYdata(target = self.target_variable)
        kfold = sklearn.model_selection.KFold(n_splits=n_folds, shuffle = True, random_state = self.seed)

        pastas = kfold.split(X, y)
        resultados = list()
        for i, (train_index, test_index) in enumerate(pastas):
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]

            X_test = X.iloc[test_index]
            y_test = y.iloc[test_index]

            models_results = Resultados()
            for model in self.models:
                resultado = self.run_model(model, X_train, y_train, X_test, y_test)           
                models_results.addResultado(i, resultado)

            resultados.append(models_results)
        return resultados

    

    def run_model(self, model, X_train, y_train, X_test, y_test):
        resultado = Resultado(model.name, model.parameters)
        resultado.addValorY("teste", y_test)
        resultado.addValorY("treino", y_treino)
        
        previsao_treino, previsao_teste = model.fit_predict(X_train, y_train, X_test)
        resultado.addPrevisao("treino", previsao_treino)
        resultado.addPrevisao("teste", previsao_teste)
        return resultado

        




class ClaimFlagAnalysis(AbstractAnalysis):
    def __init__(self, dataset = None):
        super().__init__(dataset)
        self.tarefa = "CLASSIFICACAO"
        self.target_variable = "CLAIM_FLAG"

    

class ClaimFrequencyAnalysis(AbstractAnalysis):
    def __init__(self, dataset = None):
        super().__init__(dataset)
        self.tarefa = "RegressaoFrequencia"
        self.target_variable = "CLM_FREQ"

class ClaimAmountAnalysis(AbstractAnalysis):
    def __init__(self, dataset = None):
        super().__init__(dataset)
        self.tarefa = "RegressaoValor"
        self.target_variable = "CLM_AMT"
