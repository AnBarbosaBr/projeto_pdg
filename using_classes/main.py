import pandas as pd
import numpy as np
import pipelines
import defaultdict
import namedtuple


Avaliacao = namedtuple("Avaliacao", ["criterio", "treino","teste"])
SaidaAnalise = namedtuple("SaidaAnalise", ["values_actual", "values_predicted"])

class AbstractAnalysis(object):
    TREE_ALGORITHM = "Árvore"
    GLM_ALGORITHM = "GLM"
    RBFN_ALGORITHM = "RBFN"
    def __init__(self, dataset = "../data/car_insurance_clain_train.csv", algorithms_to_use = [TREE_ALGORITHM, GLM_ALGORITHM, RBFN_ALGORITHM], seed = 42):
        # If dataset is None, it will be the default value.
        self.dataset = dataset or "../data/car_insurance_clain_train.csv" 
        self.datapipe = pipelines.DataPipeline(dataset)
        self.algorithms_to_use = algorithms_to_use
        self.seed = 42

        self.preprocessing_functions = {"Simples": data_pipeline.pre_simples,
                                "Normalizado": data_pipeline.pre_normalizado,
                                "LogDinheiro": data_pipeline.pre_logDinheiro}


        self.y_variables = ["CLM_FREQ","CLM_AMT","CLAIM_FLAG"]
        
        self.tarefa = None
        self.target_variable = None

        


    
    
    def preprocess(self, tipo : str):
        if tipo in self.preprocessing_functions.keys():
            return self.preprocessing_functions[tipo](self.raw_data)
        else:
            raise ValueError(f"tipo({tipo}) não encontrado como função de préprocessamento.")
    
    def run_tests_with_until(self, with_preprocessing : str, until_folds : int):
        '''Faz os testes usando diferentes números de folds.
        Parameters
        ----------
        preprocessing: Nome do tipo de preprocessamento usado.
        folds: Rode os testes Número de folds a usar.'''
        results = list()
        for k in range(until_folds):
            analysis_output = self.do_analysis(n_folds = k, preprocessing = with_preprocessing)
            results.append(values_analysis, values_actual)
        
        return results
    
    def do_analysis(n_folds, preprocessing):
        data, preprocess_list = self.preprocess(preprocessing)
        X = data.drop(self.y_variables, axis = 1)
        y = data.loc[ : , self.target_variable]
        kfold = sklearn.model_selection.KFold(n_splits=n_folds, shuffle = True, random_state = self.seed)

        
        




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
