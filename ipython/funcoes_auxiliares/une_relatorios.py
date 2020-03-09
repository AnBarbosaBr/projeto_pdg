import pandas as pd
import numpy as np
import os
from collections import namedtuple

class RBFParser(object):
    def __init__(self, base_path):
        self.base_path = base_path

    @staticmethod    
    def parsear_nome(nome):
        Detalhes = namedtuple("Detalhes",
                        ["tarefa","alvo","pre_processamento", "n_centroides"])                   
        lista = nome.split("_")
        tarefa = lista[1]
        alvo = lista[2]
        preprocessamento = lista[3]
        n_centroides = lista[8]
        return Detalhes(tarefa = tarefa,
                    alvo = alvo,
                    pre_processamento = preprocessamento,
                    n_centroides = n_centroides)

    def concatena_classificacao(self):
        return self.concatena_detalhes(filtro = "CLASSIFICACAO_FLAG")

    def concatena_valor(self):
        return self.concatena_detalhes(filtro = "REGRESSAO_AMT")

    def concatena_frequencia(self):
        return self.concatena_detalhes(filtro = "REGRESSAO_FREQ")

    def concatena_detalhes(self, filtro):
        if filtro not in ["CLASSIFICACAO_FLAG","REGRESSAO_AMT","REGRESSAO_FREQ"]:
            raise ValueError('Filtro deve ser um desses:\n"CLASSIFICACAO_FLAG","REGRESSAO_AMT","REGRESSAO_FREQ"')
        base_path = self.base_path
        all_files = os.listdir(base_path)
        detalhes = [arquivo for arquivo in all_files if "Detalhes" in arquivo]
        detalhesFilt = [arquivo for arquivo in detalhes if filtro in arquivo]

        final_dataset = list()
        if not detalhesFilt:
            return pd.DataFrame({"Not Found":"Not Found"})

        for nome in detalhesFilt:
            descricao = parsear_nome(nome)
            dataset = pd.read_csv(os.path.join(base_path, nome))
            dataset.rename(columns = {"Nome": "Limiar"}, inplace = True)
            dataset.insert(loc = 1, column = "N_Centroides", value = descricao.n_centroides)
            dataset.insert(loc = 3, column = "Tarefa", value = descricao.tarefa)
            dataset.insert(loc = 4, column = "Alvo", value = descricao.alvo)
            dataset.insert(loc = 5, column = "PreProcessamento", value = descricao.pre_processamento)
            final_dataset.append(dataset)
        
        finalDF = pd.concat(final_dataset)
        return finalDF

    @statimethod
    def summary_df(dataframe):
        return dataframe.groupby(by=["Limiar", "N_Centroides", "Tipo", "Tarefa","Alvo", "PreProcessamento"]).agg("sum").drop("Iteracao", axis="columns").reset_index()
    
    def salva_resumo(self, output_path, classificacao = True, valor = True, frequencia = True):
        if (classificacao or valor or frequencia):
            if not os.path.exists(output_path):
                os.makedirs(output_path, exist_ok = True)
        if classificacao:
            classDF = self.concatena_classificacao()
            classDF.to_csv(os.path.join(output_path, "busca_rbf_classificacao.csv"), index = False, encoding = "utf-8-sig")
            summary_df(classDF).to_csv(os.path.join(output_path, "busca_rbf_classificacao_resumo.csv"), index = False, encoding = "utf-8-sig")

        if valor:
            valorDF = self.concatena_valor()
            valorDF.to_csv(os.path.join(output_path, "busca_rbf_valor.csv"), index = False, encoding = "utf-8-sig")
            summary_df(valorDF).to_csv(os.path.join(output_path, "busca_rbf_valor_resumo.csv"), index = False, encoding = "utf-8-sig")
        if frequencia:
            freqDF = self.concatena_frequencia()
            freqDF.to_csv(os.path.join(output_path, "busca_rbf_frequencia.csv"), index = False, encoding = "utf-8-sig")
            summary_df(freqDF).to_csv(os.path.join(output_path, "busca_rbf_frequencia_resumo.csv"), index = False, encoding = "utf-8-sig")


if __name__=="__main__":
    base_path = "../outputs/2020_03_08/BuscaParametros"
    
    rbfParser = RBFParser(base_path)
    rbfParser.salva_resumo("../outputs/2020_03_08/RBF_Search")
    teste.concatena_classificacao().groupby(by=["Limiar", "N_Centroides", "Tipo", "Tarefa","Alvo", "PreProcessamento"]).agg("sum").drop("Iteracao", axis="columns").reset_index().filter(Tipo="Teste")
    teste.concatena_valor().groupby(by=["Limiar", "N_Centroides", "Tipo", "Tarefa","Alvo", "PreProcessamento"]).agg("sum").drop("Iteracao", axis="columns").reset_index().query("Tipo=='Teste'")
    teste.concatena_frequencia().groupby(by=["Limiar", "N_Centroides", "Tipo", "Tarefa","Alvo", "PreProcessamento"]).agg("sum").drop("Iteracao", axis="columns").reset_index().query("Tipo=='Teste'")