
#################
# # Projeto PDG
#################
import os
import pandas as pd
import numpy as np
import datetime
from funcoes_auxiliares import funcoes_auxiliares_classificacao
from funcoes_auxiliares import funcoes_auxiliares_regressao_freq
from funcoes_auxiliares import funcoes_auxiliares_regressao_valor

class ReportGenerator(object):
    def __init__(self, dict_matrizes_resultado, 
    preprocessamentos = None, parametros = None,
    nome_preprocessamento = None):
        self.tipo_report = "ABSTRACT_REPORT"
        self.timestamp_format = '%Y%m%d_%Hh%Mm%Ss'
        self.timestamp = datetime.datetime.today().strftime(self.timestamp_format)
        self.dict_matrizes_resultado = dict_matrizes_resultado
        self.preprocessamentos = preprocessamentos
        self.nome_preprocessamento = nome_preprocessamento
        self.parametros = parametros

    def generate_report(self, output_path, identificador = None):
        reportFull = self.concatena_avaliacoes()
        reportSummary = self.sumariza_avaliacoes(reportFull)
        reportPreprocessamento = self.get_df_preprocessamentos()
        reportParametros = self.get_df_parametros()

        titulo = self.generate_title(identificador)

        print(f"Salvando relatório {titulo}")
        self.salva_relatorio(titulo = titulo,
                            output_path = output_path,
                            preprocessamento = reportPreprocessamento,
                            parametros = reportParametros,
                            resultados = reportFull,
                            sumarios = reportSummary)

    def concatena_avaliacoes(self):
        raise NotImplementedError

    def sumariza_avaliacoes(self, avaliacoes):
        raise NotImplementedError

    def get_df_preprocessamentos(self):
        return pd.DataFrame(self.preprocessamentos)

    def get_df_parametros(self):
        unrolled_params = dict()
        for algoritmo, algo_params in self.parametros.items():
            for nome_param, value in algo_params.items():
                unrolled_params[(algoritmo, nome_param)] = value
        parametrosDF = pd.DataFrame.from_dict(unrolled_params, orient = "index", columns = ["Valor"])
        parametrosDF.index = pd.MultiIndex.from_tuples(parametrosDF.index, names =("Algoritmo","Parâmetro"))
        return parametrosDF.reset_index()


    def get_df_parametros_stub(self):
        return pd.DataFrame({"Algoritmo": ["Não Implementado"],
                         "Parâmetro": ["Não Implementado"],
                         "Valor": ["Não Implementado"]})
    
    def generate_title(self, elementos = None):
        titulo = self.tipo_report + "_" + self.nome_preprocessamento + "_" + self.timestamp 
        if not elementos:
            base = titulo
        elif isinstance(elementos, str):
            base = f"{titulo}_{elementos}"
        elif isinstance(elementos, list):
                base = titulo + "_" + "_".join(elementos)
        return base

    @staticmethod
    def _output_path_with_bar(output_path):
        output_path_with_bar = output_path
        if(not output_path_with_bar.endswith("/")):
            output_path_with_bar = output_path_with_bar + "/"
        return output_path_with_bar
    
    @staticmethod
    def salva_relatorio(titulo, output_path,
                    preprocessamento, parametros, resultados, sumarios):
    
        output_with_bar = ReportGenerator._output_path_with_bar(output_path)
        if not os.path.exists(output_with_bar):
            os.makedirs(output_with_bar, exist_ok = True)
        resultados.to_csv(f"{output_with_bar}Detalhes_{titulo}.csv", index = False, encoding = "utf-8-sig")
        sumarios.to_csv(f"{output_with_bar}Sumario_{titulo}.csv", index = False, encoding = "utf-8-sig")
        parametros.to_csv(f"{output_with_bar}Parametros_{titulo}.csv", index = False, encoding = "utf-8-sig")
        preprocessamento.to_csv(f"{output_with_bar}PreProcessamento_{titulo}.csv", index = False, encoding = "utf-8-sig")


class ReportFlag(ReportGenerator):
    def __init__(self, dict_matrizes_resultado, 
    preprocessamentos = None, parametros = None,
    nome_preprocessamento = None):
        # SUPER INITIATOR
        super().__init__(dict_matrizes_resultado, 
                            preprocessamentos, 
                            parametros, 
                            nome_preprocessamento)
        self.tipo_report = "CLASSIFICACAO"

    def concatena_avaliacoes(self):
        columns_details = ["Nome","Tipo","Iteracao", "True_Positive","True_Negative","False_Positive","False_Negative"]
        
        resultados_list = list()
        for modelo, resultado in self.dict_matrizes_resultado.items():
            resultados_list.append(funcoes_auxiliares_classificacao.avaliacoes_para_tabela(resultado, modelo))
        
        
        avaliacoes = pd.concat(resultados_list, axis = "index")[columns_details].reset_index().drop("index",axis="columns")
        avaliacoes = avaliacoes.sort_values(by = ["Tipo","Nome","Iteracao"], ignore_index = True)
        return avaliacoes

    def sumariza_avaliacoes(self, avaliacoes):  
        columns_summary = ["Nome","Tipo", "True_Positive","True_Negative","False_Positive","False_Negative"]
        sumarios = avaliacoes.groupby(["Nome","Tipo"]).sum().drop("Iteracao", axis="columns").reset_index()[columns_summary]
        return sumarios.sort_values(by=["Tipo","Nome"], ignore_index = True)



class ReportValor(ReportGenerator):
    def __init__(self, dict_matrizes_resultado, 
    preprocessamentos = None, parametros = None,
    nome_preprocessamento = None):
        # SUPER INITIATOR
        super().__init__(dict_matrizes_resultado, 
                        preprocessamentos, 
                        parametros, 
                        nome_preprocessamento)
        self.tipo_report = "REGRESSAO_AMT"
        
    def concatena_avaliacoes(self):
        columns_details = ["Nome","Tipo","Iteracao", "MSE"]
        
        resultados_list = list()
        for modelo, resultado in self.dict_matrizes_resultado.items():
            resultados_list.append(funcoes_auxiliares_regressao_valor.avaliacoes_para_tabela(resultado, modelo))
        
        avaliacoes = pd.concat(resultados_list, axis = "index")[columns_details].reset_index().drop("index",axis="columns")
        avaliacoes = avaliacoes.sort_values(by = ["Tipo","Nome","Iteracao"], ignore_index = True)
        return avaliacoes

    def sumariza_avaliacoes(self, avaliacoes):  
        columns_summary = ["Nome","Tipo", "MSE"]
        sumarios = avaliacoes.groupby(["Nome","Tipo"]).sum().drop("Iteracao", axis="columns").reset_index()[columns_summary]
        return sumarios.sort_values(by=["Tipo","Nome"], ignore_index = True)


class ReportFrequencia(ReportGenerator):
    def __init__(self, dict_matrizes_resultado, 
    preprocessamentos = None, parametros = None,
    nome_preprocessamento = None):
        # SUPER INITIATOR
        super().__init__(dict_matrizes_resultado, 
                        preprocessamentos, 
                        parametros, 
                        nome_preprocessamento)
        self.tipo_report = "REGRECAO_FREQ"
        
    def concatena_avaliacoes(self):
        columns_details = ["Nome","Tipo","Iteracao", "MSE"]

        resultados_list = list()
        for modelo, resultados in self.dict_matrizes_resultado.items():
            resultados_list.append(funcoes_auxiliares_regressao_freq.avaliacoes_para_tabela(resultados, modelo))
        
        avaliacoes = pd.concat(resultados_list, axis = "index")[columns_details].reset_index().drop("index",axis="columns")
        avaliacoes = avaliacoes.sort_values(by = ["Tipo","Nome","Iteracao"], ignore_index = True)
        return avaliacoes

    def sumariza_avaliacoes(self, avaliacoes):  
        columns_summary = ["Nome","Tipo", "MSE"]
        sumarios = avaliacoes.groupby(["Nome","Tipo"]).sum().drop("Iteracao", axis="columns").reset_index()[columns_summary]
        return sumarios.sort_values(by=["Tipo","Nome"], ignore_index = True)
