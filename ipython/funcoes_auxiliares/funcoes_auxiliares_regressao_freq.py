import pandas as pd
import sklearn.metrics
from collections import namedtuple

def avalia_modelo(y_treino, y_teste, treino_previsto, teste_previsto):
        return _avalia_mse(y_treino, y_teste, treino_previsto, teste_previsto)

def avaliacoes_para_tabela(lista_avaliacoes, titulo):
        return _mse_para_tabela(lista_avaliacoes, titulo)


def _avalia_mse(y_treino, y_teste, treino_previsto, teste_previsto):
        mse_treino = sklearn.metrics.mean_squared_error(y_true = y_treino, y_pred = treino_previsto)
        mse_teste = sklearn.metrics.mean_squared_error(y_true = y_teste, y_pred = teste_previsto)
        
        ResultadoRegressao = namedtuple("ResultadoRegressao", ["treino","teste"])

        return ResultadoRegressao(treino = mse_treino,
                                          teste = mse_teste)

def _avalia_poisson_deviance(y_treino, y_teste, treino_previsto, teste_previsto):
        # Usado pois a função mean_poisson_deviance exige que y_pred > 0
        FATOR_CORRECAO_DEVIANCE = 0.001
        treino_previsto_corrigido = treino_previsto + FATOR_CORRECAO_DEVIANCE
        teste_previsto_corrigido = teste_previsto   + FATOR_CORRECAO_DEVIANCE

        mean_poisson_deviance_treino = sklearn.metrics.mean_poisson_deviance(y_true = y_treino, y_pred = treino_previsto_corrigido)
        mean_poisson_deviance_teste = sklearn.metrics.mean_poisson_deviance(y_true = y_teste, y_pred = teste_previsto_corrigido)

        ResultadoRegressao = namedtuple("ResultadoRegressao", ["treino","teste"])
        return ResultadoRegressao(
                        treino = mean_poisson_deviance_treino,
                        teste = mean_poisson_deviance_teste)

                

def _mse_para_tabela(lista_avaliacoes, titulo):
        avaliacoes_df = pd.DataFrame()
        for i, avaliacao in enumerate(lista_avaliacoes):
                erro_treino = avaliacao.treino
                erro_teste = avaliacao.teste
                data = {"Nome": [titulo, titulo],
                        "Tipo": ["Treino","Teste"],
                        "Iteracao": [i, i],
                        "MSE": [erro_treino, erro_teste],
                        }
                avaliacao_df = pd.DataFrame(data = data)
                avaliacoes_df = pd.concat([avaliacoes_df, avaliacao_df], ignore_index = True, axis = "index")
        return avaliacoes_df.sort_values(by = ["Nome","Tipo","Iteracao"], ignore_index = True)

def _mse_e_deviance_para_tabela(lista_avaliacoes, titulo):
        avaliacoes_df = pd.DataFrame()
        for i, avaliacao in enumerate(lista_avaliacoes):
                for criterio, resultado in avaliacao.items():
                        erro_treino = resultado.treino
                        erro_teste = resultado.teste
                        data = {"Nome": [titulo, titulo],
                                "Tipo": ["Treino","Teste"],
                                "Criterio": criterio,
                                "Iteracao": [i, i],
                                "Value": [erro_treino, erro_teste],
                                }
                        avaliacao_df = pd.DataFrame(data = data)
                        avaliacoes_df = pd.concat([avaliacoes_df, avaliacao_df], ignore_index = True, axis = "index")
        return avaliacoes_df.sort_values(by = ["Nome","Criterio","Tipo","Iteracao"], ignore_index = True)
