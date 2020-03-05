import pandas as pd
import sklearn.metrics
from collections import namedtuple

def avalia_modelo(y_treino, y_teste, treino_previsto, teste_previsto):
        squared_error_treino = sklearn.metrics.mean_poisson_deviance(y_true = y_treino, y_pred = treino_previsto)
        squared_error_teste = sklearn.metrics.mean_poisson_deviance(y_true = y_teste, y_pred = teste_previsto)
        ResultadoRegressao = namedtuple("ResultadoRegressao", ["treino","teste"])
        return ResultadoRegressao(
                        treino = squared_error_treino,
                        teste = squared_error_teste)

def avaliacoes_para_tabela(lista_avaliacoes, titulo):
        avaliacoes_df = pd.DataFrame()
        for i, avaliacao in enumerate(lista_avaliacoes):
                erro_treino = avaliacao.treino
                erro_teste = avaliacao.teste
                data = {"Nome": [titulo, titulo],
                        "Tipo": ["Treino","Teste"],
                        "Iteracao": [i, i],
                        "PoissonDeviance": [erro_treino, erro_teste],
                        }
                avaliacao_df = pd.DataFrame(data = data)
                avaliacoes_df = pd.concat([avaliacoes_df, avaliacao_df], ignore_index = True, axis = "index")
        return avaliacoes_df.sort_values(by = ["Nome","Tipo","Iteracao"], ignore_index = True)

