import pandas as pd
import sklearn.metrics
from collections import namedtuple


def avalia_modelo(y_treino, y_teste, treino_previsto, teste_previsto):
        matriz_confusao_treino = sklearn.metrics.confusion_matrix(y_true = y_treino, y_pred = treino_previsto)
        matriz_confusao_teste = sklearn.metrics.confusion_matrix(y_true = y_teste, y_pred = teste_previsto)
        Resultado = namedtuple("Resultados", ["treino","teste"])

        return Resultado(treino = matriz_confusao_treino,
                        teste = matriz_confusao_teste)


def avaliacoes_para_tabela(lista_avaliacoes, titulo):
        avaliacoes_df = pd.DataFrame()
        for i, avaliacao in enumerate(lista_avaliacoes):
                treino_tn, treino_fp, treino_fn, treino_tp = avaliacao.treino.ravel()
                teste_tn, teste_fp, teste_fn, teste_tp = avaliacao.teste.ravel()
                
                data = {"Nome": [titulo, titulo],
                        "Tipo": ["Treino","Teste"],
                        "Iteracao": [i, i],
                        "True_Negative": [treino_tn, teste_tn],
                        "False_Positive": [treino_fp, teste_fp],
                        "False_Negative": [treino_fn, teste_fn],
                        "True_Positive": [treino_tp, teste_tp]}
                avaliacao_df = pd.DataFrame(data = data)
                avaliacoes_df = pd.concat([avaliacoes_df, avaliacao_df], ignore_index = True, axis = "index")
        # Only work on some versions of pandas
        return avaliacoes_df.sort_values(by = ["Nome","Tipo","Iteracao"], ignore_index = True)
        # return avaliacoes_df.sort_values(by = ["Nome","Tipo","Iteracao"])

