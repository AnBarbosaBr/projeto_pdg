import sklearn.tree
import statsmodels.api as sm
from funcoes_auxiliares import funcoes_auxiliares_classificacao
import RBF



TARGETS = ["CLM_FREQ","CLM_AMT","CLAIM_FLAG"]
THIS_TARGET = ["CLAIM_FLAG"]


def analisa_arvore(dados, folds, treshold = None):
    X_data = dados.drop(TARGETS, axis = 1)
    y_data = dados.loc[ : , THIS_TARGET]

    avaliacoes = list()
    for i, (train_index, test_index) in enumerate(folds):
        print(f"Árvore, iteração {i}")
        X_treino = X_data.iloc[train_index, : ]
        y_treino = y_data.iloc[train_index]

        X_teste = X_data.iloc[test_index, : ]
        y_teste = y_data.iloc[test_index]

        model = sklearn.tree.DecisionTreeClassifier()
        model.fit(X_treino, y_treino)

        treino_previsto = model.predict(X_treino)
        teste_previsto = model.predict(X_teste)

        avaliacao = funcoes_auxiliares_classificacao.avalia_modelo(y_treino, y_teste, treino_previsto, teste_previsto)
        avaliacoes.append(avaliacao)
    return avaliacoes


def analisa_glm(dados, folds, treshold):
    X_data = dados.drop(TARGETS, axis = 1)
    y_data = dados.loc[ : , THIS_TARGET]

    avaliacoes = list()
    for i, (train_index, test_index) in enumerate(folds):
        print(f"GLM, iteração {i}")
        X_treino = X_data.iloc[train_index, : ]
        y_treino = y_data.iloc[train_index]

        X_teste = X_data.iloc[test_index, : ]
        y_teste = y_data.iloc[test_index]

        exog_data = X_treino
        endog_data = y_treino
        familia = sm.families.Binomial()
        model = sm.GLM(exog = exog_data,
                       endog = endog_data,
                       family = familia)

        glm = model.fit()

        probabilidade_treino = glm.predict(X_treino)
        probabilidade_teste = glm.predict(X_teste)

        treino_previsto = probabilidade_treino >= treshold
        teste_previsto = probabilidade_teste >= treshold

        avaliacao = funcoes_auxiliares_classificacao.avalia_modelo(y_treino, y_teste, treino_previsto, teste_previsto)
        avaliacoes.append(avaliacao)
    return avaliacoes


def analisa_rbf(dados, folds, treshold, number_of_centers):
    X_data = dados.drop(TARGETS, axis = 1)
    y_data = dados.loc[ : , THIS_TARGET]

    avaliacoes = list()
    for i, (train_index, test_index) in enumerate(folds):
        print(f"RBF, iteração {i}")
        X_treino = X_data.iloc[train_index, : ]
        y_treino = y_data.iloc[train_index]

        X_teste = X_data.iloc[test_index, : ]
        y_teste = y_data.iloc[test_index]

        model = RBF.RBFClassifier(number_of_centers = number_of_centers, 
                                  algorithm = sklearn.linear_model.LinearRegression(), 
                                  random_state=42)
        model.fit(X_treino, y_treino)

        probabilidade_treino = model.predict(X_treino)
        probabilidade_teste = model.predict(X_teste)

        treino_previsto = probabilidade_treino >= treshold
        teste_previsto = probabilidade_teste >= treshold

        avaliacao = funcoes_auxiliares_classificacao.avalia_modelo(y_treino, y_teste, treino_previsto, teste_previsto)
        avaliacoes.append(avaliacao)
    return avaliacoes

