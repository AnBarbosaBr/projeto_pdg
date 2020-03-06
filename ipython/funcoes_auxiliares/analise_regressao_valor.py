import sklearn.tree
import statsmodels.api as sm
import funcoes_auxiliares.funcoes_auxiliares_regressao_valor as auxiliares
import RBF

'''Funções exportadas
    analisa_arvore(dados, folds)
    analisa_glm(dados, folds, familia)
    analisa_rbf(dados, folds, number_of_centeres)
'''

TARGETS = ["CLM_FREQ","CLM_AMT","CLAIM_FLAG"]
THIS_TARGET = ["CLM_AMT"]


def analisa_arvore(dados, folds, depth):
    X_data = dados.drop(TARGETS, axis = 1)
    y_data = dados.loc[ : , THIS_TARGET]

    avaliacoes = list()
    for i, (train_index, test_index) in enumerate(folds):
        print(f"Árvore, iteração {i}")
        X_treino = X_data.iloc[train_index, : ]
        y_treino = y_data.iloc[train_index]

        X_teste = X_data.iloc[test_index, : ]
        y_teste = y_data.iloc[test_index]

        model = sklearn.tree.DecisionTreeRegressor(max_depth = depth)
        model.fit(X_treino, y_treino)

        treino_previsto = model.predict(X_treino)
        teste_previsto = model.predict(X_teste)

        avaliacao = auxiliares.avalia_modelo(y_treino, y_teste, treino_previsto, teste_previsto)
        avaliacoes.append(avaliacao)
    return avaliacoes


def analisa_glm(dados, folds, familia):
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

        model = sm.GLM(exog = exog_data,
                       endog = endog_data,
                       family = familia)

        glm = model.fit()

        treino_previsto = glm.predict(X_treino)
        teste_previsto = glm.predict(X_teste)

        avaliacao = auxiliares.avalia_modelo(y_treino, y_teste, treino_previsto, teste_previsto)
        avaliacoes.append(avaliacao)
    return avaliacoes


def analisa_rbf(dados, folds, number_of_centers):
    X_data = dados.drop(TARGETS, axis = 1)
    y_data = dados.loc[ : , THIS_TARGET]

    avaliacoes = list()
    for i, (train_index, test_index) in enumerate(folds):
        print(f"RBF, iteração {i}")
        X_treino = X_data.iloc[train_index, : ]
        y_treino = y_data.iloc[train_index]

        X_teste = X_data.iloc[test_index, : ]
        y_teste = y_data.iloc[test_index]

        model = RBF.RBFRegressor(number_of_centers = number_of_centers, 
                                  algorithm = sklearn.linear_model.LinearRegression(), 
                                  random_state=42)
        model.fit(X_treino, y_treino)

        treino_previsto = model.predict(X_treino)
        teste_previsto = model.predict(X_teste)

        avaliacao = auxiliares.avalia_modelo(y_treino, y_teste, treino_previsto, teste_previsto)
        avaliacoes.append(avaliacao)
    return avaliacoes

