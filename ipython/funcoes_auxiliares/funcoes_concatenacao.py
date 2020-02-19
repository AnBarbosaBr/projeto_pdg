import pandas as pd
import os


def agrupa_modelos(df):
    return df.groupby(["Nome","Tipo"]).sum()


def enriquece_matriz_de_confusao(df):
    real_positivo = df["False_Negative"] + df["True_Positive"]
    real_negativo = df["False_Positive"] + df["True_Negative"]
    previsto_positivo = df["False_Positive"] + df["True_Positive"]
    previsto_negativo = df["False_Negative"] + df["True_Negative"]
    total = real_positivo + real_negativo
    recall = df["True_Positive"]/real_positivo
    precisao = df["True_Positive"]/previsto_positivo

    percentagem_prevista_negativo = previsto_negativo / total
    percentagem_prevista_positivo = previsto_positivo / total
    
    ultimo_index = df.shape[1]
    df.insert(loc = ultimo_index, column = "Recall", value = recall)
    df.insert(loc = ultimo_index + 1, column = "Precisao", value = precisao)
    df.insert(loc = ultimo_index + 2, column = "%PrevistaNegativo", value = percentagem_prevista_negativo)
    df.insert(loc = ultimo_index + 3, column = "%PrevistaPositiva", value = percentagem_prevista_positivo)
    return df


def concatena_arquivos(data_hora_desejada, pasta = 'outputs'):
    csvs = os.listdir(pasta)  
    lista = [arquivo for arquivo in csvs if (data_hora_desejada in arquivo and "avaliacao" in arquivo)]
    
    full_df = pd.DataFrame()
    for arquivo in lista:
        df_arquivo = pd.read_csv(f"outputs/{arquivo}", index_col = 0)
        full_df = pd.concat([full_df, df_arquivo], ignore_index = True)
    return full_df
    


def relatorio_grid_search_rbf(df, output_name):
    df = df.reset_index()
    descricao = df['Nome'].str.split("_", expand = True)
    df['Nome'] = descricao[0]
    df.insert(loc = 1, column = "NUM_CENTERS", value = descricao[1])
    df.insert(loc = 2, column = "TRESHOLD", value = descricao[2])
    
    df.groupby(["Nome", "Tipo", "NUM_CENTERS","TRESHOLD"]).agg(sum).to_csv(f"{output_name}_centro_treshold.csv", sep = ";", decimal = ",", encoding = "utf-8-sig")
    df.groupby(["Nome", "Tipo", "TRESHOLD", "NUM_CENTERS"]).agg(sum).to_csv(f"{output_name}_treshold_centros.csv", sep = ";", decimal = ",", encoding = "utf-8-sig")

    df.groupby(["Nome", "Tipo", "TRESHOLD"]).agg(sum).to_csv(f"{output_name}_treshold.csv", sep = ";", decimal = ",", encoding = "utf-8-sig")
    df.groupby(["Nome", "Tipo", "NUM_CENTERS"]).agg(sum).to_csv(f"{output_name}_centros.csv", sep = ";", decimal = ",", encoding = "utf-8-sig")

    df.to_csv(f"{output_name}_resumo.csv")

    
    print("Finished")


def relatorio_grid_search_glm(df, output_name):
    df = df.reset_index()
    descricao = df['Nome'].str.split("_", expand = True)
    df['Nome'] = descricao[0]
    df.insert(loc = 1, column = "FAMILIA", value = descricao[1])
    df.insert(loc = 2, column = "VALOR_TWEEDIE", value = descricao[3])
    
    df.groupby(["Nome", "Tipo", "VALOR_TWEEDIE"]).agg(sum).to_csv(f"{output_name}_valor_tweedie.csv", sep = ";", decimal = ",", encoding = "utf-8-sig")
    df.to_csv(f"{output_name}_resumo.csv", sep = ";", decimal = ",", encoding = "utf-8-sig")
    print("Finished")
