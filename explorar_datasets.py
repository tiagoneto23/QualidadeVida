import pandas as pd
import os
from tabulate import tabulate as tabulate_func
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from recolha_dados import recolha_dados

# Função para analisar a estrutura do dataset
def analisar_estrutura(df, nome):
    print(f"\n{'='*50}")
    print(f"Análise do dataset: {nome}")
    print(f"{'='*50}")

    # Informações básicas
    print(f"\nDimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")

    # Primeiras linhas
    print("\nPrimeiras 5 linhas:")
    print(tabulate_func(df.head(), headers='keys', tablefmt='psql'))

    # Informações sobre as colunas
    print("\nInformações sobre as colunas:")
    info = pd.DataFrame({
        'Tipo': df.dtypes,
        'Valores não nulos': df.count(),
        'Valores nulos': df.isnull().sum(),
        '% Valores nulos': (df.isnull().sum() / len(df) * 100).round(2),
        'Valores únicos': df.nunique()
    })
    print(tabulate_func(info, headers='keys', tablefmt='psql'))

    # Estatísticas descritivas para colunas numéricas
    colunas_numericas = df.select_dtypes(include=['number']).columns
    if len(colunas_numericas) > 0:
        print("\nEstatísticas descritivas para colunas numéricas:")
        desc = df[colunas_numericas].describe().T
        print(tabulate_func(desc, headers='keys', tablefmt='psql'))

    # Valores únicos para colunas categóricas (limitado a 10)
    colunas_categoricas = df.select_dtypes(exclude=['number']).columns
    if len(colunas_categoricas) > 0:
        print("\nValores únicos para colunas categóricas (limitado a 10):")
        for col in colunas_categoricas:
            valores_unicos = df[col].unique()
            print(f"\n{col}: {len(valores_unicos)} valores únicos")
            if len(valores_unicos) <= 10:
                print(valores_unicos)
            else:
                print(valores_unicos[:10], "... (mais valores)")

    return info

# === PROGRAMA PRINCIPAL ===

# Recolher os datasets automaticamente
datasets = recolha_dados()

# Caminho para salvar o resumo
resumo_path = os.path.join("C:/Users/Nambi/Documents/projecto", "resumo_datasets.txt")

# Abrir o arquivo de resumo para escrita
with open(resumo_path, 'w', encoding='utf-8') as f:
    f.write("RESUMO DA ANÁLISE DOS DATASETS DA PORDATA\n")
    f.write("="*50 + "\n\n")

    for caminho, df in datasets.items():
        nome_dataset = os.path.splitext(os.path.basename(caminho))[0]
        print(f"\nProcessando {nome_dataset}...")
        info = analisar_estrutura(df, nome_dataset)

        f.write(f"Dataset: {nome_dataset}\n")
        f.write(f"Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas\n")
        f.write(f"Colunas: {', '.join(df.columns)}\n")
        try:
            ano_min = df.iloc[:, 0].min()
            ano_max = df.iloc[:, 0].max()
            f.write(f"Período: {ano_min} a {ano_max}\n")
        except Exception:
            f.write("Período: Não disponível\n")
        f.write("\n" + "-"*50 + "\n\n")

print(f"\nAnálise concluída. Resumo salvo em '{resumo_path}'")
