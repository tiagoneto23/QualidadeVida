import pandas as pd
import os
import chardet
from tabulate import tabulate as tabulate_func
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Função para detectar a codificação do arquivo
def detectar_codificacao(arquivo):
    with open(arquivo, 'rb') as f:
        resultado = chardet.detect(f.read())
    return resultado['encoding']

# Função para ler os datasets com a codificação correta
def ler_dataset(arquivo):
    try:
        # Detectar a codificação do arquivo
        codificacao = detectar_codificacao(arquivo)
        print(f"Codificação detectada para {os.path.basename(arquivo)}: {codificacao}")
        
        # Tentar ler o arquivo com a codificação detectada
        df = pd.read_csv(arquivo, encoding=codificacao)
        return df
    except Exception as e:
        print(f"Erro ao ler {os.path.basename(arquivo)} com codificação {codificacao}: {e}")
        # Tentar com outras codificações comuns
        for enc in ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']:
            try:
                df = pd.read_csv(arquivo, encoding=enc)
                print(f"Leitura bem-sucedida com codificação {enc}")
                return df
            except Exception:
                continue
        
        # Se todas as tentativas falharem, tentar com engine python
        try:
            df = pd.read_csv(arquivo, encoding='latin1', engine='python')
            print(f"Leitura bem-sucedida com engine python e codificação latin1")
            return df
        except Exception as e:
            print(f"Todas as tentativas falharam: {e}")
            return None

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

# Caminho para os datasets
diretorio_datasets = "/home/ubuntu/upload/"
arquivos_datasets = [
    "GANHOMEDIOMENSAL.csv",
    "ESPERANÇADEVIDA.csv",
    "DESPESASAUDE.csv",
    "PERCEÇAODESAUDE.csv",
    "TAXADEMORTALIDADEVITAVEL.csv"
]

# Dicionário para armazenar os dataframes
datasets = {}

# Ler e analisar cada dataset
for arquivo in arquivos_datasets:
    caminho_completo = os.path.join(diretorio_datasets, arquivo)
    nome_dataset = os.path.splitext(arquivo)[0]
    
    print(f"\nProcessando {arquivo}...")
    df = ler_dataset(caminho_completo)
    
    if df is not None:
        datasets[nome_dataset] = df
        analisar_estrutura(df, nome_dataset)
    else:
        print(f"Não foi possível ler o dataset {arquivo}")

# Salvar um resumo da análise em um arquivo de texto
with open('/home/ubuntu/analise_pordata/resumo_datasets.txt', 'w') as f:
    f.write("RESUMO DA ANÁLISE DOS DATASETS DA PORDATA\n")
    f.write("="*50 + "\n\n")
    
    for nome, df in datasets.items():
        f.write(f"Dataset: {nome}\n")
        f.write(f"Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas\n")
        f.write(f"Colunas: {', '.join(df.columns)}\n")
        f.write(f"Período: {df.iloc[:, 0].min()} a {df.iloc[:, 0].max()} (assumindo que a primeira coluna é o ano)\n")
        f.write("\n" + "-"*50 + "\n\n")

print("\nAnálise concluída. Resumo salvo em '/home/ubuntu/analise_pordata/resumo_datasets.txt'")
