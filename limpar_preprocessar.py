import pandas as pd
import os
import chardet
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

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

# Função para limpar e pré-processar os datasets
def limpar_preprocessar(df, nome_dataset):
    print(f"\n{'='*50}")
    print(f"Limpeza e pré-processamento do dataset: {nome_dataset}")
    print(f"{'='*50}")
    
    # Cópia do dataframe original para não modificá-lo
    df_limpo = df.copy()
    
    # 1. Renomear colunas para facilitar o acesso
    colunas_originais = df_limpo.columns.tolist()
    colunas_novas = [
        'Ano', 'Pais', 'Regiao', 'Filtro1', 'Filtro2', 'Filtro3', 'Escala', 'Simbolo', 'Valor'
    ]
    
    # Verificar se o número de colunas corresponde
    if len(colunas_originais) == len(colunas_novas):
        df_limpo.columns = colunas_novas
        print(f"Colunas renomeadas: {colunas_originais} -> {colunas_novas}")
    else:
        print(f"Erro ao renomear colunas: número de colunas não corresponde ({len(colunas_originais)} vs {len(colunas_novas)})")
    
    # 2. Converter a coluna Ano para inteiro (quando possível)
    try:
        # Primeiro, remover valores não numéricos
        df_limpo['Ano'] = pd.to_numeric(df_limpo['Ano'], errors='coerce')
        # Converter para inteiro, mantendo NaN onde necessário
        df_limpo['Ano'] = df_limpo['Ano'].astype('Int64')
        print("Coluna 'Ano' convertida para inteiro")
    except Exception as e:
        print(f"Erro ao converter coluna 'Ano' para inteiro: {e}")
    
    # 3. Tratar valores ausentes
    # Contar valores ausentes antes da limpeza
    valores_ausentes_antes = df_limpo.isnull().sum()
    
    # Remover linhas onde o valor é nulo (não faz sentido manter registros sem valor)
    df_limpo = df_limpo.dropna(subset=['Valor'])
    print(f"Linhas removidas por valor ausente: {len(df) - len(df_limpo)}")
    
    # Preencher valores ausentes em colunas categóricas com string vazia
    for col in ['Pais', 'Regiao', 'Filtro1', 'Filtro2', 'Filtro3', 'Escala', 'Simbolo']:
        df_limpo[col] = df_limpo[col].fillna('')
    
    # Contar valores ausentes após a limpeza
    valores_ausentes_depois = df_limpo.isnull().sum()
    
    print("\nValores ausentes antes da limpeza:")
    print(valores_ausentes_antes)
    print("\nValores ausentes após a limpeza:")
    print(valores_ausentes_depois)
    
    # 4. Converter a coluna Valor para numérico
    try:
        # Converter para numérico, forçando erros para NaN
        df_limpo['Valor'] = pd.to_numeric(df_limpo['Valor'], errors='coerce')
        print("Coluna 'Valor' convertida para numérico")
    except Exception as e:
        print(f"Erro ao converter coluna 'Valor' para numérico: {e}")
    
    # 5. Detectar e tratar outliers
    if df_limpo['Valor'].dtype.kind in 'ifc':  # Verificar se é numérico
        # Calcular quartis e IQR
        Q1 = df_limpo['Valor'].quantile(0.25)
        Q3 = df_limpo['Valor'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Definir limites para outliers
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        # Identificar outliers
        outliers = df_limpo[(df_limpo['Valor'] < limite_inferior) | (df_limpo['Valor'] > limite_superior)]
        
        print(f"\nOutliers detectados: {len(outliers)} registros")
        if len(outliers) > 0:
            print("Exemplos de outliers:")
            print(outliers.head(5) if len(outliers) >= 5 else outliers)
            
            # Salvar informações sobre outliers
            outliers.to_csv(f'/home/ubuntu/analise_pordata/outliers_{nome_dataset}.csv', index=False)
            print(f"Informações sobre outliers salvas em '/home/ubuntu/analise_pordata/outliers_{nome_dataset}.csv'")
            
            # Não remover outliers automaticamente, apenas identificá-los
            # Em análises de dados socioeconômicos, outliers podem ser valores reais importantes
    
    # 6. Verificar consistência dos dados
    print("\nVerificando consistência dos dados:")
    
    # Verificar intervalo de anos
    anos_unicos = df_limpo['Ano'].dropna().unique()
    print(f"Intervalo de anos: {min(anos_unicos)} a {max(anos_unicos)}")
    
    # Verificar valores únicos em colunas categóricas
    for col in ['Pais', 'Regiao']:
        if col in df_limpo.columns:
            valores_unicos = df_limpo[col].unique()
            valores_nao_vazios = [v for v in valores_unicos if v]
            print(f"Valores únicos em '{col}': {len(valores_nao_vazios)}")
            if len(valores_nao_vazios) <= 10:
                print(valores_nao_vazios)
    
    # 7. Adicionar metadados específicos para cada dataset
    if nome_dataset == 'GANHOMEDIOMENSAL':
        df_limpo['Indicador'] = 'Ganho Médio Mensal'
        df_limpo['Unidade'] = 'euros'
    elif nome_dataset == 'ESPERANÇADEVIDA':
        df_limpo['Indicador'] = 'Esperança de Vida'
        df_limpo['Unidade'] = 'anos'
    elif nome_dataset == 'DESPESASAUDE':
        df_limpo['Indicador'] = 'Despesa em Saúde'
        df_limpo['Unidade'] = df_limpo['Escala'].str.strip()
    elif nome_dataset == 'PERCEÇAODESAUDE':
        df_limpo['Indicador'] = 'Percepção de Saúde'
        df_limpo['Unidade'] = '%'
    elif nome_dataset == 'TAXADEMORTALIDADEVITAVEL':
        df_limpo['Indicador'] = 'Taxa de Mortalidade Evitável'
        df_limpo['Unidade'] = 'por 100 mil habitantes'
    
    print(f"\nAdicionadas colunas 'Indicador' e 'Unidade' ao dataset {nome_dataset}")
    
    # 8. Salvar dataset limpo
    caminho_saida = f'/home/ubuntu/analise_pordata/{nome_dataset}_limpo.csv'
    df_limpo.to_csv(caminho_saida, index=False)
    print(f"\nDataset limpo salvo em '{caminho_saida}'")
    
    return df_limpo

# Caminho para os datasets
diretorio_datasets = "/home/ubuntu/upload/"
arquivos_datasets = [
    "GANHOMEDIOMENSAL.csv",
    "ESPERANÇADEVIDA.csv",
    "DESPESASAUDE.csv",
    "PERCEÇAODESAUDE.csv",
    "TAXADEMORTALIDADEVITAVEL.csv"
]

# Dicionário para armazenar os dataframes originais e limpos
datasets_originais = {}
datasets_limpos = {}

# Ler e limpar cada dataset
for arquivo in arquivos_datasets:
    caminho_completo = os.path.join(diretorio_datasets, arquivo)
    nome_dataset = os.path.splitext(arquivo)[0]
    
    print(f"\nProcessando {arquivo}...")
    df = ler_dataset(caminho_completo)
    
    if df is not None:
        datasets_originais[nome_dataset] = df
        datasets_limpos[nome_dataset] = limpar_preprocessar(df, nome_dataset)
    else:
        print(f"Não foi possível ler o dataset {arquivo}")

# Salvar um resumo da limpeza em um arquivo de texto
with open('/home/ubuntu/analise_pordata/resumo_limpeza.txt', 'w') as f:
    f.write("RESUMO DA LIMPEZA E PRÉ-PROCESSAMENTO DOS DATASETS DA PORDATA\n")
    f.write("="*70 + "\n\n")
    
    for nome, df_limpo in datasets_limpos.items():
        df_original = datasets_originais[nome]
        
        f.write(f"Dataset: {nome}\n")
        f.write(f"Dimensões originais: {df_original.shape[0]} linhas x {df_original.shape[1]} colunas\n")
        f.write(f"Dimensões após limpeza: {df_limpo.shape[0]} linhas x {df_limpo.shape[1]} colunas\n")
        f.write(f"Registros removidos: {df_original.shape[0] - df_limpo.shape[0]}\n")
        
        # Verificar se há outliers identificados
        caminho_outliers = f'/home/ubuntu/analise_pordata/outliers_{nome}.csv'
        if os.path.exists(caminho_outliers):
            outliers_df = pd.read_csv(caminho_outliers)
            f.write(f"Outliers identificados: {len(outliers_df)} registros\n")
        
        # Intervalo de anos
        anos_unicos = df_limpo['Ano'].dropna().unique()
        f.write(f"Intervalo de anos após limpeza: {min(anos_unicos)} a {max(anos_unicos)}\n")
        
        f.write("\n" + "-"*70 + "\n\n")

print("\nLimpeza e pré-processamento concluídos. Resumo salvo em '/home/ubuntu/analise_pordata/resumo_limpeza.txt'")
