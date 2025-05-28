import os
import pandas as pd
from recolha_dados import recolha_dados  

# Função para limpar e pré-processar os dados
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

    df_limpo = df_limpo[df_limpo['Pais'] == 'Portugal']
    print(f"Linhas mantidas apenas de Portugal: {len(df_limpo)}")

    # 2. Converter a coluna Ano para inteiro (quando possível)
    try:
        df_limpo['Ano'] = pd.to_numeric(df_limpo['Ano'], errors='coerce')
        df_limpo['Ano'] = df_limpo['Ano'].astype('Int64')
        print("Coluna 'Ano' convertida para inteiro")
    except Exception as e:
        print(f"Erro ao converter coluna 'Ano' para inteiro: {e}")

    # 3. Tratar valores ausentes
    df_limpo = df_limpo.dropna(subset=['Valor'])
    print(f"Linhas removidas por valor ausente: {len(df) - len(df_limpo)}")

    for col in ['Pais', 'Regiao', 'Filtro1', 'Filtro2', 'Filtro3', 'Escala', 'Simbolo']:
        df_limpo[col] = df_limpo[col].fillna('')

    # 4. Converter a coluna Valor para numérico
    df_limpo['Valor'] = pd.to_numeric(df_limpo['Valor'], errors='coerce')
    print("Coluna 'Valor' convertida para numérico")

    # 5. Adicionar metadados específicos para cada dataset
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

    return df_limpo

# Coleta e processa os dados automaticamente
datasets = recolha_dados()
datasets_limpos = {}

# Limpeza e pré-processamento dos dados coletados
for caminho, df in datasets.items():
    nome_dataset = os.path.basename(caminho)  
    print(f"\nProcessando dataset: {nome_dataset}")

    df_limpo = limpar_preprocessar(df, nome_dataset)
    datasets_limpos[nome_dataset] = df_limpo

