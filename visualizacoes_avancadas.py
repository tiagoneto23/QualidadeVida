import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# Configurações para visualizações
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12
sns.set_palette('viridis')

# Diretório para salvar as visualizações
diretorio_visualizacoes = '/home/ubuntu/analise_pordata/visualizacoes'
os.makedirs(diretorio_visualizacoes, exist_ok=True)

# Função para carregar os datasets limpos
def carregar_datasets_limpos():
    diretorio = '/home/ubuntu/analise_pordata/'
    datasets = {}
    
    # Lista de arquivos limpos
    arquivos_limpos = [
        "GANHOMEDIOMENSAL_limpo.csv",
        "ESPERANÇADEVIDA_limpo.csv",
        "DESPESASAUDE_limpo.csv",
        "PERCEÇAODESAUDE_limpo.csv",
        "TAXADEMORTALIDADEVITAVEL_limpo.csv"
    ]
    
    for arquivo in arquivos_limpos:
        nome = arquivo.split('_')[0]
        caminho = os.path.join(diretorio, arquivo)
        try:
            df = pd.read_csv(caminho)
            datasets[nome] = df
            print(f"Dataset {nome} carregado: {df.shape[0]} linhas x {df.shape[1]} colunas")
        except Exception as e:
            print(f"Erro ao carregar {arquivo}: {e}")
    
    return datasets

# Função para criar visualizações avançadas
def criar_visualizacoes_avancadas(datasets):
    print("\nCriando visualizações avançadas...")
    
    # 1. Gráfico de linhas múltiplas para evolução temporal de todos os indicadores (normalizados)
    criar_grafico_evolucao_multipla(datasets)
    
    # 2. Mapa de calor para correlações entre indicadores por país
    criar_mapa_calor_por_pais(datasets)
    
    # 3. Gráficos de dispersão com regressão para pares de indicadores importantes
    criar_graficos_dispersao_regressao(datasets)
    
    # 4. Gráficos de barras para comparação entre países
    criar_graficos_barras_comparativos(datasets)
    
    # 5. Gráficos de área para evolução temporal
    criar_graficos_area(datasets)
    
    # 6. Dashboard com múltiplos indicadores
    criar_dashboard_indicadores(datasets)

# 1. Gráfico de linhas múltiplas para evolução temporal de todos os indicadores (normalizados)
def criar_grafico_evolucao_multipla(datasets):
    print("Criando gráfico de evolução temporal múltipla...")
    
    # Identificar anos comuns a todos os datasets
    anos_por_dataset = {}
    for nome, df in datasets.items():
        anos_por_dataset[nome] = set(df['Ano'].unique())
    
    # Encontrar interseção de anos
    anos_comuns = set.intersection(*anos_por_dataset.values()) if anos_por_dataset else set()
    
    if not anos_comuns:
        print("Não há anos comuns suficientes para criar o gráfico de evolução múltipla.")
        return
    
    # Filtrar para Portugal e anos comuns
    df_evolucao = pd.DataFrame(index=sorted(anos_comuns))
    
    for nome, df in datasets.items():
        # Filtrar para Portugal (se aplicável) e anos comuns
        df_filtrado = df[df['Ano'].isin(anos_comuns)]
        if 'Pais' in df.columns:
            df_filtrado = df_filtrado[df_filtrado['Pais'] == 'Portugal']
        
        # Calcular média por ano
        serie_anual = df_filtrado.groupby('Ano')['Valor'].mean()
        
        # Normalizar valores (0-1) para comparabilidade
        min_val = serie_anual.min()
        max_val = serie_anual.max()
        serie_normalizada = (serie_anual - min_val) / (max_val - min_val) if max_val > min_val else serie_anual
        
        # Adicionar ao dataframe de evolução
        df_evolucao[nome] = serie_normalizada
    
    # Criar gráfico
    plt.figure(figsize=(16, 10))
    
    for coluna in df_evolucao.columns:
        plt.plot(df_evolucao.index, df_evolucao[coluna], marker='o', linewidth=2.5, label=coluna)
    
    plt.title('Evolução Temporal Comparativa dos Indicadores (Normalizados)', fontsize=16)
    plt.xlabel('Ano', fontsize=14)
    plt.ylabel('Valor Normalizado (0-1)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Salvar gráfico
    caminho_grafico = os.path.join(diretorio_visualizacoes, 'evolucao_temporal_multipla.png')
    plt.savefig(caminho_grafico)
    plt.close()
    print(f"Gráfico de evolução temporal múltipla salvo em: {caminho_grafico}")

# 2. Mapa de calor para correlações entre indicadores por país
def criar_mapa_calor_por_pais(datasets):
    print("Criando mapa de calor por país...")
    
    # Identificar países comuns com dados suficientes
    paises_por_dataset = {}
    for nome, df in datasets.items():
        if 'Pais' in df.columns:
            # Considerar apenas países com pelo menos 5 anos de dados
            paises_com_dados = df.groupby('Pais').filter(lambda x: len(x) >= 5)['Pais'].unique()
            paises_por_dataset[nome] = set(paises_com_dados)
    
    # Encontrar interseção de países
    paises_comuns = set.intersection(*paises_por_dataset.values()) if paises_por_dataset else set()
    
    if len(paises_comuns) < 5:
        print("Não há países comuns suficientes para criar o mapa de calor por país.")
        return
    
    # Limitar a 15 países para melhor visualização
    paises_selecionados = list(paises_comuns)[:15]
    
    # Criar matriz de correlação por país
    correlacoes_por_pais = {}
    
    for pais in paises_selecionados:
        # Dataframe para armazenar valores médios por ano para este país
        df_pais = pd.DataFrame()
        
        for nome, df in datasets.items():
            if 'Pais' in df.columns:
                # Filtrar para este país
                df_filtrado = df[df['Pais'] == pais]
                
                # Calcular média por ano
                df_medio = df_filtrado.groupby('Ano')['Valor'].mean().reset_index()
                
                # Renomear coluna de valor para o nome do dataset
                df_medio = df_medio.rename(columns={'Valor': nome})
                
                # Mesclar com o dataframe do país
                if df_pais.empty:
                    df_pais = df_medio
                else:
                    df_pais = pd.merge(df_pais, df_medio, on='Ano', how='outer')
        
        # Calcular matriz de correlação para este país
        if not df_pais.empty and len(df_pais) >= 5:
            matriz_correlacao = df_pais.drop('Ano', axis=1).corr()
            correlacoes_por_pais[pais] = matriz_correlacao
    
    # Criar visualização de mapa de calor para cada país
    for pais, matriz in correlacoes_por_pais.items():
        plt.figure(figsize=(12, 10))
        sns.heatmap(matriz, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title(f'Correlação entre Indicadores: {pais}', fontsize=16)
        plt.tight_layout()
        
        # Salvar mapa de calor
        caminho_heatmap = os.path.join(diretorio_visualizacoes, f'correlacao_heatmap_{pais}.png')
        plt.savefig(caminho_heatmap)
        plt.close()
        print(f"Mapa de calor para {pais} salvo em: {caminho_heatmap}")

# 3. Gráficos de dispersão com regressão para pares de indicadores importantes
def criar_graficos_dispersao_regressao(datasets):
    print("Criando gráficos de dispersão com regressão...")
    
    # Pares de indicadores para análise (baseados nas correlações mais fortes)
    pares_indicadores = [
        ('GANHOMEDIOMENSAL', 'DESPESASAUDE'),
        ('GANHOMEDIOMENSAL', 'PERCEÇAODESAUDE'),
        ('ESPERANÇADEVIDA', 'TAXADEMORTALIDADEVITAVEL'),
        ('DESPESASAUDE', 'PERCEÇAODESAUDE'),
        ('ESPERANÇADEVIDA', 'DESPESASAUDE')
    ]
    
    # Identificar anos e países comuns
    anos_por_dataset = {}
    paises_por_dataset = {}
    
    for nome, df in datasets.items():
        anos_por_dataset[nome] = set(df['Ano'].unique())
        if 'Pais' in df.columns:
            paises_por_dataset[nome] = set(df['Pais'].unique())
    
    # Encontrar interseções
    anos_comuns = set.intersection(*anos_por_dataset.values()) if anos_por_dataset else set()
    paises_comuns = set.intersection(*paises_por_dataset.values()) if paises_por_dataset else set()
    
    # Criar dataframe integrado para análise
    if anos_comuns and paises_comuns:
        # Inicializar dataframe vazio
        df_integrado = pd.DataFrame()
        
        # Para cada dataset, extrair valores médios por país e ano
        for nome, df in datasets.items():
            if 'Pais' in df.columns:
                # Filtrar para anos e países comuns
                df_filtrado = df[df['Ano'].isin(anos_comuns) & df['Pais'].isin(paises_comuns)]
                
                # Calcular média por país e ano
                df_medio = df_filtrado.groupby(['Pais', 'Ano'])['Valor'].mean().reset_index()
                
                # Renomear coluna de valor para o nome do dataset
                df_medio = df_medio.rename(columns={'Valor': nome})
                
                # Mesclar com o dataframe integrado
                if df_integrado.empty:
                    df_integrado = df_medio
                else:
                    df_integrado = pd.merge(df_integrado, df_medio, on=['Pais', 'Ano'], how='outer')
        
        # Criar gráficos de dispersão com regressão para cada par de indicadores
        for ind1, ind2 in pares_indicadores:
            if ind1 in df_integrado.columns and ind2 in df_integrado.columns:
                plt.figure(figsize=(14, 10))
                
                # Criar gráfico de dispersão com linha de regressão
                sns.scatterplot(data=df_integrado, x=ind1, y=ind2, hue='Pais', s=100, alpha=0.7)
                sns.regplot(data=df_integrado, x=ind1, y=ind2, scatter=False, color='red', line_kws={'linewidth': 2})
                
                # Calcular correlação
                corr = df_integrado[ind1].corr(df_integrado[ind2])
                
                # Adicionar título e rótulos
                plt.title(f'Relação entre {ind1} e {ind2} (r = {corr:.4f})', fontsize=16)
                plt.xlabel(f'{ind1}', fontsize=14)
                plt.ylabel(f'{ind2}', fontsize=14)
                plt.grid(True, alpha=0.3)
                
                # Ajustar legenda
                plt.legend(title='País', bbox_to_anchor=(1.05, 1), loc='upper left')
                
                plt.tight_layout()
                
                # Salvar gráfico
                caminho_scatter = os.path.join(diretorio_visualizacoes, f'dispersao_regressao_{ind1}_{ind2}.png')
                plt.savefig(caminho_scatter)
                plt.close()
                print(f"Gráfico de dispersão com regressão salvo em: {caminho_scatter}")

# 4. Gráficos de barras para comparação entre países
def criar_graficos_barras_comparativos(datasets):
    print("Criando gráficos de barras comparativos...")
    
    # Para cada dataset, criar um gráfico de barras comparando os países
    for nome, df in datasets.items():
        if 'Pais' in df.columns and df['Pais'].nunique() > 1:
            # Filtrar para o ano mais recente disponível
            ano_mais_recente = df['Ano'].max()
            df_recente = df[df['Ano'] == ano_mais_recente]
            
            # Calcular média por país
            df_paises = df_recente.groupby('Pais')['Valor'].mean().reset_index()
            
            # Ordenar por valor
            df_paises = df_paises.sort_values('Valor', ascending=False)
            
            # Limitar a 15 países para melhor visualização
            df_paises = df_paises.head(15)
            
            # Criar gráfico de barras
            plt.figure(figsize=(16, 10))
            
            # Definir cores baseadas nos valores (gradiente)
            valores = df_paises['Valor']
            cores = plt.cm.viridis(np.linspace(0, 1, len(valores)))
            
            # Criar barras
            barras = plt.bar(df_paises['Pais'], df_paises['Valor'], color=cores)
            
            # Adicionar rótulos de valor no topo das barras
            for barra in barras:
                altura = barra.get_height()
                plt.text(barra.get_x() + barra.get_width()/2., altura + 0.1,
                        f'{altura:.1f}', ha='center', va='bottom', fontsize=10)
            
            # Adicionar título e rótulos
            plt.title(f'{df["Indicador"].iloc[0]} por País ({ano_mais_recente})', fontsize=16)
            plt.xlabel('País', fontsize=14)
            plt.ylabel(f'{df["Indicador"].iloc[0]} ({df["Unidade"].iloc[0]})', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            # Salvar gráfico
            caminho_barras = os.path.join(diretorio_visualizacoes, f'barras_paises_{nome}.png')
            plt.savefig(caminho_barras)
            plt.close()
            print(f"Gráfico de barras comparativo salvo em: {caminho_barras}")

# 5. Gráficos de área para evolução temporal
def criar_graficos_area(datasets):
    print("Criando gráficos de área para evolução temporal...")
    
    # Para cada dataset, criar um gráfico de área para evolução temporal
    for nome, df in datasets.items():
        # Filtrar para Portugal (se aplicável)
        if 'Pais' in df.columns:
            df_filtrado = df[df['Pais'] == 'Portugal']
        else:
            df_filtrado = df
        
        # Verificar se há dados suficientes
        if len(df_filtrado) < 5:
            print(f"Dados insuficientes para criar gráfico de área para {nome}")
            continue
        
        # Calcular média por ano
        df_anual = df_filtrado.groupby('Ano')['Valor'].mean().reset_index()
        
        # Criar gráfico de área
        plt.figure(figsize=(16, 10))
        
        # Definir gradiente de cores
        cores = plt.cm.viridis(np.linspace(0, 1, len(df_anual)))
        
        # Criar área
        plt.fill_between(df_anual['Ano'], df_anual['Valor'], color='skyblue', alpha=0.4)
        plt.plot(df_anual['Ano'], df_anual['Valor'], color='darkblue', linewidth=2.5, marker='o')
        
        # Adicionar rótulos de valor para alguns pontos (para não sobrecarregar)
        for i in range(0, len(df_anual), max(1, len(df_anual) // 10)):
            plt.text(df_anual['Ano'].iloc[i], df_anual['Valor'].iloc[i] + 0.1,
                    f'{df_anual["Valor"].iloc[i]:.1f}', ha='center', va='bottom', fontsize=10)
        
        # Adicionar título e rótulos
        plt.title(f'Evolução Temporal de {df["Indicador"].iloc[0]} em Portugal', fontsize=16)
        plt.xlabel('Ano', fontsize=14)
        plt.ylabel(f'{df["Indicador"].iloc[0]} ({df["Unidade"].iloc[0]})', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salvar gráfico
        caminho_area = os.path.join(diretorio_visualizacoes, f'area_temporal_{nome}.png')
        plt.savefig(caminho_area)
        plt.close()
        print(f"Gráfico de área para evolução temporal salvo em: {caminho_area}")

# 6. Dashboard com múltiplos indicadores
def criar_dashboard_indicadores(datasets):
    print("Criando dashboard com múltiplos indicadores...")
    
    # Identificar anos comuns a todos os datasets
    anos_por_dataset = {}
    for nome, df in datasets.items():
        anos_por_dataset[nome] = set(df['Ano'].unique())
    
    # Encontrar interseção de anos
    anos_comuns = set.intersection(*anos_por_dataset.values()) if anos_por_dataset else set()
    
    if not anos_comuns:
        print("Não há anos comuns suficientes para criar o dashboard.")
        return
    
    # Filtrar para Portugal e anos comuns
    dados_dashboard = {}
    
    for nome, df in datasets.items():
        # Filtrar para Portugal (se aplicável) e anos comuns
        df_filtrado = df[df['Ano'].isin(anos_comuns)]
        if 'Pais' in df.columns:
            df_filtrado = df_filtrado[df_filtrado['Pais'] == 'Portugal']
        
        # Calcular média por an