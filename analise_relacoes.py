import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Configurações para visualizações
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 12
sns.set_palette('viridis')

# Diretório para salvar as visualizações e resultados
diretorio_visualizacoes = '/home/ubuntu/analise_pordata/visualizacoes'
diretorio_resultados = '/home/ubuntu/analise_pordata/resultados'
os.makedirs(diretorio_resultados, exist_ok=True)

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

# Função para criar dataframe integrado
def criar_dataframe_integrado(datasets):
    print("\nCriando dataframe integrado para análise de relações...")
    
    # Identificar anos comuns a todos os datasets
    anos_por_dataset = {}
    for nome, df in datasets.items():
        anos_por_dataset[nome] = set(df['Ano'].unique())
    
    # Encontrar interseção de anos
    anos_comuns = set.intersection(*anos_por_dataset.values()) if anos_por_dataset else set()
    print(f"Anos comuns a todos os datasets: {sorted(anos_comuns)}")
    
    # Identificar países comuns
    paises_por_dataset = {}
    for nome, df in datasets.items():
        if 'Pais' in df.columns:
            paises_por_dataset[nome] = set(df['Pais'].dropna().unique())
    
    # Encontrar interseção de países
    paises_comuns = set.intersection(*paises_por_dataset.values()) if paises_por_dataset else set()
    print(f"Países comuns a todos os datasets: {sorted(paises_comuns)}")
    
    # Criar dataframe integrado
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
        
        print(f"Dataframe integrado criado: {df_integrado.shape[0]} linhas x {df_integrado.shape[1]} colunas")
        
        # Salvar dataframe integrado
        df_integrado.to_csv(os.path.join(diretorio_resultados, 'dataframe_integrado.csv'), index=False)
        print(f"Dataframe integrado salvo em: {os.path.join(diretorio_resultados, 'dataframe_integrado.csv')}")
        
        return df_integrado
    else:
        print("Não há anos ou países comuns suficientes para criar o dataframe integrado.")
        return None

# Função para analisar relações entre variáveis
def analisar_relacoes(df_integrado):
    if df_integrado is None or df_integrado.empty:
        print("Dataframe integrado vazio ou nulo. Não é possível analisar relações.")
        return
    
    print("\nAnalisando relações entre variáveis...")
    
    # 1. Análise de correlação
    analisar_correlacoes(df_integrado)
    
    # 2. Análise de regressão
    analisar_regressoes(df_integrado)
    
    # 3. Análise de componentes principais (PCA)
    analisar_pca(df_integrado)
    
    # 4. Análise de clusters
    analisar_clusters(df_integrado)
    
    # 5. Análise de tendências temporais
    analisar_tendencias_temporais(df_integrado)

# 1. Análise de correlação
def analisar_correlacoes(df_integrado):
    print("\n1. Análise de Correlação")
    
    # Colunas numéricas (excluindo Ano)
    colunas_numericas = df_integrado.select_dtypes(include=['number']).columns.tolist()
    colunas_numericas.remove('Ano') if 'Ano' in colunas_numericas else None
    
    # Calcular matriz de correlação
    matriz_correlacao = df_integrado[colunas_numericas].corr()
    print("\nMatriz de Correlação:")
    print(matriz_correlacao)
    
    # Identificar correlações mais fortes (positivas e negativas)
    correlacoes = []
    for i in range(len(matriz_correlacao.columns)):
        for j in range(i+1, len(matriz_correlacao.columns)):
            col1 = matriz_correlacao.columns[i]
            col2 = matriz_correlacao.columns[j]
            corr = matriz_correlacao.iloc[i, j]
            correlacoes.append((col1, col2, corr))
    
    # Ordenar por valor absoluto de correlação
    correlacoes_ordenadas = sorted(correlacoes, key=lambda x: abs(x[2]), reverse=True)
    
    print("\nCorrelações mais fortes:")
    for col1, col2, corr in correlacoes_ordenadas[:5]:
        print(f"{col1} vs {col2}: {corr:.4f}")
    
    # Salvar resultados
    with open(os.path.join(diretorio_resultados, 'analise_correlacao.txt'), 'w') as f:
        f.write("ANÁLISE DE CORRELAÇÃO\n")
        f.write("="*50 + "\n\n")
        
        f.write("Matriz de Correlação:\n")
        f.write(str(matriz_correlacao) + "\n\n")
        
        f.write("Correlações mais fortes:\n")
        for col1, col2, corr in correlacoes_ordenadas:
            f.write(f"{col1} vs {col2}: {corr:.4f}\n")
            
            # Adicionar interpretação para correlações fortes
            if abs(corr) > 0.7:
                if corr > 0:
                    f.write(f"  Interpretação: Forte correlação positiva - quando {col1} aumenta, {col2} tende a aumentar também.\n")
                else:
                    f.write(f"  Interpretação: Forte correlação negativa - quando {col1} aumenta, {col2} tende a diminuir.\n")
            elif abs(corr) > 0.5:
                if corr > 0:
                    f.write(f"  Interpretação: Correlação positiva moderada entre {col1} e {col2}.\n")
                else:
                    f.write(f"  Interpretação: Correlação negativa moderada entre {col1} e {col2}.\n")
            else:
                f.write(f"  Interpretação: Correlação fraca entre {col1} e {col2}.\n")
            
            f.write("\n")
    
    print(f"Análise de correlação salva em: {os.path.join(diretorio_resultados, 'analise_correlacao.txt')}")

# 2. Análise de regressão
def analisar_regressoes(df_integrado):
    print("\n2. Análise de Regressão")
    
    # Colunas numéricas (excluindo Ano)
    colunas_numericas = df_integrado.select_dtypes(include=['number']).columns.tolist()
    colunas_numericas.remove('Ano') if 'Ano' in colunas_numericas else None
    
    # Calcular matriz de correlação para identificar pares para regressão
    matriz_correlacao = df_integrado[colunas_numericas].corr()
    
    # Identificar correlações mais fortes (positivas e negativas)
    correlacoes = []
    for i in range(len(matriz_correlacao.columns)):
        for j in range(i+1, len(matriz_correlacao.columns)):
            col1 = matriz_correlacao.columns[i]
            col2 = matriz_correlacao.columns[j]
            corr = matriz_correlacao.iloc[i, j]
            correlacoes.append((col1, col2, corr))
    
    # Ordenar por valor absoluto de correlação
    correlacoes_ordenadas = sorted(correlacoes, key=lambda x: abs(x[2]), reverse=True)
    
    # Analisar regressão para os 3 pares com correlações mais fortes
    resultados_regressao = []
    
    for col1, col2, corr in correlacoes_ordenadas[:3]:
        print(f"\nAnalisando regressão: {col1} vs {col2}")
        
        # Remover linhas com valores nulos
        df_temp = df_integrado[[col1, col2]].dropna()
        
        # Verificar se há dados suficientes
        if len(df_temp) < 10:
            print(f"Dados insuficientes para análise de regressão entre {col1} e {col2}")
            continue
        
        # Realizar regressão linear
        X = df_temp[col1].values.reshape(-1, 1)
        y = df_temp[col2].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(df_temp[col1], df_temp[col2])
        
        print(f"Equação da reta: {col2} = {slope:.4f} * {col1} + {intercept:.4f}")
        print(f"R²: {r_value**2:.4f}")
        print(f"p-valor: {p_value:.4f}")
        
        # Armazenar resultados
        resultados_regressao.append({
            'var_x': col1,
            'var_y': col2,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err
        })
        
        # Criar gráfico de regressão
        plt.figure(figsize=(12, 8))
        sns.regplot(x=col1, y=col2, data=df_temp, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        plt.title(f'Regressão Linear: {col1} vs {col2} (R² = {r_value**2:.4f})', fontsize=16)
        plt.xlabel(col1, fontsize=14)
        plt.ylabel(col2, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Salvar gráfico
        caminho_grafico = os.path.join(diretorio_visualizacoes, f'regressao_{col1}_{col2}.png')
        plt.savefig(caminho_grafico)
        plt.close()
        print(f"Gráfico de regressão salvo em: {caminho_grafico}")
    
    # Salvar resultados
    with open(os.path.join(diretorio_resultados, 'analise_regressao.txt'), 'w') as f:
        f.write("ANÁLISE DE REGRESSÃO LINEAR\n")
        f.write("="*50 + "\n\n")
        
        for resultado in resultados_regressao:
            f.write(f"Regressão: {resultado['var_x']} vs {resultado['var_y']}\n")
            f.write(f"Equação da reta: {resultado['var_y']} = {resultado['slope']:.4f} * {resultado['var_x']} + {resultado['intercept']:.4f}\n")
            f.write(f"R²: {resultado['r_squared']:.4f}\n")
            f.write(f"p-valor: {resultado['p_value']:.4f}\n")
            f.write(f"Erro padrão: {resultado['std_err']:.4f}\n")
            
            # Adicionar interpretação
            f.write("\nInterpretação:\n")
            
            if resultado['p_value'] < 0.05:
                f.write("- A relação é estatisticamente significativa (p < 0.05).\n")
            else:
                f.write("- A relação não é estatisticamente significativa (p >= 0.05).\n")
            
            if resultado['r_squared'] > 0.7:
                f.write(f"- O modelo explica uma grande parte da variação em {resultado['var_y']} (R² > 0.7).\n")
            elif resultado['r_squared'] > 0.5:
                f.write(f"- O modelo explica uma parte moderada da variação em {resultado['var_y']} (R² > 0.5).\n")
            else:
                f.write(f"- O modelo explica apenas uma pequena parte da variação em {resultado['var_y']} (R² < 0.5).\n")
            
            if resultado['slope'] > 0:
                f.write(f"- Para cada unidade de aumento em {resultado['var_x']}, {resultado['var_y']} aumenta em média {resultado['slope']:.4f} unidades.\n")
            else:
                f.write(f"- Para cada unidade de aumento em {resultado['var_x']}, {resultado['var_y']} diminui em média {abs(resultado['slope']):.4f} unidades.\n")
            
            f.write("\n" + "-"*50 + "\n\n")
    
    print(f"Análise de regressão salva em: {os.path.join(diretorio_resultados, 'analise_regressao.txt')}")

# 3. Análise de componentes principais (PCA)
def analisar_pca(df_integrado):
    print("\n3. Análise de Componentes Principais (PCA)")
    
    # Colunas numéricas (excluindo Ano)
    colunas_numericas = df_integrado.select_dtypes(include=['number']).columns.tolist()
    colunas_numericas.remove('Ano') if 'Ano' in colunas_numericas else None
    
    # Verificar se há dados suficientes
    if len(colunas_numericas) < 2:
        print("Dados insuficientes para análise PCA (menos de 2 variáveis numéricas)")
        return
    
    # Remover linhas com valores nulos
    df_pca = df_integrado[colunas_numericas].dropna()
    
    if len(df_pca) < 10:
        print("Dados insuficientes para análise PCA (menos de 10 observações completas)")
        return
    
    # Padronizar os dados
    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(df_pca)
    
    # Aplicar PCA
    pca = PCA()
    componentes_principais = pca.fit_transform(dados_padronizados)
    
    # Variância explicada
    variancia_explicada = pca.explained_variance_ratio_
    variancia_acumulada = np.cumsum(variancia_explicada)
    
    print("\nVariância explicada por componente:")
    for i, var in enumerate(variancia_explicada):
        print(f"PC{i+1}: {var:.4f} ({variancia_acumulada[i]:.4f} acumulada)")
    
    # Criar gráfico de variância explicada
    plt.figure(figsize=(12, 8))
    plt.bar(range(1, len(variancia_explicada) + 1), variancia_explicada, alpha=0.7, label='Individual')
    plt.step(range(1, len(variancia_acumulada) + 1), variancia_acumulada, where='mid', label='Acumulada')
    plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Limite 80%')
    plt.title('Variância Explicada por Componente Principal', fontsize=16)
    plt.xlabel('Componente Principal', fontsize=14)
    plt.ylabel('Proporção de Variância Explicada', fontsize=14)
    plt.xticks(range(1, len(variancia_explicada) + 1))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Salvar gráfico
    caminho_grafico = os.path.join(diretorio_visualizacoes, 'pca_variancia_explicada.png')
    plt.savefig(caminho_grafico)
    plt.close()
    print(f"Gráfico de variância explicada salvo em: {caminho_grafico}")
    
    # Analisar cargas (loadings) das variáveis
    cargas = pca.components_
    
    # Criar dataframe de cargas
    df_cargas = pd.DataFrame(
        cargas.T,
        columns=[f'PC{i+1}' for i in range(cargas.shape[0])],
        index=colunas_numericas
    )
    
    print("\nCargas das variáveis nos componentes principais:")
    print(df_cargas)
    
    # Criar gráfico de cargas para PC1 e PC2
    if len(cargas) >= 2:
        plt.figure(figsize=(12, 10))
        
        # Plotar setas para cada variável
        for i, var in enumerate(colunas_numericas):
            plt.arrow(0, 0, cargas[0, i], cargas[1, i], head_width=0.05, head_length=0.05, fc='blue', ec='blue')
            plt.text(cargas[0, i]*1.15, cargas[1, i]*1.15, var, fontsize=12)
        
        # Adicionar círculo de correlação
        circulo = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
        plt.gca().add_patch(circulo)
        
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.grid(True, alpha=0.3)
        plt.title('Círculo de Correlação - PC1 vs PC2', fontsize=16)
        plt.xlabel(f'PC1 ({variancia_explicada[0]:.2%})', fontsize=14)
        plt.ylabel(f'PC2 ({variancia_explicada[1]:.2%})', fontsize=14)
        plt.tight_layout()
        
        # Salvar gráfico
        caminho_grafico = os.path.jo
(Content truncated due to size limit. Use line ranges to read in chunks)