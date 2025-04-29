import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Configurações para visualizações
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
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

# Função para análise estatística descritiva
def analise_estatistica_descritiva(datasets):
    resultados = {}
    
    for nome, df in datasets.items():
        print(f"\n{'='*50}")
        print(f"Análise Estatística Descritiva: {nome}")
        print(f"{'='*50}")
        
        # Estatísticas descritivas para a coluna Valor
        estatisticas = df['Valor'].describe()
        print("\nEstatísticas descritivas para Valor:")
        print(estatisticas)
        
        # Verificar normalidade (teste de Shapiro-Wilk)
        # Limitando a 5000 amostras devido a limitações do teste
        amostra = df['Valor'].dropna().sample(min(5000, len(df['Valor'].dropna())))
        shapiro_test = stats.shapiro(amostra)
        print(f"\nTeste de normalidade (Shapiro-Wilk):")
        print(f"Estatística: {shapiro_test.statistic:.4f}, p-valor: {shapiro_test.pvalue:.4f}")
        if shapiro_test.pvalue < 0.05:
            print("Conclusão: Os dados não seguem uma distribuição normal (p < 0.05)")
        else:
            print("Conclusão: Os dados seguem uma distribuição normal (p >= 0.05)")
        
        # Análise por país (se aplicável)
        if 'Pais' in df.columns and df['Pais'].nunique() > 1:
            print("\nEstatísticas por país (top 5 países por valor médio):")
            stats_por_pais = df.groupby('Pais')['Valor'].agg(['mean', 'std', 'min', 'max', 'count']).sort_values('mean', ascending=False).head()
            print(stats_por_pais)
        
        # Análise por região (se aplicável)
        if 'Regiao' in df.columns and df['Regiao'].nunique() > 1:
            print("\nEstatísticas por região (top 5 regiões por valor médio):")
            stats_por_regiao = df.groupby('Regiao')['Valor'].agg(['mean', 'std', 'min', 'max', 'count']).sort_values('mean', ascending=False).head()
            print(stats_por_regiao)
        
        # Análise temporal
        print("\nEvolução temporal (média por ano):")
        evolucao_temporal = df.groupby('Ano')['Valor'].mean()
        print(evolucao_temporal)
        
        # Calcular taxa de crescimento anual (se aplicável)
        if len(evolucao_temporal) > 1:
            taxa_crescimento = evolucao_temporal.pct_change() * 100
            print("\nTaxa de crescimento anual (%):")
            print(taxa_crescimento)
            
            # Taxa de crescimento média
            taxa_media = taxa_crescimento.mean()
            print(f"\nTaxa de crescimento média: {taxa_media:.2f}%")
        
        # Armazenar resultados
        resultados[nome] = {
            'estatisticas': estatisticas,
            'normalidade': shapiro_test,
            'evolucao_temporal': evolucao_temporal
        }
        
        # Salvar estatísticas em arquivo
        with open(f'/home/ubuntu/analise_pordata/estatisticas_{nome}.txt', 'w') as f:
            f.write(f"ANÁLISE ESTATÍSTICA: {nome}\n")
            f.write("="*50 + "\n\n")
            
            f.write("Estatísticas descritivas para Valor:\n")
            f.write(str(estatisticas) + "\n\n")
            
            f.write(f"Teste de normalidade (Shapiro-Wilk):\n")
            f.write(f"Estatística: {shapiro_test.statistic:.4f}, p-valor: {shapiro_test.pvalue:.4f}\n")
            if shapiro_test.pvalue < 0.05:
                f.write("Conclusão: Os dados não seguem uma distribuição normal (p < 0.05)\n\n")
            else:
                f.write("Conclusão: Os dados seguem uma distribuição normal (p >= 0.05)\n\n")
            
            f.write("Evolução temporal (média por ano):\n")
            f.write(str(evolucao_temporal) + "\n\n")
            
            if len(evolucao_temporal) > 1:
                f.write("Taxa de crescimento anual (%):\n")
                f.write(str(taxa_crescimento) + "\n\n")
                f.write(f"Taxa de crescimento média: {taxa_media:.2f}%\n")
    
    return resultados

# Função para criar visualizações básicas
def criar_visualizacoes_basicas(datasets):
    for nome, df in datasets.items():
        print(f"\n{'='*50}")
        print(f"Criando visualizações para: {nome}")
        print(f"{'='*50}")
        
        # 1. Histograma da distribuição de valores
        plt.figure(figsize=(12, 8))
        sns.histplot(df['Valor'].dropna(), kde=True)
        plt.title(f'Distribuição de {df["Indicador"].iloc[0]}')
        plt.xlabel(f'{df["Indicador"].iloc[0]} ({df["Unidade"].iloc[0]})')
        plt.ylabel('Frequência')
        plt.tight_layout()
        caminho_hist = os.path.join(diretorio_visualizacoes, f'{nome}_histograma.png')
        plt.savefig(caminho_hist)
        plt.close()
        print(f"Histograma salvo em: {caminho_hist}")
        
        # 2. Evolução temporal (média por ano)
        plt.figure(figsize=(14, 8))
        evolucao = df.groupby('Ano')['Valor'].mean()
        evolucao.plot(marker='o')
        plt.title(f'Evolução Temporal de {df["Indicador"].iloc[0]}')
        plt.xlabel('Ano')
        plt.ylabel(f'{df["Indicador"].iloc[0]} ({df["Unidade"].iloc[0]})')
        plt.grid(True)
        plt.tight_layout()
        caminho_evolucao = os.path.join(diretorio_visualizacoes, f'{nome}_evolucao_temporal.png')
        plt.savefig(caminho_evolucao)
        plt.close()
        print(f"Gráfico de evolução temporal salvo em: {caminho_evolucao}")
        
        # 3. Boxplot por país (se aplicável)
        if 'Pais' in df.columns and df['Pais'].nunique() > 1:
            # Limitar a 15 países para melhor visualização
            paises_top = df.groupby('Pais')['Valor'].mean().sort_values(ascending=False).head(15).index
            df_top = df[df['Pais'].isin(paises_top)]
            
            plt.figure(figsize=(16, 10))
            sns.boxplot(x='Pais', y='Valor', data=df_top)
            plt.title(f'Distribuição de {df["Indicador"].iloc[0]} por País')
            plt.xlabel('País')
            plt.ylabel(f'{df["Indicador"].iloc[0]} ({df["Unidade"].iloc[0]})')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            caminho_boxplot = os.path.join(diretorio_visualizacoes, f'{nome}_boxplot_paises.png')
            plt.savefig(caminho_boxplot)
            plt.close()
            print(f"Boxplot por país salvo em: {caminho_boxplot}")
        
        # 4. Boxplot por região (se aplicável)
        if 'Regiao' in df.columns and df['Regiao'].nunique() > 1 and df['Regiao'].str.strip().nunique() > 1:
            # Filtrar regiões não vazias
            df_regioes = df[df['Regiao'].str.strip() != '']
            
            # Limitar a 15 regiões para melhor visualização
            regioes_top = df_regioes.groupby('Regiao')['Valor'].mean().sort_values(ascending=False).head(15).index
            df_top = df_regioes[df_regioes['Regiao'].isin(regioes_top)]
            
            plt.figure(figsize=(16, 10))
            sns.boxplot(x='Regiao', y='Valor', data=df_top)
            plt.title(f'Distribuição de {df["Indicador"].iloc[0]} por Região')
            plt.xlabel('Região')
            plt.ylabel(f'{df["Indicador"].iloc[0]} ({df["Unidade"].iloc[0]})')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            caminho_boxplot_regiao = os.path.join(diretorio_visualizacoes, f'{nome}_boxplot_regioes.png')
            plt.savefig(caminho_boxplot_regiao)
            plt.close()
            print(f"Boxplot por região salvo em: {caminho_boxplot_regiao}")

# Função para analisar correlações entre datasets
def analisar_correlacoes(datasets):
    print("\n" + "="*50)
    print("Análise de Correlações entre Datasets")
    print("="*50)
    
    # Preparar dataframe para análise de correlação
    # Vamos integrar os dados por país e ano
    
    # 1. Identificar intervalo comum de anos
    anos_por_dataset = {}
    for nome, df in datasets.items():
        anos_por_dataset[nome] = set(df['Ano'].unique())
    
    # Encontrar interseção de anos
    anos_comuns = set.intersection(*anos_por_dataset.values()) if anos_por_dataset else set()
    print(f"\nAnos comuns a todos os datasets: {sorted(anos_comuns)}")
    
    # 2. Identificar países comuns
    paises_por_dataset = {}
    for nome, df in datasets.items():
        if 'Pais' in df.columns:
            paises_por_dataset[nome] = set(df['Pais'].dropna().unique())
    
    # Encontrar interseção de países
    paises_comuns = set.intersection(*paises_por_dataset.values()) if paises_por_dataset else set()
    print(f"\nPaíses comuns a todos os datasets: {sorted(paises_comuns)}")
    
    # 3. Criar dataframe integrado para correlação
    if anos_comuns and paises_comuns:
        # Inicializar dataframe vazio
        df_correlacao = pd.DataFrame()
        
        # Para cada dataset, extrair valores médios por país e ano
        for nome, df in datasets.items():
            if 'Pais' in df.columns:
                # Filtrar para anos e países comuns
                df_filtrado = df[df['Ano'].isin(anos_comuns) & df['Pais'].isin(paises_comuns)]
                
                # Calcular média por país e ano
                df_medio = df_filtrado.groupby(['Pais', 'Ano'])['Valor'].mean().reset_index()
                
                # Renomear coluna de valor para o nome do dataset
                df_medio = df_medio.rename(columns={'Valor': nome})
                
                # Mesclar com o dataframe de correlação
                if df_correlacao.empty:
                    df_correlacao = df_medio
                else:
                    df_correlacao = pd.merge(df_correlacao, df_medio, on=['Pais', 'Ano'], how='outer')
        
        # Calcular matriz de correlação
        matriz_correlacao = df_correlacao.drop(['Pais', 'Ano'], axis=1).corr()
        print("\nMatriz de Correlação entre Datasets:")
        print(matriz_correlacao)
        
        # Salvar matriz de correlação
        with open('/home/ubuntu/analise_pordata/matriz_correlacao.txt', 'w') as f:
            f.write("MATRIZ DE CORRELAÇÃO ENTRE DATASETS\n")
            f.write("="*50 + "\n\n")
            f.write(str(matriz_correlacao) + "\n\n")
            f.write("\nNota: Correlações calculadas usando valores médios por país e ano.\n")
            f.write(f"Anos considerados: {sorted(anos_comuns)}\n")
            f.write(f"Países considerados: {sorted(paises_comuns)}\n")
        
        # Visualizar matriz de correlação como heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(matriz_correlacao, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Correlação entre Indicadores')
        plt.tight_layout()
        caminho_heatmap = os.path.join(diretorio_visualizacoes, 'matriz_correlacao_heatmap.png')
        plt.savefig(caminho_heatmap)
        plt.close()
        print(f"Heatmap de correlação salvo em: {caminho_heatmap}")
        
        # Analisar correlações específicas
        print("\nCorrelações mais fortes (positivas):")
        correlacoes = []
        for i in range(len(matriz_correlacao.columns)):
            for j in range(i+1, len(matriz_correlacao.columns)):
                col1 = matriz_correlacao.columns[i]
                col2 = matriz_correlacao.columns[j]
                corr = matriz_correlacao.iloc[i, j]
                correlacoes.append((col1, col2, corr))
        
        # Ordenar por valor absoluto de correlação
        correlacoes_ordenadas = sorted(correlacoes, key=lambda x: abs(x[2]), reverse=True)
        
        # Mostrar correlações mais fortes
        for col1, col2, corr in correlacoes_ordenadas:
            print(f"{col1} vs {col2}: {corr:.4f}")
            
            # Criar gráfico de dispersão para correlações fortes
            if abs(corr) > 0.5:  # Apenas para correlações significativas
                plt.figure(figsize=(10, 8))
                sns.scatterplot(data=df_correlacao, x=col1, y=col2, hue='Pais', alpha=0.7)
                plt.title(f'Correlação entre {col1} e {col2} (r = {corr:.4f})')
                plt.xlabel(f'{col1}')
                plt.ylabel(f'{col2}')
                plt.grid(True, alpha=0.3)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                caminho_scatter = os.path.join(diretorio_visualizacoes, f'correlacao_{col1}_{col2}.png')
                plt.savefig(caminho_scatter)
                plt.close()
                print(f"Gráfico de dispersão salvo em: {caminho_scatter}")
        
        return df_correlacao, matriz_correlacao
    else:
        print("Não há anos ou países comuns suficientes para análise de correlação.")
        return None, None

# Função principal
def main():
    print("Iniciando análise estatística dos datasets da PORDATA...")
    
    # Carregar datasets limpos
    datasets = carregar_datasets_limpos()
    
    # Realizar análise estatística descritiva
    resultados_estatisticos = analise_estatistica_descritiva(datasets)
    
    # Criar visualizações básicas
    criar_visualizacoes_basicas(datasets)
    
    # Analisar correlações entre datasets
    df_correlacao, matriz_correlacao = analisar_correlacoes(datasets)
    
    print("\nAnálise estatística concluída. Resultados salvos em '/home/ubuntu/analise_pordata/'")

if __name__ == "__main__":
    main()
