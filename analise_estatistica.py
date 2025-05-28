import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import limpeza_dados 

# Configurações para visualizações
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_palette('viridis')

# Diretórios
diretorio_base = os.path.join(os.getcwd(), "analise_pordata")
diretorio_visualizacoes = os.path.join(diretorio_base, "visualizacoes")
diretorio_resultados = os.path.join(diretorio_base, "resultados")
os.makedirs(diretorio_visualizacoes, exist_ok=True)
os.makedirs(diretorio_resultados, exist_ok=True)


def carregar_datasets_limpos():
    datasets = limpeza_dados.datasets_limpos
    return datasets

# Estatística descritiva
def analise_estatistica_descritiva(datasets):
    resultados = {}
    for nome, df in datasets.items():
        print(f"\n{'='*50}\nAnálise Estatística: {nome}\n{'='*50}")
        try:
            estatisticas = df['Valor'].describe()
            print(estatisticas)
            amostra = df['Valor'].dropna().sample(min(5000, len(df['Valor'].dropna())))
            shapiro = stats.shapiro(amostra)
            print(f"\nShapiro-Wilk: estatística = {shapiro.statistic:.4f}, p-valor = {shapiro.pvalue:.4f}")

            evolucao = df.groupby('Ano')['Valor'].mean()
            taxa_crescimento = evolucao.pct_change() * 100
            taxa_media = taxa_crescimento.mean()

            resultados[nome] = {
                'estatisticas': estatisticas,
                'normalidade': shapiro,
                'evolucao_temporal': evolucao
            }

            with open(os.path.join(diretorio_resultados, f'estatisticas_{nome}.txt'), 'w') as f:
                f.write(str(estatisticas) + '\n')
                f.write(f"Shapiro-Wilk: {shapiro.statistic:.4f}, p-valor: {shapiro.pvalue:.4f}\n")
                f.write("Evolução temporal:\n" + str(evolucao) + "\n")
                f.write("Taxa de crescimento anual (%):\n" + str(taxa_crescimento) + "\n")
                f.write(f"Taxa de crescimento média: {taxa_media:.2f}%\n")

        except Exception as e:
            print(f"Erro na análise de {nome}: {e}")
    return resultados

# Visualizações
def criar_visualizacoes_basicas(datasets):
    for nome, df in datasets.items():
        try:
            plt.figure()
            sns.histplot(df['Valor'].dropna(), kde=True)
            plt.title(f"Distribuição - {nome}")
            plt.tight_layout()
            plt.savefig(os.path.join(diretorio_visualizacoes, f"{nome}_histograma.png"))
            plt.close()

            plt.figure()
            df.groupby('Ano')['Valor'].mean().plot(marker='o')
            plt.title(f"Evolução Temporal - {nome}")
            plt.tight_layout()
            plt.savefig(os.path.join(diretorio_visualizacoes, f"{nome}_evolucao_temporal.png"))
            plt.close()
        except Exception as e:
            print(f"Erro nas visualizações de {nome}: {e}")

# Correlações
def analisar_correlacoes(datasets):
    anos = [set(df['Ano'].unique()) for df in datasets.values() if 'Ano' in df.columns]
    paises = [set(df['Pais'].dropna().unique()) for df in datasets.values() if 'Pais' in df.columns]
    anos_comuns = set.intersection(*anos) if anos else set()
    paises_comuns = set.intersection(*paises) if paises else set()

    if not anos_comuns or not paises_comuns:
        print(" Anos ou países comuns insuficientes para análise de correlação.")
        return None, None

    df_geral = pd.DataFrame()
    for nome, df in datasets.items():
        if 'Pais' in df.columns:
            df_filtrado = df[df['Ano'].isin(anos_comuns) & df['Pais'].isin(paises_comuns)]
            df_medio = df_filtrado.groupby(['Pais', 'Ano'])['Valor'].mean().reset_index()
            df_medio = df_medio.rename(columns={'Valor': nome})
            df_geral = df_medio if df_geral.empty else pd.merge(df_geral, df_medio, on=['Pais', 'Ano'])

    corr = df_geral.drop(['Pais', 'Ano'], axis=1).corr()
    print("\nMatriz de Correlação:")
    print(corr)

    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Matriz de Correlação")
    plt.tight_layout()
    plt.savefig(os.path.join(diretorio_visualizacoes, "matriz_correlacao.png"))
    plt.close()
    return df_geral, corr

# Main
def main():
    print("Iniciando análise estatística dos datasets da PORDATA...")
    datasets = carregar_datasets_limpos()  
    if not datasets:
        print("Nenhum dataset carregado com sucesso. Encerrando.")
        return

    analise_estatistica_descritiva(datasets)
    criar_visualizacoes_basicas(datasets)
    analisar_correlacoes(datasets)
    print(f"\n Análise estatística concluída. Resultados salvos em: {diretorio_base}")

if __name__ == "__main__":
    main()
