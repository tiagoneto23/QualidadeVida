import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mtick
from scipy import stats
import warnings
import limpeza_dados  # Importa o módulo de limpeza de dados

# Configurações globais
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# Definir diretório base
DIRETORIO_BASE = "C:/Users/fuguz/Documents/ProjetoPROG/ElementosDeIACD/TRABALHOPRATICO/analise_pordata/visualizacoes"
diretorio_visualizacoes = os.path.join(DIRETORIO_BASE, "visualizacoes")

# Criar diretório para salvar visualizações
os.makedirs(diretorio_visualizacoes, exist_ok=True)

# Definir paletas de cores personalizadas
paleta_portugal = ['#006600', '#FF0000']
paleta_europa = sns.color_palette("Blues_r", 10)
paleta_correlacao = sns.diverging_palette(240, 10, as_cmap=True)
paleta_multipla = sns.color_palette("husl", 10)


# Carregar datasets limpos
def carregar_datasets():
    """Carrega os datasets limpos e retorna um dicionário com os dataframes."""
    # Carrega os datasets limpos diretamente do módulo `limpeza_dados`
    datasets = limpeza_dados.datasets_limpos  # Acessa o dicionário de datasets limpos do módulo de limpeza

    # Verificar se os datasets foram carregados corretamente
    if not datasets:
        print("Nenhum dataset carregado. Verificando arquivos CSV limpos...")
        datasets = {}

        # Lista de nomes dos datasets
        nomes_datasets = [
            'GANHOMEDIOMENSAL',
            'ESPERANÇADEVIDA',
            'DESPESASAUDE',
            'PERCEÇAODESAUDE',
            'TAXADEMORTALIDADEVITAVEL'
        ]

        # Carregar cada dataset a partir dos arquivos CSV limpos
        for nome in nomes_datasets:
            try:
                caminho = os.path.join(DIRETORIO_BASE, f'{nome}_limpo.csv')
                datasets[nome] = pd.read_csv(caminho, encoding='utf-8')
                print(f"Dataset {nome} carregado: {datasets[nome].shape[0]} registros")
            except Exception as e:
                print(f"Erro ao carregar {nome}: {e}")

    return datasets


# Função para criar dataframe integrado
def criar_dataframe_integrado(datasets):
    """
    Cria um dataframe integrado com dados de Portugal para todos os indicadores,
    para o período comum a todos os datasets.
    """
    # Filtrar apenas dados de Portugal
    dfs_portugal = {}
    for nome, df in datasets.items():
        # Verificar se 'Portugal' está na coluna 'Pais'
        if 'Pais' in df.columns and 'Portugal' in df['Pais'].values:
            dfs_portugal[nome] = df[df['Pais'] == 'Portugal'].copy()
        # Caso contrário, verificar se há dados para Portugal na coluna 'Regiao'
        elif 'Regiao' in df.columns and 'Portugal' in df['Regiao'].values:
            dfs_portugal[nome] = df[df['Regiao'] == 'Portugal'].copy()
        else:
            print(f"Não foram encontrados dados para Portugal no dataset {nome}")

    # Identificar o período comum a todos os datasets
    anos_por_dataset = {nome: set(df['Ano'].unique()) for nome, df in dfs_portugal.items()}
    anos_comuns = set.intersection(*anos_por_dataset.values()) if anos_por_dataset else set()

    if not anos_comuns:
        print("Não há anos comuns entre os datasets para Portugal")
        # Usar o maior período possível para cada par de datasets
        df_integrado = pd.DataFrame(columns=['Ano'])
        return df_integrado

    # Criar dataframe base com os anos comuns
    df_integrado = pd.DataFrame({'Ano': sorted(list(anos_comuns))})

    # Adicionar valores de cada indicador
    for nome, df in dfs_portugal.items():
        # Filtrar para anos comuns
        df_filtrado = df[df['Ano'].isin(anos_comuns)]

        # Agrupar por ano (caso haja múltiplos registros por ano) e obter a média
        df_agrupado = df_filtrado.groupby('Ano')['Valor'].mean().reset_index()

        # Mesclar com o dataframe integrado
        df_integrado = pd.merge(df_integrado, df_agrupado, on='Ano', how='left', suffixes=('', f'_{nome}'))

        # Renomear a coluna de valor
        df_integrado.rename(columns={'Valor': nome}, inplace=True)

    return df_integrado


# Função para criar gráfico de evolução temporal múltipla
def criar_grafico_evolucao_multipla(df_integrado):
    """
    Cria um gráfico de evolução temporal para todos os indicadores,
    normalizando os valores para permitir comparação.
    """
    # Verificar se há dados suficientes
    if df_integrado.empty or df_integrado.shape[1] <= 1:
        print("Dados insuficientes para criar gráfico de evolução múltipla")
        return

    # Criar cópia do dataframe para normalização
    df_norm = df_integrado.copy()

    # Normalizar cada indicador (min-max scaling)
    for col in df_norm.columns:
        if col != 'Ano' and not df_norm[col].isna().all():
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val > min_val:  # Evitar divisão por zero
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)

    # Criar figura
    plt.figure(figsize=(12, 8))

    # Plotar cada indicador
    for i, col in enumerate(df_norm.columns):
        if col != 'Ano' and not df_norm[col].isna().all():
            plt.plot(df_norm['Ano'], df_norm[col], marker='o', linewidth=2,
                     label=col, color=paleta_multipla[i % len(paleta_multipla)])

    # Configurar gráfico
    plt.title('Evolução Temporal Normalizada dos Indicadores em Portugal', fontsize=16)
    plt.xlabel('Ano', fontsize=14)
    plt.ylabel('Valor Normalizado (0-1)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12, loc='best')
    plt.tight_layout()

    # Salvar gráfico
    plt.savefig(os.path.join(diretorio_visualizacoes, 'evolucao_temporal_multipla.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Gráfico de evolução temporal múltipla criado com sucesso")

# Função para criar dataframe integrado para comparação entre países
def criar_dataframe_paises(datasets, ano_referencia=None):
    """
    Cria um dataframe integrado com dados do ano mais recente comum a todos os datasets,
    para comparação entre países.
    """
    # Filtrar apenas dados por país (não por região)
    dfs_paises = {}
    for nome, df in datasets.items():
        if 'Pais' in df.columns:
            # Remover linhas onde País é nulo
            df_filtrado = df[df['Pais'].notna()].copy()
            if not df_filtrado.empty:
                dfs_paises[nome] = df_filtrado

    # Se não houver datasets com dados por país, retornar dataframe vazio
    if not dfs_paises:
        print("Não foram encontrados datasets com dados por país")
        return pd.DataFrame()

    # Identificar países comuns a todos os datasets
    paises_por_dataset = {nome: set(df['Pais'].unique()) for nome, df in dfs_paises.items()}
    paises_comuns = set.intersection(*paises_por_dataset.values())

    # Se não houver países comuns, usar todos os países disponíveis
    if not paises_comuns:
        print("Não há países comuns entre os datasets, usando todos os países disponíveis")
        todos_paises = set()
        for paises in paises_por_dataset.values():
            todos_paises.update(paises)
        paises_comuns = todos_paises

    # Determinar o ano de referência (mais recente comum a todos os datasets)
    if ano_referencia is None:
        anos_por_dataset = {nome: set(df['Ano'].unique()) for nome, df in dfs_paises.items()}
        anos_comuns = set.intersection(*anos_por_dataset.values())
        if anos_comuns:
            ano_referencia = max(anos_comuns)
        else:
            # Se não houver anos comuns, usar o ano mais recente de cada dataset
            anos_recentes = {nome: max(df['Ano']) for nome, df in dfs_paises.items()}
            # Usar a mediana dos anos mais recentes
            ano_referencia = int(np.median(list(anos_recentes.values())))

    print(f"Usando ano de referência: {ano_referencia}")

    # Criar dataframe base com os países
    df_paises = pd.DataFrame({'Pais': sorted(list(paises_comuns))})

    # Adicionar valores de cada indicador para o ano de referência
    for nome, df in dfs_paises.items():
        # Filtrar para o ano de referência ou o ano mais próximo
        anos_disponiveis = sorted(df['Ano'].unique())
        if ano_referencia in anos_disponiveis:
            ano_usado = ano_referencia
        else:
            # Encontrar o ano mais próximo disponível
            anos_disponiveis = np.array(anos_disponiveis)
            idx = (np.abs(anos_disponiveis - ano_referencia)).argmin()
            ano_usado = anos_disponiveis[idx]
            print(f"Para {nome}, usando ano {ano_usado} em vez de {ano_referencia}")

        # Filtrar para o ano usado
        df_filtrado = df[df['Ano'] == ano_usado]

        # Filtrar para países comuns e obter a média por país
        df_filtrado = df_filtrado[df_filtrado['Pais'].isin(paises_comuns)]
        df_agrupado = df_filtrado.groupby('Pais')['Valor'].mean().reset_index()

        # Mesclar com o dataframe de países
        df_paises = pd.merge(df_paises, df_agrupado, on='Pais', how='left', suffixes=('', f'_{nome}'))

        # Renomear a coluna de valor
        df_paises.rename(columns={'Valor': nome}, inplace=True)

    return df_paises

# Função para criar gráficos de dispersão com regressão
def criar_graficos_dispersao_regressao(df_integrado, df_paises):
    """
    Cria gráficos de dispersão com linha de regressão para pares de indicadores
    com correlações significativas.
    """
    # Lista de pares de indicadores a analisar
    pares_indicadores = [
        ('GANHOMEDIOMENSAL', 'DESPESASAUDE'),
        ('GANHOMEDIOMENSAL', 'PERCEÇAODESAUDE'),
        ('ESPERANÇADEVIDA', 'TAXADEMORTALIDADEVITAVEL'),
        ('DESPESASAUDE', 'PERCEÇAODESAUDE'),
        ('ESPERANÇADEVIDA', 'DESPESASAUDE')
    ]

    # Criar gráficos para evolução temporal (usando df_integrado)
    for x_col, y_col in pares_indicadores:
        # Verificar se ambas as colunas existem e têm dados suficientes
        if x_col in df_integrado.columns and y_col in df_integrado.columns:
            # Filtrar linhas sem valores nulos
            df_filtrado = df_integrado.dropna(subset=[x_col, y_col])

            if len(df_filtrado) >= 5:  # Verificar se há dados suficientes
                # Criar figura
                plt.figure(figsize=(10, 8))

                # Plotar pontos
                sns.regplot(x=x_col, y=y_col, data=df_filtrado,
                            scatter_kws={'s': 80, 'alpha': 0.7},
                            line_kws={'color': 'red', 'linewidth': 2},
                            ci=95)

                # Calcular correlação
                corr, p_value = stats.pearsonr(df_filtrado[x_col], df_filtrado[y_col])

                # Calcular regressão linear
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    df_filtrado[x_col], df_filtrado[y_col]
                )

                # Adicionar equação da reta e R²
                equation = f"y = {slope:.2f}x + {intercept:.2f}"
                r_squared = f"R² = {r_value ** 2:.3f}"
                correlation = f"Correlação: {corr:.3f}"

                plt.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction',
                             fontsize=12, ha='left', va='top')
                plt.annotate(r_squared, xy=(0.05, 0.90), xycoords='axes fraction',
                             fontsize=12, ha='left', va='top')
                plt.annotate(correlation, xy=(0.05, 0.85), xycoords='axes fraction',
                             fontsize=12, ha='left', va='top')

                # Configurar gráfico
                plt.title(f'Relação entre {x_col} e {y_col} em Portugal (Evolução Temporal)',
                          fontsize=16)
                plt.xlabel(x_col, fontsize=14)
                plt.ylabel(y_col, fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()

                # Salvar gráfico
                plt.savefig(os.path.join(diretorio_visualizacoes, f'dispersao_temporal_{x_col}_{y_col}.png'),
                            dpi=300, bbox_inches='tight')
                plt.close()

                print(f"Gráfico de dispersão temporal entre {x_col} e {y_col} criado com sucesso")

    # Criar gráficos para comparação entre países (usando df_paises)
    for x_col, y_col in pares_indicadores:
        # Verificar se ambas as colunas existem e têm dados suficientes
        if x_col in df_paises.columns and y_col in df_paises.columns:
            # Filtrar linhas sem valores nulos
            df_filtrado = df_paises.dropna(subset=[x_col, y_col])

            if len(df_filtrado) >= 5:  # Verificar se há dados suficientes
                # Criar figura
                plt.figure(figsize=(12, 10))

                # Plotar pontos
                scatter = plt.scatter(df_filtrado[x_col], df_filtrado[y_col],
                                      s=100, alpha=0.7, c='blue')

                # Adicionar rótulos para cada país
                for i, row in df_filtrado.iterrows():
                    plt.annotate(row['Pais'],
                                 xy=(row[x_col], row[y_col]),
                                 xytext=(5, 5),
                                 textcoords='offset points',
                                 fontsize=10,
                                 alpha=0.8)

                # Calcular correlação
                corr, p_value = stats.pearsonr(df_filtrado[x_col], df_filtrado[y_col])

                # Calcular regressão linear
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    df_filtrado[x_col], df_filtrado[y_col]
                )

                # Plotar linha de regressão
                x_range = np.linspace(df_filtrado[x_col].min(), df_filtrado[x_col].max(), 100)
                y_pred = slope * x_range + intercept
                plt.plot(x_range, y_pred, 'r-', linewidth=2)

                # Adicionar equação da reta e R²
                equation = f"y = {slope:.2f}x + {intercept:.2f}"
                r_squared = f"R² = {r_value ** 2:.3f}"
                correlation = f"Correlação: {corr:.3f}"
                p_value_text = f"p-valor: {p_value:.4f}"

                plt.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction',
                             fontsize=12, ha='left', va='top')
                plt.annotate(r_squared, xy=(0.05, 0.90), xycoords='axes fraction',
                             fontsize=12, ha='left', va='top')
                plt.annotate(correlation, xy=(0.05, 0.85), xycoords='axes fraction',
                             fontsize=12, ha='left', va='top')
                plt.annotate(p_value_text, xy=(0.05, 0.80), xycoords='axes fraction',
                             fontsize=12, ha='left', va='top')

                # Destacar Portugal
                if 'Portugal' in df_filtrado['Pais'].values:
                    portugal_data = df_filtrado[df_filtrado['Pais'] == 'Portugal']
                    plt.scatter(portugal_data[x_col], portugal_data[y_col],
                                s=150, color='red', edgecolors='black', zorder=5,
                                label='Portugal')
                    plt.legend(fontsize=12)

                # Configurar gráfico
                plt.title(f'Relação entre {x_col} e {y_col} por País', fontsize=16)
                plt.xlabel(x_col, fontsize=14)
                plt.ylabel(y_col, fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()

                # Salvar gráfico
                plt.savefig(os.path.join(diretorio_visualizacoes, f'dispersao_paises_{x_col}_{y_col}.png'),
                            dpi=300, bbox_inches='tight')
                plt.close()

                print(f"Gráfico de dispersão entre países para {x_col} e {y_col} criado com sucesso")


# Função para criar gráficos de barras comparativos
def criar_graficos_barras_comparativos(df_paises):
    """
    Cria gráficos de barras comparando os valores mais recentes de cada indicador
    entre diferentes países europeus.
    """
    # Verificar se há dados suficientes
    if df_paises.empty or df_paises.shape[1] <= 1:
        print("Dados insuficientes para criar gráficos de barras comparativos")
        return

    # Criar gráfico para cada indicador
    for col in df_paises.columns:
        if col != 'Pais' and not df_paises[col].isna().all():
            # Filtrar linhas sem valores nulos
            df_filtrado = df_paises.dropna(subset=[col]).copy()

            if len(df_filtrado) >= 3:  # Verificar se há dados suficientes
                # Ordenar por valor
                df_filtrado = df_filtrado.sort_values(by=col, ascending=False)

                # Limitar a 15 países para melhor visualização
                if len(df_filtrado) > 15:
                    # Garantir que Portugal esteja incluído
                    if 'Portugal' in df_filtrado['Pais'].values:
                        portugal_idx = df_filtrado[df_filtrado['Pais'] == 'Portugal'].index[0]
                        portugal_rank = df_filtrado.index.get_loc(portugal_idx)

                        # Se Portugal estiver fora dos top 14, incluí-lo e os top 14
                        if portugal_rank >= 14:
                            top_countries = df_filtrado.iloc[:14]
                            portugal_data = df_filtrado[df_filtrado['Pais'] == 'Portugal']
                            df_filtrado = pd.concat([top_countries, portugal_data])
                        else:
                            # Caso contrário, apenas os top 15
                            df_filtrado = df_filtrado.iloc[:15]
                    else:
                        # Se Portugal não estiver nos dados, apenas os top 15
                        df_filtrado = df_filtrado.iloc[:15]

                # Criar figura
                plt.figure(figsize=(14, 8))

                # Criar paleta de cores com destaque para Portugal
                cores = ['#1f77b4'] * len(df_filtrado)  # Cor padrão para todos os países
                if 'Portugal' in df_filtrado['Pais'].values:
                    portugal_idx = df_filtrado[df_filtrado['Pais'] == 'Portugal'].index
                    for idx in portugal_idx:
                        cores[df_filtrado.index.get_loc(idx)] = '#d62728'  # Vermelho para Portugal

                # Plotar barras
                barras = plt.bar(df_filtrado['Pais'], df_filtrado[col], color=cores)

                # Adicionar valores sobre as barras
                for barra in barras:
                    altura = barra.get_height()
                    plt.text(barra.get_x() + barra.get_width() / 2., altura + 0.01 * max(df_filtrado[col]),
                             f'{altura:.1f}', ha='center', va='bottom', rotation=0, fontsize=9)

                # Configurar gráfico
                plt.title(f'Comparação de {col} entre Países Europeus', fontsize=16)
                plt.xlabel('País', fontsize=14)
                plt.ylabel(col, fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, linestyle='--', alpha=0.7, axis='y')
                plt.tight_layout()

                # Salvar gráfico
                plt.savefig(os.path.join(diretorio_visualizacoes, f'barras_comparativo_{col}.png'),
                            dpi=300, bbox_inches='tight')
                plt.close()

                print(f"Gráfico de barras comparativo para {col} criado com sucesso")


# Função para criar gráficos de área para evolução temporal
def criar_graficos_area_temporal(datasets):
    """
    Cria gráficos de área para visualizar a evolução temporal de cada indicador em Portugal.
    """
    # Para cada dataset
    for nome, df in datasets.items():
        # Verificar se há dados para Portugal
        df_portugal = None

        # Verificar se 'Portugal' está na coluna 'Pais'
        if 'Pais' in df.columns and 'Portugal' in df['Pais'].values:
            df_portugal = df[df['Pais'] == 'Portugal'].copy()
        # Caso contrário, verificar se há dados para Portugal na coluna 'Regiao'
        elif 'Regiao' in df.columns and 'Portugal' in df['Regiao'].values:
            df_portugal = df[df['Regiao'] == 'Portugal'].copy()

        if df_portugal is not None and not df_portugal.empty:
            # Agrupar por ano e calcular média
            df_agrupado = df_portugal.groupby('Ano')['Valor'].mean().reset_index()

            if len(df_agrupado) >= 5:  # Verificar se há dados suficientes
                # Criar figura
                plt.figure(figsize=(12, 8))

                # Plotar área
                plt.fill_between(df_agrupado['Ano'], df_agrupado['Valor'],
                                 color=paleta_portugal[0], alpha=0.6)
                plt.plot(df_agrupado['Ano'], df_agrupado['Valor'],
                         color=paleta_portugal[1], linewidth=2, marker='o')

                # Configurar gráfico
                plt.title(f'Evolução de {nome} em Portugal', fontsize=16)
                plt.xlabel('Ano', fontsize=14)
                plt.ylabel('Valor', fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.7)

                # Adicionar anotações para valores inicial e final
                if len(df_agrupado) > 1:
                    primeiro_ano = df_agrupado['Ano'].iloc[0]
                    ultimo_ano = df_agrupado['Ano'].iloc[-1]
                    primeiro_valor = df_agrupado['Valor'].iloc[0]
                    ultimo_valor = df_agrupado['Valor'].iloc[-1]

                    # Calcular variação percentual
                    variacao_pct = ((ultimo_valor - primeiro_valor) / primeiro_valor) * 100

                    # Adicionar anotações
                    plt.annotate(f'{primeiro_valor:.2f}',
                                 xy=(primeiro_ano, primeiro_valor),
                                 xytext=(0, 10),
                                 textcoords='offset points',
                                 ha='center',
                                 fontsize=10,
                                 bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))

                    plt.annotate(f'{ultimo_valor:.2f}\n({variacao_pct:+.1f}%)',
                                 xy=(ultimo_ano, ultimo_valor),
                                 xytext=(0, 10),
                                 textcoords='offset points',
                                 ha='center',
                                 fontsize=10,
                                 bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))

                plt.tight_layout()

                # Salvar gráfico
                plt.savefig(os.path.join(diretorio_visualizacoes, f'area_temporal_{nome}.png'),
                            dpi=300, bbox_inches='tight')
                plt.close()

                print(f"Gráfico de área temporal para {nome} criado com sucesso")


# Função para criar dashboard integrado
def criar_dashboard_integrado(df_integrado, df_paises):
    """
    Cria um dashboard que combina múltiplas visualizações em uma única figura.
    """
    # Verificar se há dados suficientes
    if df_integrado.empty or df_paises.empty:
        print("Dados insuficientes para criar dashboard integrado")
        return

    # Criar figura com grid layout
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig)

    # 1. Gráfico de evolução temporal múltipla (topo esquerda)
    ax1 = fig.add_subplot(gs[0, 0])

    # Normalizar cada indicador para o gráfico de evolução múltipla
    df_norm = df_integrado.copy()
    for col in df_norm.columns:
        if col != 'Ano' and not df_norm[col].isna().all():
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val > min_val:  # Evitar divisão por zero
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)

    # Plotar cada indicador normalizado
    for i, col in enumerate(df_norm.columns):
        if col != 'Ano' and not df_norm[col].isna().all():
            ax1.plot(df_norm['Ano'], df_norm[col], marker='o', linewidth=2,
                     label=col, color=paleta_multipla[i % len(paleta_multipla)])

    ax1.set_title('Evolução Temporal Normalizada', fontsize=14)
    ax1.set_xlabel('Ano', fontsize=12)
    ax1.set_ylabel('Valor Normalizado (0-1)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=8, loc='best')

    # 2. Mapa de calor de correlação (topo centro)
    ax2 = fig.add_subplot(gs[0, 1])

    # Calcular matriz de correlação
    colunas_numericas = [col for col in df_integrado.columns if col != 'Ano']
    if len(colunas_numericas) >= 2:
        corr_matrix = df_integrado[colunas_numericas].corr()

        # Plotar mapa de calor
        sns.heatmap(corr_matrix, annot=True, cmap=paleta_correlacao,
                    vmin=-1, vmax=1, center=0, linewidths=.5,
                    annot_kws={"size": 10}, ax=ax2)

        ax2.set_title('Matriz de Correlação', fontsize=14)
    else:
        ax2.text(0.5, 0.5, "Dados insuficientes para matriz de correlação",
                 ha='center', va='center', fontsize=12)

    # 3. Gráfico de barras comparativo (topo direita)
    ax3 = fig.add_subplot(gs[0, 2])

    # Escolher um indicador para o gráfico de barras
    indicador_barras = None
    for col in df_paises.columns:
        if col != 'Pais' and not df_paises[col].isna().all():
            indicador_barras = col
            break

    if indicador_barras:
        # Filtrar linhas sem valores nulos
        df_filtrado = df_paises.dropna(subset=[indicador_barras]).copy()

        if len(df_filtrado) >= 3:
            # Ordenar por valor e limitar a 10 países
            df_filtrado = df_filtrado.sort_values(by=indicador_barras, ascending=False)
            if len(df_filtrado) > 10:
                # Garantir que Portugal esteja incluído
                if 'Portugal' in df_filtrado['Pais'].values:
                    portugal_idx = df_filtrado[df_filtrado['Pais'] == 'Portugal'].index[0]
                    portugal_rank = df_filtrado.index.get_loc(portugal_idx)

                    if portugal_rank >= 9:
                        top_countries = df_filtrado.iloc[:9]
                        portugal_data = df_filtrado[df_filtrado['Pais'] == 'Portugal']
                        df_filtrado = pd.concat([top_countries, portugal_data])
                    else:
                        df_filtrado = df_filtrado.iloc[:10]
                else:
                    df_filtrado = df_filtrado.iloc[:10]

            # Criar paleta de cores com destaque para Portugal
            cores = ['#1f77b4'] * len(df_filtrado)
            if 'Portugal' in df_filtrado['Pais'].values:
                portugal_idx = df_filtrado[df_filtrado['Pais'] == 'Portugal'].index
                for idx in portugal_idx:
                    cores[df_filtrado.index.get_loc(idx)] = '#d62728'

            # Plotar barras
            barras = ax3.bar(df_filtrado['Pais'], df_filtrado[indicador_barras], color=cores)

            ax3.set_title(f'Comparação de {indicador_barras}', fontsize=14)
            ax3.set_xlabel('País', fontsize=12)
            ax3.set_ylabel(indicador_barras, fontsize=12)
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, linestyle='--', alpha=0.7, axis='y')
    else:
        ax3.text(0.5, 0.5, "Dados insuficientes para gráfico de barras",
                 ha='center', va='center', fontsize=12)

    # 4-6. Gráficos de dispersão com regressão (linha do meio)
    # Definir pares de indicadores para os gráficos de dispersão
    pares_dispersao = [
        ('GANHOMEDIOMENSAL', 'DESPESASAUDE'),
        ('ESPERANÇADEVIDA', 'TAXADEMORTALIDADEVITAVEL'),
        ('DESPESASAUDE', 'PERCEÇAODESAUDE')
    ]

    for i, (x_col, y_col) in enumerate(pares_dispersao):
        ax = fig.add_subplot(gs[1, i])

        # Verificar se ambas as colunas existem em df_paises
        if x_col in df_paises.columns and y_col in df_paises.columns:
            # Filtrar linhas sem valores nulos
            df_filtrado = df_paises.dropna(subset=[x_col, y_col])

            if len(df_filtrado) >= 5:
                # Plotar pontos
                ax.scatter(df_filtrado[x_col], df_filtrado[y_col],
                           s=50, alpha=0.7, c='blue')

                # Destacar Portugal
                if 'Portugal' in df_filtrado['Pais'].values:
                    portugal_data = df_filtrado[df_filtrado['Pais'] == 'Portugal']
                    ax.scatter(portugal_data[x_col], portugal_data[y_col],
                               s=100, color='red', edgecolors='black', zorder=5)

                # Calcular regressão linear
                if len(df_filtrado) > 1:  # Precisa de pelo menos 2 pontos
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        df_filtrado[x_col], df_filtrado[y_col]
                    )

                    # Plotar linha de regressão
                    x_range = np.linspace(df_filtrado[x_col].min(), df_filtrado[x_col].max(), 100)
                    y_pred = slope * x_range + intercept
                    ax.plot(x_range, y_pred, 'r-', linewidth=2)

                    # Adicionar R²
                    ax.annotate(f"R² = {r_value ** 2:.2f}", xy=(0.05, 0.95),
                                xycoords='axes fraction', fontsize=10)

                ax.set_title(f'{x_col} vs {y_col}', fontsize=14)
                ax.set_xlabel(x_col, fontsize=10)
                ax.set_ylabel(y_col, fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.7)
            else:
                ax.text(0.5, 0.5, "Dados insuficientes", ha='center', va='center', fontsize=12)
        else:
            ax.text(0.5, 0.5, "Indicadores não disponíveis", ha='center', va='center', fontsize=12)

    # 7-9. Gráficos de tendência temporal para indicadores individuais (linha inferior)
    indicadores = [col for col in df_integrado.columns if col != 'Ano'][:3]  # Limitar a 3

    for i, indicador in enumerate(indicadores):
        ax = fig.add_subplot(gs[2, i])

        if indicador in df_integrado.columns:
            # Filtrar linhas sem valores nulos
            df_filtrado = df_integrado.dropna(subset=[indicador])

            if len(df_filtrado) >= 3:
                # Plotar linha e área
                ax.fill_between(df_filtrado['Ano'], df_filtrado[indicador],
                                color=paleta_portugal[0], alpha=0.3)
                ax.plot(df_filtrado['Ano'], df_filtrado[indicador],
                        color=paleta_portugal[1], linewidth=2, marker='o')

                # Calcular tendência
                if len(df_filtrado) > 2:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        df_filtrado['Ano'], df_filtrado[indicador]
                    )

                    # Determinar direção da tendência
                    if slope > 0 and p_value < 0.05:
                        tendencia = "Crescente"
                    elif slope < 0 and p_value < 0.05:
                        tendencia = "Decrescente"
                    else:
                        tendencia = "Estável"

                    # Adicionar informação de tendência
                    ax.annotate(f"Tendência: {tendencia}", xy=(0.05, 0.95),
                                xycoords='axes fraction', fontsize=10)

                    # Adicionar linha de tendência
                    x_range = np.array([df_filtrado['Ano'].min(), df_filtrado['Ano'].max()])
                    y_pred = slope * x_range + intercept
                    ax.plot(x_range, y_pred, 'k--', linewidth=1.5)

                ax.set_title(f'Tendência de {indicador}', fontsize=14)
                ax.set_xlabel('Ano', fontsize=12)
                ax.set_ylabel(indicador, fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.7)
            else:
                ax.text(0.5, 0.5, "Dados insuficientes", ha='center', va='center', fontsize=12)
        else:
            ax.text(0.5, 0.5, "Indicador não disponível", ha='center', va='center', fontsize=12)

    # Ajustar layout e adicionar título geral
    plt.suptitle('Dashboard de Indicadores Socioeconômicos da PORDATA', fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Salvar dashboard
    plt.savefig(os.path.join(diretorio_visualizacoes, 'dashboard_indicadores.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Dashboard integrado criado com sucesso")


# Função para criar gráfico de tendências temporais
def criar_grafico_tendencias_temporais(df_integrado):
    """
    Cria um gráfico de tendências temporais para todos os indicadores em Portugal.
    """
    # Verificar se há dados suficientes
    if df_integrado.empty or df_integrado.shape[1] <= 1:
        print("Dados insuficientes para criar gráfico de tendências temporais")
        return

    # Criar figura
    plt.figure(figsize=(14, 10))

    # Definir cores para cada indicador
    cores = paleta_multipla

    # Plotar cada indicador em um subplot separado
    indicadores = [col for col in df_integrado.columns if col != 'Ano']
    n_indicadores = len(indicadores)

    if n_indicadores == 0:
        print("Nenhum indicador disponível para plotar")
        return

    # Determinar número de linhas e colunas para subplots
    n_cols = min(3, n_indicadores)
    n_rows = (n_indicadores + n_cols - 1) // n_cols  # Arredondamento para cima

    # Criar subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=True)

    # Converter axes para array 2D se necessário
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Plotar cada indicador
    for i, indicador in enumerate(indicadores):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # Filtrar dados válidos
        df_filtrado = df_integrado.dropna(subset=[indicador])

        if len(df_filtrado) >= 3:
            # Plotar dados
            ax.plot(df_filtrado['Ano'], df_filtrado[indicador],
                    marker='o', linewidth=2, color=cores[i % len(cores)])

            # Calcular tendência linear
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df_filtrado['Ano'], df_filtrado[indicador]
            )

            # Plotar linha de tendência
            x_range = np.array([df_filtrado['Ano'].min(), df_filtrado['Ano'].max()])
            y_pred = slope * x_range + intercept
            ax.plot(x_range, y_pred, 'r--', linewidth=1.5)

            # Adicionar informações de tendência
            if p_value < 0.05:
                tendencia = "Crescente" if slope > 0 else "Decrescente"
                significancia = "significativa"
            else:
                tendencia = "Estável"
                significancia = "não significativa"

            ax.text(0.05, 0.95, f"Tendência: {tendencia}\n(p={p_value:.4f}, {significancia})",
                    transform=ax.transAxes, fontsize=10, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # Configurar subplot
            ax.set_title(indicador, fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)

            # Adicionar rótulos apenas para subplots na borda esquerda e inferior
            if col == 0:
                ax.set_ylabel('Valor', fontsize=10)
            if row == n_rows - 1:
                ax.set_xlabel('Ano', fontsize=10)
        else:
            ax.text(0.5, 0.5, "Dados insuficientes", ha='center', va='center', fontsize=12)

    # Ocultar subplots vazios
    for i in range(n_indicadores, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axes[row, col])

    # Ajustar layout
    plt.suptitle('Tendências Temporais dos Indicadores em Portugal', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Salvar gráfico
    plt.savefig(os.path.join(diretorio_visualizacoes, 'tendencias_temporais.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Gráfico de tendências temporais criado com sucesso")


# Função principal
def main():
    """Função principal para executar todas as visualizações."""
    print("Iniciando criação de visualizações avançadas...")

    # Carregar datasets
    datasets = carregar_datasets()

    # Verificar se há datasets carregados
    if not datasets:
        print("Nenhum dataset foi carregado. Verifique os arquivos CSV.")
        return

    # Criar dataframes integrados
    print("\nCriando dataframe integrado para Portugal...")
    df_integrado = criar_dataframe_integrado(datasets)

    print("\nCriando dataframe integrado para comparação entre países...")
    df_paises = criar_dataframe_paises(datasets)

    # Criar visualizações
    print("\nCriando visualizações avançadas...")

    # 1. Gráfico de evolução temporal múltipla
    print("\n1. Criando gráfico de evolução temporal múltipla...")
    criar_grafico_evolucao_multipla(df_integrado)

    # 2. Gráficos de dispersão com regressão
    print("\n2. Criando gráficos de dispersão com regressão...")
    criar_graficos_dispersao_regressao(df_integrado, df_paises)

    # 3. Gráficos de barras comparativos
    print("\n3. Criando gráficos de barras comparativos...")
    criar_graficos_barras_comparativos(df_paises)

    # 4. Gráficos de área para evolução temporal
    print("\n4. Criando gráficos de área para evolução temporal...")
    criar_graficos_area_temporal(datasets)

    # 5. Dashboard integrado
    print("\n5. Criando dashboard integrado...")
    criar_dashboard_integrado(df_integrado, df_paises)

    # 6. Gráfico de tendências temporais
    print("\n6. Criando gráfico de tendências temporais...")
    criar_grafico_tendencias_temporais(df_integrado)

    print("\nVisualizações avançadas criadas com sucesso!")
    print(f"Todos os gráficos foram salvos no diretório: {os.path.abspath(diretorio_visualizacoes)}")


# Executar o script
if __name__ == "__main__":
    main()
