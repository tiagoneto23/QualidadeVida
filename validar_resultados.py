import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configurações para visualizações
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_palette('viridis')

# Diretório base (na mesma pasta do script)
diretorio_base = os.path.join(os.getcwd(), "analise_pordata")

# Diretórios internos
diretorio_visualizacoes = os.path.join(diretorio_base, "visualizacoes")
diretorio_resultados = os.path.join(diretorio_base, "resultados")

# Criar diretórios, se não existirem
os.makedirs(diretorio_visualizacoes, exist_ok=True)
os.makedirs(diretorio_resultados, exist_ok=True)

diretorio_validacao = os.path.join(diretorio_base, "validacao")
os.makedirs(diretorio_validacao, exist_ok=True)

def validar_resultados():
    """
    Função principal para validar os resultados da análise.
    Realiza verificações de consistência, robustez e validade dos resultados.
    """
    print("Iniciando validação dos resultados da análise...")

    # 1. Verificar a existência dos arquivos de resultados
    verificar_arquivos_resultados()

    # 2. Validar a consistência dos dados limpos
    validar_consistencia_dados()

    # 3. Verificar a robustez das correlações
    verificar_robustez_correlacoes()

    # 4. Validar as análises de regressão
    validar_analises_regressao()

    print("Validação concluída. Resultados salvos em:", diretorio_validacao)


def verificar_arquivos_resultados():
    """
    Verifica se todos os arquivos de resultados esperados existem.
    """
    print("\n1. Verificando a existência dos arquivos de resultados...")

    # Lista de arquivos esperados
    arquivos_esperados = [
        # Datasets limpos
        "C:/Users/fuguz/Documents/ProjetoPROG/ElementosDeIACD/TRABALHOPRATICO/GANHOMEDIOMENSAL_limpo.csv",
        "C:/Users/fuguz/Documents/ProjetoPROG/ElementosDeIACD/TRABALHOPRATICO/ESPERANÇADEVIDA_limpo.csv",
        "C:/Users/fuguz/Documents/ProjetoPROG/ElementosDeIACD/TRABALHOPRATICO/DESPESASAUDE_limpo.csv",
        "C:/Users/fuguz/Documents/ProjetoPROG/ElementosDeIACD/TRABALHOPRATICO/PERCEÇAODESAUDE_limpo.csv",
        "C:/Users/fuguz/Documents/ProjetoPROG/ElementosDeIACD/TRABALHOPRATICO/TAXADEMORTALIDADEVITAVEL_limpo.csv",

        # Resumos e análises
        "C:/Users/fuguz/Documents/ProjetoPROG/ElementosDeIACD/TRABALHOPRATICO/resumo_datasets.txt",
        "C:/Users/fuguz/Documents/ProjetoPROG/ElementosDeIACD/TRABALHOPRATICO/resumo_limpeza.txt",

        # Relatório final
        "C:/Users/fuguz/Documents/ProjetoPROG/ElementosDeIACD/TRABALHOPRATICO/relatorio_final.txt"
    ]

    # Verificar cada arquivo
    arquivos_faltantes = []
    for arquivo in arquivos_esperados:
        if not os.path.exists(arquivo):
            arquivos_faltantes.append(arquivo)

    # Reportar resultados
    if arquivos_faltantes:
        print(f"ATENÇÃO: {len(arquivos_faltantes)} arquivos esperados não foram encontrados:")
        for arquivo in arquivos_faltantes:
            print(f"  - {arquivo}")
    else:
        print("Todos os arquivos de resultados esperados foram encontrados.")

    # Salvar resultados
    with open(os.path.join(diretorio_validacao, "verificacao_arquivos.txt"), "w") as f:
        f.write("VERIFICAÇÃO DE ARQUIVOS DE RESULTADOS\n")
        f.write("=" * 50 + "\n\n")

        if arquivos_faltantes:
            f.write(f"{len(arquivos_faltantes)} arquivos esperados não foram encontrados:\n")
            for arquivo in arquivos_faltantes:
                f.write(f"  - {arquivo}\n")
        else:
            f.write("Todos os arquivos de resultados esperados foram encontrados.\n")

    return len(arquivos_faltantes) == 0


def validar_consistencia_dados():
    """
    Valida a consistência dos dados limpos.
    Verifica se os dados estão completos, se não há valores nulos inesperados,
    e se os tipos de dados estão corretos.
    """
    print("\n2. Validando a consistência dos dados limpos...")

    # Lista de datasets limpos
    datasets_limpos = [
        "GANHOMEDIOMENSAL.csv_limpo.csv",
        "ESPERANÇADEVIDA.csv_limpo.csv",
        "DESPESASAUDE.csv_limpo.csv",
        "PERCEÇAODESAUDE.csv_limpo.csv",
        "TAXADEMORTALIDADEVITAVEL.csv_limpo.csv"
    ]

    resultados_validacao = {}

    for arquivo in datasets_limpos:
        caminho = os.path.join("C:/Users/fuguz/Documents/ProjetoPROG/ElementosDeIACD/TRABALHOPRATICO", arquivo)

        if not os.path.exists(caminho):
            print(f"ERRO: Arquivo {caminho} não encontrado.")
            resultados_validacao[arquivo] = {
                "existe": False,
                "mensagem": "Arquivo não encontrado."
            }
            continue

        try:
            # Carregar dataset
            df = pd.read_csv(caminho)

            # Verificar valores nulos
            valores_nulos = df.isnull().sum()
            tem_nulos = valores_nulos.sum() > 0

            # Verificar tipos de dados
            tipo_ano = pd.api.types.is_integer_dtype(df['Ano']) or pd.api.types.is_float_dtype(df['Ano'])
            tipo_valor = pd.api.types.is_numeric_dtype(df['Valor'])

            # Verificar valores negativos em colunas que não deveriam ter
            valores_negativos = (df['Valor'] < 0).sum() if 'Valor' in df.columns else 0

            # Armazenar resultados
            resultados_validacao[arquivo] = {
                "existe": True,
                "dimensoes": df.shape,
                "valores_nulos": tem_nulos,
                "detalhe_nulos": valores_nulos.to_dict() if tem_nulos else "Nenhum valor nulo",
                "tipo_ano_correto": tipo_ano,
                "tipo_valor_correto": tipo_valor,
                "valores_negativos": valores_negativos,
                "mensagem": "OK" if not tem_nulos and tipo_ano and tipo_valor and valores_negativos == 0 else "Problemas encontrados"
            }

            print(f"Dataset {arquivo}: {df.shape[0]} linhas x {df.shape[1]} colunas")
            if tem_nulos:
                print(f"  ATENÇÃO: Valores nulos encontrados: {valores_nulos.sum()}")
            if not tipo_ano:
                print(f"  ATENÇÃO: Tipo de dados incorreto para a coluna 'Ano'")
            if not tipo_valor:
                print(f"  ATENÇÃO: Tipo de dados incorreto para a coluna 'Valor'")
            if valores_negativos > 0:
                print(f"  ATENÇÃO: {valores_negativos} valores negativos encontrados na coluna 'Valor'")

        except Exception as e:
            print(f"ERRO ao validar {arquivo}: {e}")
            resultados_validacao[arquivo] = {
                "existe": True,
                "mensagem": f"Erro ao validar: {e}"
            }

    # Salvar resultados
    with open(os.path.join(diretorio_validacao, "validacao_consistencia.txt"), "w") as f:
        f.write("VALIDAÇÃO DE CONSISTÊNCIA DOS DADOS\n")
        f.write("=" * 50 + "\n\n")

        for arquivo, resultado in resultados_validacao.items():
            f.write(f"Dataset: {arquivo}\n")
            f.write(f"  Existe: {resultado['existe']}\n")

            if resultado['existe']:
                if 'dimensoes' in resultado:
                    f.write(f"  Dimensões: {resultado['dimensoes'][0]} linhas x {resultado['dimensoes'][1]} colunas\n")
                if 'valores_nulos' in resultado:
                    f.write(f"  Valores nulos: {resultado['valores_nulos']}\n")
                    if resultado['valores_nulos']:
                        f.write(f"  Detalhe de nulos: {resultado['detalhe_nulos']}\n")
                if 'tipo_ano_correto' in resultado:
                    f.write(f"  Tipo de dados correto para 'Ano': {resultado['tipo_ano_correto']}\n")
                if 'tipo_valor_correto' in resultado:
                    f.write(f"  Tipo de dados correto para 'Valor': {resultado['tipo_valor_correto']}\n")
                if 'valores_negativos' in resultado:
                    f.write(f"  Valores negativos em 'Valor': {resultado['valores_negativos']}\n")

            f.write(f"  Mensagem: {resultado['mensagem']}\n\n")

    return resultados_validacao


def verificar_robustez_correlacoes():
    """
    Verifica a robustez das correlações encontradas.
    Realiza análises de sensibilidade para confirmar que as correlações
    não são resultado de outliers ou de um pequeno número de observações.
    """
    print("\n3. Verificando a robustez das correlações...")

    # Tentar carregar o dataframe integrado
    caminho_df_integrado = "C:/Users/fuguz/Documents/ProjetoPROG/ElementosDeIACD/TRABALHOPRATICO/dataframe_integrado.csv"

    if not os.path.exists(caminho_df_integrado):
        print(f"ERRO: Arquivo {caminho_df_integrado} não encontrado.")

        # Salvar resultado
        with open(os.path.join(diretorio_validacao, "verificacao_correlacoes.txt"), "w") as f:
            f.write("VERIFICAÇÃO DE ROBUSTEZ DAS CORRELAÇÕES\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"ERRO: Arquivo {caminho_df_integrado} não encontrado.\n")
            f.write("Não foi possível verificar a robustez das correlações.\n")

        return False

    try:
        # Carregar dataframe integrado
        df_integrado = pd.read_csv(caminho_df_integrado)

        # Verificar se há dados suficientes
        if len(df_integrado) < 5:
            print("ATENÇÃO: Poucos dados para análise robusta de correlações.")

            # Salvar resultado
            with open(os.path.join(diretorio_validacao, "verificacao_correlacoes.txt"), "w") as f:
                f.write("VERIFICAÇÃO DE ROBUSTEZ DAS CORRELAÇÕES\n")
                f.write("=" * 50 + "\n\n")
                f.write("ATENÇÃO: Poucos dados para análise robusta de correlações.\n")
                f.write(f"Número de observações: {len(df_integrado)}\n")
                f.write("Recomenda-se cautela na interpretação das correlações encontradas.\n")

            return False

        # Colunas numéricas (excluindo Ano)
        colunas_numericas = df_integrado.select_dtypes(include=['number']).columns.tolist()
        if 'Ano' in colunas_numericas:
            colunas_numericas.remove('Ano')

        # Calcular matriz de correlação original
        matriz_correlacao_original = df_integrado[colunas_numericas].corr()

        # Análise de sensibilidade: remover uma observação de cada vez
        correlacoes_sem_uma_obs = []

        for i in range(len(df_integrado)):
            # Criar cópia sem a observação i
            df_sem_i = df_integrado.drop(df_integrado.index[i])

            # Calcular nova matriz de correlação
            matriz_correlacao_sem_i = df_sem_i[colunas_numericas].corr()

            # Armazenar
            correlacoes_sem_uma_obs.append(matriz_correlacao_sem_i)

        # Calcular variação nas correlações
        variacoes = {}

        for i in range(len(colunas_numericas)):
            for j in range(i + 1, len(colunas_numericas)):
                col1 = colunas_numericas[i]
                col2 = colunas_numericas[j]

                # Correlação original
                corr_original = matriz_correlacao_original.loc[col1, col2]

                # Correlações sem uma observação
                corrs_sem_uma_obs = [m.loc[col1, col2] for m in correlacoes_sem_uma_obs]

                # Calcular variação
                variacao_min = min(corrs_sem_uma_obs) - corr_original
                variacao_max = max(corrs_sem_uma_obs) - corr_original
                variacao_abs_max = max(abs(variacao_min), abs(variacao_max))

                # Armazenar
                variacoes[(col1, col2)] = {
                    "correlacao_original": corr_original,
                    "variacao_min": variacao_min,
                    "variacao_max": variacao_max,
                    "variacao_abs_max": variacao_abs_max
                }

        # Ordenar variações por magnitude
        variacoes_ordenadas = sorted(variacoes.items(), key=lambda x: x[1]["variacao_abs_max"], reverse=True)

        # Reportar resultados
        print("Variações nas correlações ao remover uma observação:")
        for (col1, col2), var in variacoes_ordenadas[:5]:  # Top 5 variações
            print(
                f"  {col1} vs {col2}: {var['correlacao_original']:.4f} (variação máxima: {var['variacao_abs_max']:.4f})")

        # Determinar robustez
        correlacoes_robustas = True
        correlacoes_problematicas = []

        for (col1, col2), var in variacoes_ordenadas:
            # Considerar não robusta se a variação for maior que 0.2 ou se mudar de sinal
            if var["variacao_abs_max"] > 0.2 or (
                    var["correlacao_original"] * (var["correlacao_original"] + var["variacao_min"]) < 0) or (
                    var["correlacao_original"] * (var["correlacao_original"] + var["variacao_max"]) < 0):
                correlacoes_robustas = False
                correlacoes_problematicas.append((col1, col2, var))

        if not correlacoes_robustas:
            print("ATENÇÃO: Algumas correlações não são robustas:")
            for col1, col2, var in correlacoes_problematicas:
                print(
                    f"  {col1} vs {col2}: {var['correlacao_original']:.4f} (variação máxima: {var['variacao_abs_max']:.4f})")
        else:
            print("Todas as correlações são robustas à remoção de observações individuais.")

        # Salvar resultados
        with open(os.path.join(diretorio_validacao, "verificacao_correlacoes.txt"), "w") as f:
            f.write("VERIFICAÇÃO DE ROBUSTEZ DAS CORRELAÇÕES\n")
            f.write("=" * 50 + "\n\n")

            f.write("Matriz de correlação original:\n")
            f.write(str(matriz_correlacao_original) + "\n\n")

            f.write("Variações nas correlações ao remover uma observação:\n")
            for (col1, col2), var in variacoes_ordenadas:
                f.write(f"{col1} vs {col2}: {var['correlacao_original']:.4f}\n")
                f.write(f"  Variação mínima: {var['variacao_min']:.4f}\n")
                f.write(f"  Variação máxima: {var['variacao_max']:.4f}\n")
                f.write(f"  Variação absoluta máxima: {var['variacao_abs_max']:.4f}\n\n")

            if not correlacoes_robustas:
                f.write("ATENÇÃO: Algumas correlações não são robustas:\n")
                for col1, col2, var in correlacoes_problematicas:
                    f.write(
                        f"  {col1} vs {col2}: {var['correlacao_original']:.4f} (variação máxima: {var['variacao_abs_max']:.4f})\n")
                f.write("\nRecomenda-se cautela na interpretação destas correlações.\n")
            else:
                f.write("Todas as correlações são robustas à remoção de observações individuais.\n")
                f.write("As correlações encontradas podem ser consideradas confiáveis.\n")

        return correlacoes_robustas

    except Exception as e:
        print(f"ERRO ao verificar robustez das correlações: {e}")

        # Salvar resultado
        with open(os.path.join(diretorio_validacao, "verificacao_correlacoes.txt"), "w") as f:
            f.write("VERIFICAÇÃO DE ROBUSTEZ DAS CORRELAÇÕES\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"ERRO ao verificar robustez das correlações: {e}\n")

        return False


def validar_analises_regressao():
    """
    Valida as análises de regressão realizadas.
    Verifica pressupostos da regressão linear e a significância estatística dos resultados.
    """
    print("\n4. Validando as análises de regressão...")

    # Tentar carregar o dataframe integrado
    caminho_df_integrado = "C:/Users/fuguz\DATASETS/dataframe_integrado.csv"

    if not os.path.exists(caminho_df_integrado):
        print(f"ERRO: Arquivo {caminho_df_integrado} não encontrado.")

        # Salvar resultado
        with open(os.path.join(diretorio_validacao, "validacao_regressao.txt"), "w") as f:
            f.write("VALIDAÇÃO DAS ANÁLISES DE REGRESSÃO\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"ERRO: Arquivo {caminho_df_integrado} não encontrado.\n")
            f.write("Não foi possível validar as análises de regressão.\n")




def main():
    """
    Função main que inicia a validação dos resultados.
    """
    validar_resultados()

# Ponto de entrada
if __name__ == "__main__":
    main()
