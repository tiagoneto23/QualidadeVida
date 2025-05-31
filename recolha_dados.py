import os
import pandas as pd

# ================================
# ATENÇÃO: ALTERAR ESTE CAMINHO!
# Diretório onde estão guardados os ficheiros .csv
DIRETORIO_BASE = "C:/Users/fuguz/Documents/ProjetoPROG/ElementosDeIACD/TRABALHOPRATICO"
# ================================

def recolha_dados():
    arquivos_desejados = [
        'DESPESASAUDE.csv',
        'ESPERANÇADEVIDA.csv',
        'GANHOMEDIOMENSAL.csv',
        'PERCEÇAODESAUDE.csv',
        'TAXADEMORTALIDADEVITAVEL.csv'
    ]

    datasets = {}

    # Agora usamos a variável global DIRETORIO_BASE
    for raiz, dirs, arquivos in os.walk(DIRETORIO_BASE):  # Usamos DIRETORIO_BASE diretamente
        for nome_ficheiro in arquivos:
            if nome_ficheiro in arquivos_desejados:
                caminho = os.path.join(raiz, nome_ficheiro)

                try:
                    # Tentativa de leitura com codificação 'utf-16le'
                    df = pd.read_csv(caminho, encoding='utf-16le')
                    datasets[caminho] = df
                    print(f"Arquivo carregado de {caminho}")
                except UnicodeDecodeError:
                    try:
                        # Tentativa com codificação 'latin1'
                        df = pd.read_csv(caminho, encoding='latin1')
                        datasets[caminho] = df
                        print(f"Arquivo carregado de {caminho} com codificação 'latin1'")
                    except Exception as e:
                        print(f"Erro ao ler {caminho} com codificação 'latin1': {e}")
                except Exception as e:
                    print(f"Erro ao ler {caminho}: {e}")

    return datasets
