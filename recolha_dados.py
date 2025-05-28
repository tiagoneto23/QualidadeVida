import os
import pandas as pd

DIRETORIO_BASE = "C:/Users/Nambi/Documents/"

def recolha_dados():
    arquivos_desejados = [
        'DESPESASAUDE.csv',
        'ESPERANÇADEVIDA.csv',
        'GANHOMEDIOMENSAL.csv',
        'PERCEÇAODESAUDE.csv',
        'TAXADEMORTALIDADEVITAVEL.csv'
    ]

    datasets = {}

    for raiz, dirs, arquivos in os.walk(DIRETORIO_BASE):
        for nome_ficheiro in arquivos:
            if nome_ficheiro in arquivos_desejados:
                caminho = os.path.join(raiz, nome_ficheiro)

                try:
                    df = pd.read_csv(caminho, encoding='utf-16le')
                    datasets[caminho] = df
                    print(f"Arquivo carregado de {caminho}")
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(caminho, encoding='latin1')
                        datasets[caminho] = df
                        print(f"Arquivo carregado de {caminho} com codificação 'latin1'")
                    except Exception as e:
                        print(f"Erro ao ler {caminho} com codificação 'latin1': {e}")
                except Exception as e:
                    print(f"Erro ao ler {caminho}: {e}")

    return datasets
