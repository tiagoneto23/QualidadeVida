import pandas as pd

def recolha(): #Recolha manual dos dados
    try:
        df = pd.read_csv('610.csv', encoding='utf-8')
        print("Sucesso")
        return df
    except UnicodeDecodeError: #Todos os datasets da PORDATA devem estar em UTF-16, mas por precaução..
        print("O ficheiro não está em UTF-8, a tentar com encoding diferente...")
        try:
            df = pd.read_csv('610.csv', encoding='utf-16le')
            return df
        except FileNotFoundError:
            print("Ficheiro não encontrado, certifique-se que está na diretoria certa")
            return None
        except Exception as e:
            print(e)
            return None
    except FileNotFoundError:
        print("Ficheiro não encontrado, certifique-se que está na diretoria certa")
        return None
    except Exception as e:
        print(e)
        return None

df = recolha()
print(df)

