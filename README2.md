# Qualidade de Vida em Portugal em relação a países Europeus

Projeto desenvolvido no âmbito da unidade curricular de Ciência de Dados.

## Objetivo

Analisar a evolução da qualidade de vida em Portugal com base em dados oficiais recolhidos da PORDATA, através de técnicas de ciência de dados, incluindo limpeza, exploração, visualização e análise estatística.

## Estrutura do Projeto

| Script                     | Função                                                                 |
|----------------------------|-----------------------------------------------------------------------|
| `recolha_dados.py`         | Centraliza e organiza os dados extraídos de fontes externas           |
| `limpeza_dados.py`         | Realiza a limpeza e normalização dos dados                            |
| `explorar_datasets.py`     | Gera estatísticas descritivas e análises exploratórias iniciais       |
| `visualizacoes_avancadas.py` | Cria gráficos e visualizações para melhor interpretação dos dados     |
| `analise_estatistica.py`   | Aplica análises estatísticas, incluindo matriz de correlação          |
| `analise_relacoes.py`      | Avalia relações entre variáveis e identifica padrões relevantes       |
| `validar_resultados.py`    | Verifica consistência e valida os resultados obtidos                  |

## Configuração do Ambiente

1. **Pré-requisitos**:
   - Python 3.8+
   - Bibliotecas listadas em `requirements.txt`

2. **Instalação**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Estrutura de Diretórios**:
   ```
   /
   ├── dados/                  # Arquivos CSV originais da PORDATA
   ├── scripts/                # Todos os scripts Python do projeto
   ├── resultados/             # Resultados das análises (CSVs, relatórios)
   └── visualizacoes/          # Gráficos e dashboards gerados
   ```

## Como Executar

1. Coloque seus arquivos CSV na pasta `dados/`
2. Execute o pipeline completo:
   ```bash
   python scripts/pipeline.py
   ```
   Ou execute módulos individualmente:
   ```bash
   python scripts/recolha_dados.py
   python scripts/limpeza_dados.py
   # ... e assim por diante
   ```

## Documentação dos Módulos

### `recolha_dados.py`
- **Configuração**:
  ```python
  DIRETORIO_BASE = "caminho/para/pasta/dados"  # Atualize com seu caminho
  ```
- **Arquivos Necessários**:
  - `DESPESASAUDE.csv`
  - `ESPERANÇADEVIDA.csv`
  - `GANHOMEDIOMENSAL.csv`
  - `PERCEÇAODESAUDE.csv`
  - `TAXADEMORTALIDADEVITAVEL.csv`

### `limpeza_dados.py`
- **Saída**:
  - Arquivos com sufixo `_limpo.csv` na pasta `resultados/`
  - Padronização de colunas:
    ```python
    ['Ano', 'Pais', 'Regiao', 'Filtro1', 'Filtro2', 'Filtro3', 'Escala', 'Simbolo', 'Valor']
    ```

### `visualizacoes_avancadas.py`
- **Gráficos Gerados**:
  - Evolução temporal (linha)
  - Comparação entre países (barras)
  - Matriz de correlação (heatmap)
  - Dashboards consolidados

## Principais Resultados

- 📈 **Expectativa de Vida**: Aumento de 12 anos desde 1960
- 💰 **Rendimento**: Correlação forte (0.82) com despesas em saúde
- 🏥 **Mortalidade**: Redução de 23% em causas evitáveis (2012-2020)
- 😊 **Percepção de Saúde**: 78% da população avalia como "boa" ou "muito boa"

## Autores
- [Felipe Sant'ana](link_github_se_existir)
- [Tiago Neto](link_github_se_existir)
- [Simão Nambi](link_github_se_existir)

## Referências
- [PORDATA - Base de Dados Portugal Contemporâneo](https://www.pordata.pt)

