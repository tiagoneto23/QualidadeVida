# Qualidade de Vida em Portugal

Projeto desenvolvido no âmbito da unidade curricular de Ciência de Dados.

## Objetivo

Analisar a evolução da qualidade de vida em Portugal com base em dados oficiais recolhidos da PORDATA, através de técnicas de ciência de dados, incluindo limpeza, exploração, visualização e análise estatística.

---

## Estrutura do Projeto

O projeto está dividido em módulos, cada um responsável por uma etapa do processo de análise:

| Script                     | Função                                                                 |
|---------------------------|------------------------------------------------------------------------|
| `recolha_dados.py`        | Centraliza e organiza os dados extraídos de fontes externas.           |
| `limpeza_dados.py`        | Realiza a limpeza e normalização dos dados.                            |
| `explorar_datasets.py`    | Gera estatísticas descritivas e análises exploratórias iniciais.       |
| `visualizacoes_avancadas.py` | Cria gráficos e visualizações para melhor interpretação dos dados.   |
| `analise_estatistica.py`  | Aplica análises estatísticas, incluindo matriz de correlação.          |
| `analise_relacoes.py`     | Avalia relações entre variáveis e identifica padrões relevantes.       |
| `validar_resultados.py`   | Verifica consistência e valida os resultados obtidos.                  |

---

## Conjunto de Dados Utilizados

Todos os dados foram recolhidos da [PORDATA](https://www.pordata.pt/) e estão em formato `.csv`:

- `TAXADEMORTALIDADE.csv`
- `PERCEÇAODESAUDE.csv`
- `GANHOMEDIOMENSAL.csv`
- `ESPERANÇADEVIDA.csv`
- `DESPESASAUDE.csv`

---

## Principais Insights

- **A expectativa de vida aumentou** consistentemente desde 1960.
- **Despesas com saúde cresceram** entre 2000 e 2020, com correlação forte com o ganho médio mensal.
- **A percepção de saúde melhora** à medida que o rendimento médio aumenta.
- **A taxa de mortalidade evitável caiu** após 2012, indicando avanços em prevenção.

---

## Técnicas Utilizadas

- Limpeza de dados com `pandas`
- Análise descritiva e estatística
- Visualizações com `matplotlib` e `seaborn`
- Correlações entre variáveis
- Avaliação de relações multivariadas

---

##  Relatórios e Entregas

- ✅ **[Relatório Final](relatorio_final.ipynb)** com explicações organizadas por fase (para visualização no Google Colab ou Jupyter)
- ✅ Repositório Git com todo o código usado no projeto
- ✅ Apresentação narrativa das conclusões extraídas a partir dos dados analisados

---

## Requisitos

- Python 3.x
- pandas
- matplotlib
- seaborn
- numpy

Para instalar os pacotes necessários:

```bash
pip install pandas matplotlib seaborn numpy
```

---

## Como Executar o Projeto

1. Clone o repositório:

```bash
git clone https://github.com/tiagoneto23/QualidadeVida.git
```

2. Execute os scripts ou o relatório principal em [Google Colab](https://colab.research.google.com/) ou localmente com Jupyter:

- Abrir `relatorio_final.ipynb` para ver o relatório completo com análises e conclusões.

3. Para executar scripts individuais, use:

```bash
python nome_do_script.py
```

---

## Autores
- Felipe Sant'ana
- Tiago Neto
- Simão Nambi 

---

##  Referências

- PORDATA: [https://www.pordata.pt](https://www.pordata.pt)
- Instituto Nacional de Estatística (INE)
- Organização Mundial da Saúde (OMS)

---
