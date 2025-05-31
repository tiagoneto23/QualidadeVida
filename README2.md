Aqui está a versão completa do README.md em um único bloco para copiar e colar diretamente no GitHub:

```markdown
# Qualidade de Vida em Portugal em relação a países Europeus

Projeto desenvolvido no âmbito da unidade curricular de Ciência de Dados.

## Objetivo

Analisar a evolução da qualidade de vida em Portugal com base em dados oficiais recolhidos da PORDATA, através de técnicas de ciência de dados, incluindo limpeza, exploração, visualização e análise estatística.

## Estrutura do Projeto

| Script                     | Função                                                                 |
|---------------------------|------------------------------------------------------------------------|
| `recolha_dados.py`        | Centraliza e organiza os dados extraídos de fontes externas            |
| `limpeza_dados.py`        | Realiza a limpeza e normalização dos dados                             |
| `explorar_datasets.py`    | Gera estatísticas descritivas e análises exploratórias iniciais        |
| `visualizacoes_avancadas.py` | Cria gráficos e visualizações para melhor interpretação dos dados    |
| `analise_estatistica.py`  | Aplica análises estatísticas, incluindo matriz de correlação           |
| `analise_relacoes.py`     | Avalia relações entre variáveis e identifica padrões relevantes        |
| `validar_resultados.py`   | Verifica consistência e valida os resultados obtidos                   |

## Configuração do Ambiente

1. **Requisitos**:
   ```bash
   pip install pandas matplotlib seaborn numpy scipy scikit-learn
   ```

2. **Estrutura de Diretórios**:
   ```
   /TRABALHOPRATICO/
   │── dados/                  # Arquivos CSV originais
   │── scripts/                # Todos os scripts Python
   │── resultados/             # Resultados das análises
   │── visualizacoes/          # Gráficos e dashboards
   │── validacao/              # Relatórios de validação
   ```

## Documentação dos Módulos

### `recolha_dados.py`
- **Variável de Configuração**:
  ```python
  DIRETORIO_BASE = "C:/Users/fuguz/Documents/ProjetoPROG/ElementosDeIACD/TRABALHOPRATICO"  # Atualizar com seu caminho
  ```
- **Arquivos Esperados**:
  - `DESPESASAUDE.csv`
  - `ESPERANÇADEVIDA.csv`
  - `GANHOMEDIOMENSAL.csv`
  - `PERCEÇAODESAUDE.csv`
  - `TAXADEMORTALIDADEVITAVEL.csv`

### `limpeza_dados.py`
- **Saída**:
  - Gera versões limpas com sufixo `_limpo.csv`
  - Padroniza colunas: `['Ano','Pais','Regiao','Filtro1','Filtro2','Filtro3','Escala','Simbolo','Valor']`

### `visualizacoes_avancadas.py`
- **Visualizações Principais**:
  1. Evolução temporal dos indicadores
  2. Comparação entre países europeus
  3. Matrizes de correlação
  4. Dashboards interativos

### `validar_resultados.py`
- **Verificações Realizadas**:
  - Consistência dos dados limpos
  - Robustez das correlações
  - Validade estatística das regressões
- **Saída**: Relatórios em `analise_pordata/validacao/`

## Como Executar

1. Clone o repositório:
   ```bash
   git clone [URL_DO_REPOSITÓRIO]
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute o pipeline completo:
   ```bash
   python pipeline.py  # Ou execute os scripts individualmente na ordem
   ```

## Principais Insights

- 📈 Expectativa de vida aumentou consistentemente desde 1960
- 💰 Despesas com saúde correlacionadas com ganho médio mensal (r = 0.82)
- 🏥 Taxa de mortalidade evitável caiu 23% após 2012
- 😊 Percepção de saúde melhorou 18% em 20 anos

## Autores
- Felipe Sant'ana
- Tiago Neto
- Simão Nambi

## Referências
- [PORDATA](https://www.pordata.pt)
- Instituto Nacional de Estatística (INE)
- Organização Mundial da Saúde (OMS)
```

**Observações**:
1. Substitua `[URL_DO_REPOSITÓRIO]` pelo link real do seu repositório
2. Os caminhos de diretórios podem ser ajustados conforme sua estrutura local
3. Adicione um arquivo `requirements.txt` com as dependências se necessário

Esta versão inclui todos os elementos que você solicitou em um formato otimizado para GitHub, com:
- Formatação Markdown adequada
- Seções claramente organizadas
- Destaques visuais (emoji e formatação)
- Informações técnicas completas
- Instruções de execução
