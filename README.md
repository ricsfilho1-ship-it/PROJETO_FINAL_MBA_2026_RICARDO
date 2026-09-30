# FireRisk AI — Predição de ocorrência de queimadas no Brasil

## 1. Objetivo do projeto

Este projeto propõe uma pipeline de Machine Learning para estimar o risco de ocorrência de pelo menos um foco de queimada em um município em determinado dia, utilizando informações meteorológicas, geográficas e de cobertura do solo.

A pergunta de negócio é:

> **Com base nas condições meteorológicas e nas características ambientais de um município, é possível antecipar a ocorrência de focos de queimadas?**

A ideia é transformar dados públicos em uma ferramenta de apoio à priorização de monitoramento e resposta preventiva.

## 2. Variável alvo

A variável alvo é `fire_occurrence`:

- `0`: não houve foco de queimada no município naquele dia;
- `1`: houve pelo menos um foco de queimada no município naquele dia.

A coluna `focos_queimada` é mantida somente para análise descritiva. **Ela não deve ser usada como variável preditora**, pois define diretamente a variável alvo e causaria vazamento de informação (*data leakage*).

## 3. Fontes de dados fornecidas

Os arquivos usados nesta etapa foram:

| Arquivo | Conteúdo | Dimensão observada |
|---|---|---:|
| `BDMEP-2025.csv` | Observações meteorológicas | 4.891.176 linhas |
| `BDMEP-estacao.csv` | Cadastro e localização das estações | 612 estações |
| `MapBiomas.csv` | Área por classe de cobertura do solo | 2.286.656 linhas, 1985–2021 |
| `queimadas_2024_2025.zip` | Registros de focos de queimadas | 8.226.944 linhas |

A base de queimadas de 2025 termina em **12/02/2025**, enquanto a meteorologia continua até 30/06/2025. Para evitar comparar períodos incompatíveis, a base de modelagem foi limitada à janela comum de **01/01/2025 a 12/02/2025**.

## 4. Unidade de análise e base final

A unidade analítica é **município-dia**.

Depois da consolidação, a base possui:

- **23.306 observações município-dia**;
- **542 municípios** com dados meteorológicos utilizáveis;
- **3.606 observações positivas** (`fire_occurrence = 1`);
- taxa positiva de aproximadamente **15,47%**;
- **16.312 focos de queimadas** registrados dentro dos municípios cobertos pela base meteorológica no período analisado.

Esse nível de desbalanceamento é relevante, mas não extremo. Por isso, na Etapa 2 será importante avaliar F1-score, Recall, Precision e ROC-AUC, além da acurácia.

## 5. Principais variáveis

### Meteorologia

- `precipitacao_diaria`
- `pressao_media`
- `radiacao_global_diaria`
- `temperatura_media`
- `temperatura_maxima`
- `temperatura_minima`
- `umidade_media`
- `rajada_maxima`
- `vento_medio`

### Geografia

- `latitude`
- `longitude`
- `altitude`
- `sigla_uf`

### Calendário

- `mes`
- `dia_do_ano`
- `dia_semana`

### Cobertura do solo

As colunas `mb_area_class_*` representam as áreas das classes de MapBiomas em **2021**, o ano mais recente contido no arquivo fornecido. A taxonomia textual das classes não acompanha os arquivos recebidos, portanto o projeto preserva os IDs originais sem inventar rótulos.

## 6. Preparação e limpeza dos dados

### 6.1 Duplicidades meteorológicas

Durante a exploração foi identificado que o arquivo meteorológico possui duplicidade quase integral por `id_estacao + data + hora`. Na janela analisada havia 1.156.872 linhas, mas apenas 579.984 combinações únicas de estação, data e hora.

O tratamento aplicado foi:

1. remoção de duplicatas exatas;
2. quando ainda havia mais de um registro no mesmo horário, consolidação dos valores numéricos pela média;
3. após a limpeza, cada estação-dia passou a ter exatamente 24 posições horárias.

### 6.2 Agregação meteorológica

Os dados horários foram convertidos para o nível diário por estação. Depois, quando um município possuía mais de uma estação, os valores foram agregados para município-dia.

### 6.3 Queimadas

Os focos foram agregados por município-dia. A quantidade foi armazenada em `focos_queimada`, e a variável alvo binária foi criada a partir dela.

### 6.4 MapBiomas

Foi utilizado o ano de 2021, pois é o último ano disponível no arquivo fornecido. Cada classe foi transformada em uma coluna de área por município.

### 6.5 Dados ausentes

Os dados meteorológicos apresentam ausência relevante em algumas variáveis. Na base consolidada, os maiores percentuais são aproximadamente:

- precipitação diária: **31,94%**;
- rajada máxima: **25,74%**;
- vento médio: **25,62%**;
- umidade média: **23,63%**;
- radiação global diária: **21,72%**.

Os valores ausentes foram **mantidos na base consolidada**. A decisão é proposital: na Etapa 2, a imputação deve ser ajustada apenas no conjunto de treinamento ou dentro de uma pipeline/cross-validation para evitar vazamento de informação.

## 7. Exploração inicial

A taxa de dias com ocorrência de queimada na amostra é de aproximadamente **15,5%**.

Entre as variáveis meteorológicas, as correlações de Spearman mais fortes com a variável alvo foram:

- temperatura máxima: aproximadamente **+0,268**;
- temperatura média: aproximadamente **+0,249**;
- umidade média: aproximadamente **-0,230**;
- precipitação diária: aproximadamente **-0,181**;
- radiação global diária: aproximadamente **+0,167**.

Esses valores não provam causalidade, mas ajudam a definir hipóteses e orientar a modelagem.

## 8. Teste de hipótese estatístico

### Premissa de negócio

Dias com ocorrência de queimadas tendem a apresentar menor umidade relativa do ar.

### Hipóteses

- **H0:** a distribuição de umidade média dos dias com queimadas não é menor que a dos dias sem queimadas.
- **H1:** a umidade média nos dias com queimadas é menor que nos dias sem queimadas.

Como a distribuição não precisa ser normal, foi aplicado o teste **Mann–Whitney U unilateral**.

Resultado observado:

- média de umidade em dias com queimada: **70,08%**;
- média em dias sem queimada: **76,96%**;
- mediana em dias com queimada: **71,67%**;
- mediana em dias sem queimada: **78,42%**;
- p-valor: **< 0,001**.

Com nível de significância de 5%, rejeitamos H0. Na amostra analisada, dias com ocorrência de queimadas apresentam umidade significativamente menor.

## 9. KPIs de sucesso para a Etapa 2

Como o problema é de classificação, serão acompanhadas:

- Recall;
- Precision;
- F1-score;
- ROC-AUC;
- Acurácia;
- Matriz de confusão.

O **Recall da classe positiva** será tratado como KPI prioritário, pois um falso negativo significa deixar de sinalizar um município-dia em que ocorreu uma queimada.

A escolha final do modelo deverá equilibrar Recall e Precision para evitar uma quantidade excessiva de falsos alertas.

## 10. Limitações identificadas

1. A cobertura temporal comum disponível nesta entrega é curta: 43 dias de 2025.
2. Nem todos os municípios brasileiros possuem estações meteorológicas na base, portanto o escopo final ficou limitado a 542 municípios com dados utilizáveis.
3. Algumas variáveis meteorológicas possuem percentual relevante de valores ausentes.
4. O MapBiomas fornecido termina em 2021 e é usado como uma característica estrutural do município, não como condição dinâmica de 2025.
5. O arquivo de MapBiomas fornecido não contém um dicionário textual das classes; por isso os IDs foram preservados.
6. Correlações e testes de hipótese desta etapa são associações estatísticas e não estabelecem causalidade.

## 11. Estrutura do repositório

```text
projeto_final_queimadas_etapa1/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── 01_EDA_queimadas.ipynb
├── src/
│   └── consolidar_base.py
├── data/
│   └── processed/
│       ├── dataset_ml.csv
│       └── dicionario_dados.csv
└── reports/
    └── figures/
        ├── distribuicao_alvo.png
        ├── correlacao_spearman.png
        ├── umidade_por_alvo.png
        └── temperatura_maxima_por_alvo.png
```

Os quatro arquivos brutos são grandes e não devem ser enviados diretamente para um repositório GitHub comum. O `.gitignore` já está preparado para ignorar uma pasta `data/raw/`.

## 12. Como reproduzir

Instale as dependências:

```bash
pip install -r requirements.txt
```

Ajuste `RAW_DIR` no final de `src/consolidar_base.py` para o local dos quatro arquivos brutos e execute:

```bash
python src/consolidar_base.py
```

Depois abra:

```text
notebooks/01_EDA_queimadas.ipynb
```

## 13. Próxima etapa

Na Etapa 2 serão comparados pelo menos três modelos de classificação, por exemplo:

1. Logistic Regression;
2. Random Forest;
3. LightGBM ou XGBoost.

A modelagem deverá incluir imputação sem leakage, tratamento de variáveis categóricas, avaliação das métricas, cálculo de impacto financeiro e plano de teste A/B.
