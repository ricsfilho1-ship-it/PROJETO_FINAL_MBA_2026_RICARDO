# FireRisk AI — Etapa 1 corrigida

## 1. Problema de negócio

Este projeto prepara uma pipeline de Machine Learning para estimar a ocorrência de pelo menos um foco de queimada em um município em determinado dia, combinando clima, geografia e cobertura/uso do solo.

> **Pergunta:** com base nas condições meteorológicas e características ambientais de um município, é possível antecipar dias com ocorrência de focos de queimadas?

A unidade analítica é **município-dia** e a variável alvo é `fire_occurrence`:

- `0`: não houve foco de queimada;
- `1`: houve pelo menos um foco de queimada.

`focos_queimada` é mantida somente para análise descritiva e **não pode ser usada como feature**, pois define o alvo e causaria *data leakage*.

## 2. Fontes

| Arquivo | Conteúdo |
|---|---|
| `BDMEP-2025.csv` | Observações meteorológicas horárias do INMET |
| `BDMEP-estacao.csv` | Cadastro e localização das estações |
| `MapBiomas.csv` | Área municipal por classe de cobertura/uso do solo |
| `queimadas_2024_2025.zip` | Focos de queimadas |

A janela comum entre clima e queimadas usada nesta etapa é **01/01/2025 a 12/02/2025**.

## 3. Correções aplicadas após a primeira avaliação

### 3.1 Cobertura climática — linhas sem nenhuma informação de clima

A auditoria mostrou **4.451 linhas município-dia (19,1% da base inicial)** sem nenhuma das nove variáveis meteorológicas. Elas estavam distribuídas em 135 municípios; **85 municípios não tinham nenhum dado de clima em nenhum dos 43 dias**.

Essas linhas foram removidas da base de Machine Learning, pois nelas o modelo teria apenas geografia e MapBiomas, o que aumentaria o risco de memorizar o município em vez de aprender a relação entre clima e queimadas.

Após a correção:

- linhas finais: **18,855**;
- municípios finais: **457**;
- municípios totalmente sem clima removidos: **85**;
- linhas sem qualquer clima removidas: **4451**;
- taxa da classe positiva: **15.30%**.

A auditoria completa está em `reports/qualidade/cobertura_climatica_municipios.csv` e a lista dos 85 municípios em `reports/qualidade/municipios_excluidos_sem_clima.csv`.

### 3.2 Controle de qualidade físico antes de outliers estatísticos

O IQR continua sendo usado para identificar extremos estatísticos, mas **não é mais usado como justificativa para aceitar automaticamente valores extremos**.

Os dados automáticos do INMET são disponibilizados como dados brutos; o próprio INMET alerta que falhas de sensor/comunicação podem ocorrer. Por isso foram adicionadas regras físicas básicas antes da análise de outliers.

Para radiação, a soma diária é validada contra a **radiação extraterrestre diária (FAO-56)** calculada para latitude e data. Valores `<= 0` ou acima de 105% desse limite físico são convertidos para `NaN`.

Foram identificados **45 registros estação-dia de radiação inválida**, incluindo valores diários que chegavam a 87,9 MJ/m² e 22 registros com zero. O detalhe está em `reports/qualidade/radiacao_invalida.csv`.

Fontes metodológicas:

- INMET — dados automáticos são brutos e podem conter falhas: https://portal.inmet.gov.br/noticias/saiba-como-acessar-os-dados-meteorol%C3%B3gicos-dispon%C3%ADveis-no-site-do-inmet
- INMET — Radiação Global em kJ/m²: https://portal.inmet.gov.br/glossario/nublado
- FAO-56 — cálculo da radiação extraterrestre diária: https://www.fao.org/4/x0490e/x0490e07.htm

### 3.3 Legenda MapBiomas

As colunas deixaram de ser nomes genéricos como `mb_area_class_3`. Agora usam nomes interpretáveis, por exemplo:

- `mb_area_formacao_florestal` — classe 3;
- `mb_area_pastagem` — classe 15;
- `mb_area_soja` — classe 39;
- `mb_area_area_urbanizada` — classe 24;
- `mb_area_rio_lago_oceano` — classe 33.

O arquivo fornecido cobre 1985–2021, período consistente com a **Coleção 7 do MapBiomas**. A correspondência foi feita com a legenda oficial da coleção: https://brasil.mapbiomas.org/wp-content/uploads/sites/4/2023/08/EN__Codigos_da_legenda_Colecao_7.pdf

O ID `0` aparece no CSV fornecido, mas não consta na legenda oficial consultada; por transparência ele foi mantido como `mb_area_id0_sem_legenda`, sem inventar um significado.

O dicionário completo está em `data/processed/dicionario_dados.csv`.

## 4. Base final

A base corrigida possui:

- **18,855 observações município-dia**;
- **457 municípios**;
- **2,885 observações positivas**;
- taxa positiva de **15.30%**;
- **13,550 focos** nas linhas com clima utilizável.

Ainda existem valores ausentes em variáveis meteorológicas individuais. Eles são preservados nesta etapa e deverão ser imputados **somente dentro do conjunto de treino / pipeline** na Etapa 2 para evitar leakage.

## 5. EDA incluído

O notebook `notebooks/01_EDA_queimadas.ipynb` inclui:

1. estrutura e distribuição do alvo;
2. auditoria de cobertura climática por linha e município;
3. missing values após a correção;
4. estatísticas descritivas;
5. controle de qualidade físico e IQR;
6. histogramas das features meteorológicas;
7. correlação com o alvo;
8. análise das classes do MapBiomas;
9. análise por região e UF;
10. teste de hipótese Mann–Whitney para umidade;
11. metas dos KPIs para a modelagem.

## 6. Hipótese estatística

**H0:** a distribuição de umidade média nos dias com queimadas não é menor do que nos dias sem queimadas.  
**H1:** a umidade média nos dias com queimadas é menor.

Na base corrigida, a umidade média é aproximadamente **70,08%** em dias com queimada e **76,96%** em dias sem queimada. O teste Mann–Whitney U unilateral produz `p < 0,001`, portanto H0 é rejeitada no nível de 5%.

Essa associação não demonstra causalidade, mas apoia a relevância da umidade para a modelagem.

## 7. KPIs e metas da Etapa 2

As metas foram definidas **antes da nova modelagem**, para tornar o critério de sucesso explícito:

| KPI | Meta | Papel |
|---|---:|---|
| Recall da classe positiva | **≥ 0,70** | KPI principal; reduzir eventos não sinalizados |
| F1-score | **≥ 0,40** | Equilibrar recall e precisão |
| Precision | **≥ 0,30** | Limitar excesso de falsos alertas |
| ROC-AUC | **≥ 0,75** | Capacidade de separação das classes |
| False Negative Rate | **≤ 0,30** | Meta operacional complementar |
| Acurácia | Monitorada | Não será critério isolado devido ao desbalanceamento |

Essas metas são iniciais e deverão ser confrontadas com o custo real de falso positivo e falso negativo na Etapa 2.

## 8. Estrutura do repositório

```text
projeto_final_queimadas_etapa1_corrigido/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── 01_EDA_queimadas.ipynb
├── src/
│   └── consolidar_base.py
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       ├── dataset_ml.csv
│       └── dicionario_dados.csv
└── reports/
    ├── figures/
    └── qualidade/
        ├── cobertura_climatica_municipios.csv
        ├── municipios_excluidos_sem_clima.csv
        ├── radiacao_invalida.csv
        └── regras_qualidade.csv
```

**O pacote entregue não contém uma pasta `.git`**, portanto não carrega arquivos internos do Git. Os arquivos brutos também não são versionados por serem grandes.

## 9. Como executar

### Executar somente o EDA

A base processada já está incluída. A partir da raiz do repositório:

```bash
jupyter notebook notebooks/01_EDA_queimadas.ipynb
```

O notebook localiza a raiz do projeto automaticamente, então funciona mesmo quando iniciado pela raiz do repositório.

### Reconstruir a base a partir dos arquivos brutos

Coloque os quatro arquivos em `data/raw/` e execute:

```bash
python src/consolidar_base.py \
  --raw-dir data/raw \
  --output data/processed/dataset_ml.csv \
  --quality-dir reports/qualidade
```

## 10. Observações metodológicas

- O MapBiomas é de 2021 e funciona como característica estrutural, não como condição dinâmica de 2025.
- A janela temporal é curta (43 dias), o que limita generalizações sazonais.
- Alguns municípios possuem cobertura climática parcial; dias sem **nenhum** dado climático foram removidos e a cobertura por município é auditada separadamente.
- Geografia e MapBiomas são constantes por município. Na Etapa 2 é recomendável avaliar também uma estratégia de validação por grupo/município como teste de robustez, além da divisão temporal.
- Correlação e teste de hipótese indicam associação, não causalidade.
