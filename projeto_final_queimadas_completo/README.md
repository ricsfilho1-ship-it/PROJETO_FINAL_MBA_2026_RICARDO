# FireRisk AI — Projeto Final de Machine Learning

## Visão geral

Este projeto desenvolve uma pipeline de Machine Learning de ponta a ponta para estimar a ocorrência de pelo menos um foco de queimada em determinado **município-dia**, combinando dados meteorológicos, localização geográfica e cobertura/uso do solo.

A variável alvo é `fire_occurrence`:

- `0`: não houve foco de queimada no município naquele dia;
- `1`: houve pelo menos um foco de queimada.

O objetivo de negócio é apoiar a priorização de alertas e recursos, dando maior importância à redução de **falsos negativos**.

---

## Fontes de dados

- **BDMEP 2025** — observações meteorológicas;
- **Cadastro de estações BDMEP** — município, latitude, longitude e altitude;
- **MapBiomas** — características estruturais de uso/cobertura do solo;
- **Queimadas 2024–2025** — focos de queimadas usados para construir a variável alvo.

A janela comum utilizada no projeto é **01/01/2025 a 12/02/2025**.

---

# Etapa 1 — preparação, qualidade e EDA

A primeira versão foi revisada após feedback da avaliação. As principais correções foram:

- identificação de **85 municípios sem nenhum dado meteorológico**;
- remoção das **4.451 linhas** sem qualquer informação climática;
- base final de **18.855 observações** e **457 municípios**;
- validação física de radiação antes da análise por IQR;
- tratamento dos registros de radiação inválidos como ausentes, em vez de classificá-los automaticamente como extremos reais;
- nomes interpretáveis para as classes MapBiomas;
- histogramas, análise geográfica por região/UF e análise das classes MapBiomas;
- teste de hipótese sobre umidade e ocorrência de queimadas;
- definição prévia das metas de KPI.

### Metas definidas antes da modelagem

| KPI | Meta |
|---|---:|
| Recall | ≥ 0,70 |
| F1-score | ≥ 0,40 |
| Precision | ≥ 0,30 |
| ROC-AUC | ≥ 0,75 |
| False Negative Rate | ≤ 0,30 |

Notebook: `notebooks/01_EDA_queimadas.ipynb`

---

# Etapa 2 — modelagem e inteligência de negócio

A separação foi temporal:

- treino: **2025-01-01 a 2025-01-28**;
- validação: **2025-01-29 a 2025-02-04**;
- teste: **2025-02-05 a 2025-02-12**.

Foram comparados:

1. Regressão Logística;
2. Random Forest;
3. LightGBM.

### Resultados com threshold padrão 0,50

| Modelo | Acurácia | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Regressão Logística | 0,662 | 0,288 | 0,796 | 0,422 | 0,790 |
| Random Forest | 0,845 | 0,504 | 0,343 | 0,408 | 0,796 |
| LightGBM | 0,785 | 0,385 | 0,642 | **0,482** | **0,809** |

O LightGBM foi selecionado pelo melhor equilíbrio geral. O threshold operacional foi ajustado somente com a validação para **0,42**, atingindo todas as metas de KPI no teste.

Notebook: `notebooks/02_modelagem_inteligencia_negocio.ipynb`

A Etapa 2 também contém:

- cálculo parametrizável de impacto financeiro;
- três cenários ilustrativos de custo;
- plano de teste A/B;
- teste de robustez com municípios completamente não vistos no treinamento.

---

# Etapa 3 — otimização e IA responsável

## Optuna

O LightGBM foi otimizado com **Optuna** usando três janelas de validação temporal construídas somente dentro do período de treinamento.

A execução entregue utilizou **6 trials** para manter o projeto reproduzível em ambiente acadêmico. O número pode ser aumentado usando a variável `OPTUNA_TRIALS`.

Melhor ROC-AUC médio na validação temporal interna: **0.814**.

O threshold operacional foi novamente selecionado no conjunto de validação e ficou em **0.34**.

### Resultado final da Etapa 3 no teste

| KPI | Resultado | Meta | Status |
|---|---:|---:|---|
| Recall | 0.716 | ≥ 0,70 | Atingida |
| Precision | 0.360 | ≥ 0,30 | Atingida |
| F1-score | 0.479 | ≥ 0,40 | Atingida |
| ROC-AUC | 0.818 | ≥ 0,75 | Atingida |
| FNR | 0.284 | ≤ 0,30 | Atingida |

Comparado ao LightGBM da Etapa 2, a otimização aumentou o F1 de **0.466** para **0.479** e o ROC-AUC de **0.809** para **0.818**.

## SHAP

As três variáveis com maior importância global agregada foram:

1. **mb_log_total_area** — 20.3% da importância SHAP;
2. **temperatura_maxima** — 14.4%;
3. **radiacao_global_diaria** — 9.8%.

SHAP é usado como método de explicabilidade do modelo e **não deve ser interpretado como causalidade**.

## Análise de viés

A base não contém gênero, raça ou atributos demográficos. Portanto, a análise possível neste projeto é geográfica.

- gap de recall entre regiões: **0.257**;
- maior recall regional: **Centro-Oeste**;
- menor recall regional: **Sudeste**;
- UFs com amostra suficiente para comparação: **16**.

Essas diferenças devem ser monitoradas, mas não provam viés injusto por si só, pois podem refletir frequência de eventos, composição climática e tamanho de amostra.

Notebook: `notebooks/03_otimizacao_ia_responsavel.ipynb`

---

# Estrutura do repositório

```text
projeto_final_queimadas_completo/
├── README.md
├── ENTREGA_FINAL.md
├── CORRECOES_AVALIACAO.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_EDA_queimadas.ipynb
│   ├── 02_modelagem_inteligencia_negocio.ipynb
│   └── 03_otimizacao_ia_responsavel.ipynb
├── src/
│   ├── consolidar_base.py
│   ├── modelagem_etapa2.py
│   └── otimizacao_explicabilidade_etapa3.py
├── data/
│   ├── raw/.gitkeep
│   └── processed/
│       ├── dataset_ml.csv
│       └── dicionario_dados.csv
├── models/
│   ├── lightgbm_stage2.joblib
│   ├── lightgbm_stage2_metadata.json
│   ├── lightgbm_stage3_evaluation.joblib
│   ├── lightgbm_stage3_final.joblib
│   └── lightgbm_stage3_metadata.json
└── reports/
    ├── figures/
    ├── qualidade/
    ├── etapa2/
    └── etapa3/
```

---

# Como executar

## 1. Ambiente

```bash
pip install -r requirements.txt
```

## 2. Notebooks

```bash
jupyter notebook notebooks/01_EDA_queimadas.ipynb
jupyter notebook notebooks/02_modelagem_inteligencia_negocio.ipynb
jupyter notebook notebooks/03_otimizacao_ia_responsavel.ipynb
```

## 3. Reexecutar a modelagem

```bash
python src/modelagem_etapa2.py
python src/otimizacao_explicabilidade_etapa3.py
```

Para aumentar a busca do Optuna:

```bash
OPTUNA_TRIALS=20 python src/otimizacao_explicabilidade_etapa3.py
```

---

# Principais limitações

- janela temporal curta de 43 dias;
- MapBiomas mais recente usado como característica estrutural não coincide temporalmente com 2025;
- ainda existem valores meteorológicos ausentes, tratados dentro das pipelines;
- o impacto financeiro depende de premissas ilustrativas enquanto não houver custos reais do cliente;
- a análise de viés é limitada à geografia porque a base não possui atributos demográficos;
- o modelo precisa ser revalidado com outras épocas do ano e novos anos antes de uso crítico.

---

# Conclusão

O projeto entrega uma pipeline completa e documentada: consolidação e qualidade dos dados, EDA, teste de hipótese, três modelos de classificação, métricas de negócio, impacto financeiro, plano de teste A/B, otimização com Optuna, explicabilidade com SHAP e análise de desempenho por grupos geográficos.

O resultado final atende às metas definidas no início do projeto e demonstra potencial como prova de conceito para priorização de alertas de queimadas. Antes de produção, a recomendação é ampliar o histórico, executar o teste controlado e implementar monitoramento contínuo por região.
