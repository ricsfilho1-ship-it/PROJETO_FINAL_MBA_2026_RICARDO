# Entrega final — FireRisk AI

## Checklist dos requisitos

### Etapa 1

- [x] variável alvo definida: `fire_occurrence`;
- [x] KPIs e metas definidos;
- [x] script de consolidação em Python;
- [x] EDA completo;
- [x] análise de correlação e distribuição do alvo;
- [x] missing values e outliers;
- [x] histogramas e análises geográficas/MapBiomas;
- [x] teste de hipótese estatístico;
- [x] base processada;
- [x] dicionário de dados;
- [x] README e estrutura GitHub corrigida.

### Etapa 2

- [x] três modelos de ML;
- [x] Accuracy, Precision, Recall, F1 e ROC-AUC;
- [x] matriz de confusão e curvas de comparação;
- [x] cálculo de impacto financeiro parametrizável;
- [x] cenários financeiros documentados como ilustrativos;
- [x] plano de teste A/B;
- [x] escolha justificada do LightGBM;
- [x] threshold ajustado somente com validação.

### Etapa 3

- [x] otimização do LightGBM com Optuna;
- [x] validação temporal interna sem uso do teste;
- [x] SHAP;
- [x] identificação das três variáveis de maior peso;
- [x] análise de desempenho por região e UF;
- [x] discussão de limitações de fairness;
- [x] modelo final serializado;
- [x] relatório executivo da Etapa 3.

## Resultado final

O LightGBM otimizado atingiu no teste:

- Recall: **0.716**;
- Precision: **0.360**;
- F1: **0.479**;
- ROC-AUC: **0.818**;
- FNR: **0.284**.

Todas as metas de KPI definidas na Etapa 1 foram atendidas.

## Arquivos principais para avaliação

1. `README.md`
2. `notebooks/01_EDA_queimadas.ipynb`
3. `notebooks/02_modelagem_inteligencia_negocio.ipynb`
4. `notebooks/03_otimizacao_ia_responsavel.ipynb`
5. `data/processed/dataset_ml.csv`
6. `data/processed/dicionario_dados.csv`
7. `reports/etapa2/relatorio_executivo_etapa2.md`
8. `reports/etapa3/relatorio_executivo_etapa3.md`
9. `models/lightgbm_stage3_final.joblib`

## Observação para GitHub

A pasta `.git` não faz parte do pacote. Depois de extrair o ZIP, inicialize ou copie o conteúdo para o repositório GitHub preservando a estrutura de pastas.
