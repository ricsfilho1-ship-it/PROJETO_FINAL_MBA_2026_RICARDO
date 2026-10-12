# Relatório Executivo — Etapa 2

## Objetivo
Treinar e comparar três modelos de classificação para prever ocorrência de queimadas no nível município-dia usando a base corrigida da Etapa 1.

## Divisão temporal
- Treino: 2025-01-01 a 2025-01-28 — 12,342 linhas.
- Validação: 2025-01-29 a 2025-02-04 — 3,046 linhas.
- Teste: 2025-02-05 a 2025-02-12 — 3,467 linhas.

## Comparação no teste — threshold 0,50
| modelo              |   acuracia |   precisao |   recall |   f1_score |   roc_auc |   false_negative_rate |
|:--------------------|-----------:|-----------:|---------:|-----------:|----------:|----------------------:|
| Regressão Logística |      0.662 |      0.288 |    0.796 |      0.422 |     0.79  |                 0.204 |
| Random Forest       |      0.845 |      0.504 |    0.343 |      0.408 |     0.796 |                 0.657 |
| LightGBM            |      0.785 |      0.385 |    0.642 |      0.482 |     0.809 |                 0.358 |

A acurácia do baseline que sempre prevê ausência de queimada é **0.845**, reforçando que acurácia isolada é inadequada para este problema.

## Modelo selecionado
O LightGBM teve o melhor F1-score e ROC-AUC na comparação padrão. O threshold foi então ajustado usando **somente a validação**.

Threshold operacional: **0,42**.

Resultados finais no teste:
- Precision: **0.349**
- Recall: **0.701**
- F1-score: **0.466**
- ROC-AUC: **0.809**
- False Negative Rate: **0.299**

Todas as metas definidas na Etapa 1 são atingidas no conjunto de teste com o threshold operacional.

## Impacto financeiro
A base não contém custos reais, então o cálculo é apresentado como **cenário parametrizável**, não como economia comprovada.

No cenário base ilustrativo (FN = R$ 50.000 e FP = R$ 2.000), o LightGBM operacional apresenta economia estimada de **R$ 17,492,000** no recorte de teste em relação ao cenário sem alerta.

## Robustez geográfica
Como geografia e MapBiomas são constantes por município, foi incluído um teste adicional em municípios totalmente não vistos durante o treino. O LightGBM obteve:
- Recall: **0.746**
- Precision: **0.382**
- F1-score: **0.505**
- ROC-AUC: **0.821**

Esse resultado reduz, embora não elimine, a preocupação de que o desempenho dependa apenas de memorizar os municípios.

## Próxima etapa
Na Etapa 3, o LightGBM será otimizado com Optuna, explicado com SHAP e avaliado quanto a possíveis diferenças de desempenho por região/UF.
