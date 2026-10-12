# Modelos salvos

## Etapa 2

`lightgbm_stage2.joblib` — pipeline selecionada na Etapa 2, com threshold operacional 0,42.

## Etapa 3

`lightgbm_stage3_evaluation.joblib` — modelo usado para produzir as métricas finais de teste da Etapa 3. Foi treinado somente no período de treino; o threshold **0.34** foi selecionado na validação.

`lightgbm_stage3_final.joblib` — modelo final reajustado em treino + validação depois da avaliação. O conjunto de teste não entrou no treinamento.

Para produzir a classe operacional:

```python
prob = model.predict_proba(X)[:, 1]
pred = (prob >= 0.34).astype(int)
```

O método `predict()` usa 0,50 por padrão e não reproduz o threshold operacional do projeto.
