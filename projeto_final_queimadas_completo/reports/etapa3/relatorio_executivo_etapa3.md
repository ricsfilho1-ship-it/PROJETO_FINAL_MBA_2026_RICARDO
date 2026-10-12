# Relatório executivo — Etapa 3

## Otimização

O LightGBM foi otimizado com **Optuna** em 6 trials. A função objetivo foi o ROC-AUC médio de três janelas de validação temporal construídas somente dentro do período de treinamento. O conjunto de teste não participou da escolha de hiperparâmetros nem do threshold.

- melhor ROC-AUC médio na CV temporal: **0.814**;
- threshold operacional escolhido na validação: **0.34**.

## Resultado final no teste

- Accuracy: **0.758**
- Precision: **0.360**
- Recall: **0.716**
- F1-score: **0.479**
- ROC-AUC: **0.818**
- False Negative Rate: **0.284**

Comparado à Etapa 2, o F1 passou de **0.466** para **0.479**, o ROC-AUC de **0.809** para **0.818** e o recall de **0.701** para **0.716**.

## Explicabilidade SHAP

As três variáveis com maior importância global agregada pelo valor absoluto médio de SHAP foram: **mb_log_total_area, temperatura_maxima, radiacao_global_diaria**. A importância SHAP mede contribuição para a predição do modelo; ela não deve ser interpretada como causalidade.

## Análise de viés

Como a base não possui gênero, raça ou outros atributos demográficos, a análise de viés possível neste projeto foi geográfica. O desempenho foi comparado por região e por UF, com um filtro mínimo para evitar conclusões baseadas em grupos muito pequenos.

- gap de recall entre regiões elegíveis: **0.257**;
- região com maior recall: **Centro-Oeste**;
- região com menor recall: **Sudeste**;
- UFs elegíveis para comparação: **16**.

Diferenças entre regiões não provam viés injusto por si só: também podem refletir composição climática, frequência de eventos e tamanho da amostra. Antes de uso operacional, recomenda-se monitorar recall/FNR por região ao longo do tempo e ampliar a janela histórica.

## Recomendações

1. revalidar o modelo com pelo menos um ciclo anual completo e, idealmente, anos adicionais;
2. monitorar drift e métricas por região/UF;
3. manter o threshold como parâmetro de negócio, revisado conforme o custo real de falsos negativos e falsos positivos;
4. usar SHAP como apoio de explicação, sem tratá-lo como evidência causal;
5. somente considerar produção após teste A/B/controlado e definição de governança dos alertas.
