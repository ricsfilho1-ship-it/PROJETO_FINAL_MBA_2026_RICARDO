# Plano de Teste A/B — FireRisk AI

## Objetivo
Validar se disponibilizar alertas preditivos do modelo reduz eventos não antecipados sem gerar um volume operacional inviável de falsos alertas.

## Hipótese
**H0:** o uso do alerta não reduz a taxa de eventos não antecipados em relação ao processo atual.  
**H1:** o uso do alerta reduz a taxa de eventos não antecipados em relação ao processo atual.

## Desenho
- **Grupo A (controle):** processo atual, sem alertas do modelo.
- **Grupo B (tratamento):** processo atual + alertas FireRisk AI.
- **Unidade de randomização:** município ou unidade operacional, para reduzir contaminação entre grupos.
- **Estratificação:** região e faixa histórica de risco antes da randomização.
- **Threshold pré-registrado:** 0,42, salvo mudança formal antes do início do teste.

## KPIs
**Primário:** proporção de eventos reais antecipados (recall operacional).  
**Secundários:** falsos negativos, falsos positivos, tempo de resposta, custo por alerta, custo total e percentual de alertas acionáveis.

## Duração
Executar durante uma janela que inclua quantidade suficiente de eventos e condições meteorológicas variadas. O tamanho amostral deve ser calculado antes do início usando a taxa histórica de ocorrência, alfa de 5%, poder de 80% e o efeito mínimo de negócio considerado relevante.

## Critério de sucesso
O tratamento deve reduzir os eventos não antecipados e o custo total sem ultrapassar o limite operacional acordado para falsos alertas.

## Cuidados
- Não alterar o threshold durante o teste sem regra pré-definida.
- Monitorar indisponibilidade de dados meteorológicos.
- Não usar o resultado deste teste para afirmar causalidade fora da população/época observada sem nova validação.
