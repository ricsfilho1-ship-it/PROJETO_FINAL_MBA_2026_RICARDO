# Correções realizadas após a avaliação da Etapa 1

## 1. Municípios/linhas sem dados climáticos

**Problema identificado:** havia linhas em que todas as variáveis meteorológicas estavam ausentes, o que permitia ao modelo trabalhar apenas com informações constantes do município.

**Correção aplicada:**
- auditoria explícita por linha e por município;
- identificadas **4.451 linhas município-dia sem nenhuma variável climática**;
- identificados **85 municípios com 0 dias de clima em todo o período**;
- todas as linhas sem qualquer dado meteorológico foram removidas da base de ML;
- base final: **18.855 linhas e 457 municípios**.

Arquivos de auditoria:
- `reports/qualidade/cobertura_climatica_municipios.csv`
- `reports/qualidade/municipios_excluidos_sem_clima.csv`

## 2. Valores fisicamente inválidos

**Problema identificado:** o IQR havia sido interpretado de forma ampla demais, tratando extremos como potencialmente reais sem uma checagem física anterior.

**Correção aplicada:**
- separação entre controle de qualidade físico e outlier estatístico;
- regras básicas para precipitação, pressão, umidade, vento e consistência de temperatura;
- para radiação, cálculo da radiação extraterrestre diária pela formulação FAO-56;
- radiação diária `<= 0` ou acima de 105% do limite extraterrestre é convertida para `NaN`;
- **45 registros estação-dia de radiação** foram sinalizados e corrigidos, incluindo os valores zero e o pico de aproximadamente 87,9 MJ/m².

Arquivos de auditoria:
- `reports/qualidade/radiacao_invalida.csv`
- `reports/qualidade/regras_qualidade.csv`

O IQR continua no EDA, mas agora apenas como detector de extremos estatísticos, sem remoção automática.

## 3. Dicionário MapBiomas

**Problema identificado:** as colunas apareciam somente como IDs de classe.

**Correção aplicada:**
- nomes das colunas substituídos por nomes interpretáveis da legenda oficial da Coleção 7;
- dicionário contém ID da classe, nome, unidade, origem e uso no projeto;
- o ID `0`, que aparece no arquivo recebido mas não consta na legenda oficial consultada, foi mantido explicitamente como `mb_area_id0_sem_legenda`, sem inventar significado.

Arquivo:
- `data/processed/dicionario_dados.csv`

## 4. Organização e reprodutibilidade do repositório

**Problema identificado:** estrutura do Git não correspondia ao README, arquivos `.git` foram incluídos e o notebook dependia do diretório de execução.

**Correção aplicada:**
- repositório reorganizado em `notebooks/`, `src/`, `data/processed/`, `reports/`;
- nenhum diretório `.git` incluído no pacote;
- `.gitignore` criado;
- arquivos brutos ficam em `data/raw/` e são ignorados pelo Git;
- notebook agora localiza automaticamente a raiz do projeto e foi testado a partir da raiz do repositório;
- `src/consolidar_base.py` foi testado e reproduz exatamente o CSV entregue.

## 5. EDA adicional

Foram adicionados:
- histogramas das variáveis meteorológicas;
- análise da composição MapBiomas;
- recorte por região;
- recorte por UF com controle de tamanho mínimo de amostra;
- auditoria de cobertura climática;
- visualizações adicionais salvas em `reports/figures/`.

## 6. Metas para KPIs

Foram adicionadas metas explícitas para a Etapa 2:

- Recall positivo >= 0,70;
- F1-score >= 0,40;
- Precision >= 0,30;
- ROC-AUC >= 0,75;
- False Negative Rate <= 0,30;
- Acurácia será monitorada, mas não usada isoladamente.
