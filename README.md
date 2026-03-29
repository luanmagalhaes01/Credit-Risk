# Credit Risk Analysis

Projeto de análise e predição de risco de crédito utilizando Machine Learning para identificar os principais fatores que contribuem para a inadimplência e prever a probabilidade de default de novos clientes.

## Objetivo

Auxiliar instituições financeiras a melhorar suas estratégias de concessão de crédito por meio de modelos preditivos baseados em dados históricos de empréstimos.

**Variável-alvo**: `loan_status` (0 = Adimplente, 1 = Inadimplente)

## Dataset

**Fonte**: [Credit Risk Dataset — Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data)

- 32.416 registros (32.396 após pré-processamento)
- 12 variáveis originais → 21 features após encoding
- Desbalanceamento de classes: 78,1% adimplente / 21,9% inadimplente

**Principais variáveis**:

| Categoria | Variáveis |
|-----------|-----------|
| Perfil do cliente | `person_age`, `person_income`, `person_home_ownership`, `person_emp_length` |
| Características do empréstimo | `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`, `loan_percent_income` |
| Histórico de crédito | `cb_person_default_on_file`, `cb_person_cred_hist_length` |

## Pipeline

1. **EDA** — distribuições, análise bivariada, correlações e comparação entre adimplentes e inadimplentes
2. **Pré-processamento** — imputação de valores ausentes (mediana), remoção de duplicatas, One-Hot Encoding e StandardScaler
3. **Modelagem** — Logistic Regression, Random Forest e Gradient Boosting com validação cruzada estratificada (5 folds)
4. **Avaliação** — AUC-ROC, precisão, recall e F1-Score com foco em minimizar falsos negativos

## Resultados

| Modelo | AUC-ROC (Test) | AUC-ROC (CV) | Precision (Default) | Recall (Default) | F1-Score |
|--------|:--------------:|:------------:|:-------------------:|:----------------:|:--------:|
| Logistic Regression | 0.8575 | 0.8524 | 0.5064 | 0.7814 | 0.6145 |
| Random Forest | 0.9176 | 0.9140 | 0.7056 | 0.7842 | 0.7428 |
| Gradient Boosting | 0.9332 | 0.9285 | 0.9376 | 0.7102 | 0.8082 |

O **Gradient Boosting** obteve o melhor AUC-ROC geral, mas a **Regressão Logística** foi recomendada para produção por sua interpretabilidade, auditabilidade e compatibilidade com formatos de scorecard regulatório.

## Principais Fatores de Risco

1. **Grade do empréstimo** e **taxa de juros** — preditores mais fortes de default
2. **Histórico de inadimplência anterior** — aumenta significativamente o risco
3. **Finalidade do empréstimo** — consolidação de dívidas apresenta maior risco relativo
4. **Renda e relação dívida/renda** — menor renda e maior comprometimento de renda elevam o risco

## Tecnologias

- **Python 3.10**
- `pandas`, `numpy` — manipulação de dados
- `matplotlib`, `seaborn` — visualização
- `scikit-learn` — modelagem e métricas
- `statsmodels` — análise de multicolinearidade (VIF)
