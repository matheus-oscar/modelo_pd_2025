# Modelo de Probabilidade de Default (PD) 

## 📌 Descrição do Projeto  

Este repositório contém o desenvolvimento de um **modelo de Probabilidade de Default (PD)** aplicado sobre uma base fictícia de clientes, transações e histórico de inadimplência.  

O fluxo foi estruturado para refletir as **boas práticas de modelagem de risco de crédito**, abordando os processos de ***feature engineering*, consolidação da ABT (Analytical Base Table), seleção de variáveis e a modelagem em si**.  

- **Desenvolvimento do Projeto:** [Projeto PD](notebook/case_PD.ipynb)  

---

## Estrutura do projeto

.
├── data/                   # Dados brutos e processados 
│   ├── raw/                # Bases originais (ex: clientes_case.csv, transacoes_case.csv)
│   └── processed/          # ABTs finais prontas para modelagem        
├── notebooks/              # Desenvolvimento do modelo. Contém análises exploratórias e
│   └── case_PD.ipynb       # modelagem
│
├── pipeline/               # Scripts modulares para execução do pipeline. 
│   ├── feature_clientes.py
│   ├── feature_transacoes.py
│   ├── feature_inadimplencia.py
│   ├── criar_abt.py      # Consolidação da ABT
│   └── utils.py            # Funções auxiliares reutilizáveis
│
├── reports/                # Saídas analíticas
│   ├── resultados.pdf      # Relatório consolidado
│   ├── figures/            # Gráficos e imagens finais
│   └── logs/               # Logs de execução
│
├── tests/                  # Testes unitários e validação do pipeline
│   └── test_utils.py
│
├── requirements.txt        # Dependências do projeto
├── .gitignore              # Arquivos/pastas ignorados no versionamento
└── README.md               # Documentação principal


---

## 🎯 Entendimento do Negócio  

O objetivo é estimar a **probabilidade de inadimplência (default)** em determinado horizonte de tempo, considerando o perfil cadastral, histórico transacional e comportamento prévio dos clientes.  

**DEFAULT DA CARTEIRA DE CLIENTES: 10%**

**KPIs de avaliação:**  
- AUC (ROC-AUC) 
- KS (Kolmogorov-Smirnov)  
- *Precision* e *recall*

---

## ⚙️ Pré-Processamento dos Dados  

1. Padronização e consistência de colunas.  
2. Construção da **ABT (Analytical Base Table)** com janelas de observação e performance (`mes_safra`).  
   - Garantido que apenas informações **de M-1** estejam disponíveis em M, evitando vazamento de informações de safras ainda não maturadas.  
3. Tratamento de valores faltantes e variáveis categóricas.  

---

## 🛠️ Seleção de Variáveis  

Antes da modelagem, foi realizada uma etapa de pré-seleção:  

- **Correlação**: remoção de variáveis com correlação > 0.8.  
- **Information Value (IV)**: exclusão de variáveis com baixo poder discriminatório.  

---

## 🤖 Modelagem  

Modelos avaliados:  
- **Decision Trees** – baseline com regras simples de classificação.  
- **Regressão Logística** – referência estatística e interpretabilidade.  
- **CatBoost** – algoritmo de *gradient boosting* para maior performance esperada.  

### 🔎 Estratégia de Treino  
- Divisão treino/validação baseada em `mes_safra`.  
- **RandomizedSearchCV** para ajuste de hiperparâmetros.  

### 📊 Resultados  


---

## 📈 Conclusão Estratégica  

- A abordagem confirmou que **variáveis transacionais e de recência** têm maior relevância na explicação do default.  
- O processo de **seleção de variáveis (correlação + IV)** reduziu dimensionalidade e aumentou a robustez do modelo.  
- O desbalanceamento se mostrou a maior questão a ser resolvida

---

## 🚀 Próximos Passos  

1. **Feature Engineering avançada** – incluir novas variáveis derivadas, interações e efeitos de sazonalidade.  
2. **Análise de Lifting** – avaliação da capacidade de segmentação dos clientes em decisores de negócio.  
3. **Balanceamento de classes** – investigar técnicas de oversampling/undersampling.  
4. **Deployment** – disponibilizar o modelo como API ou batch processável.  
5. **Versionamento** – acoplar rastreamento de experimentos (ex: MLflow).  
