# Modelo de Probabilidade de Default (PD) 

## 📌 Descrição do Projeto  

Este repositório contém o desenvolvimento de um **modelo de Probabilidade de Default (PD)** aplicado sobre uma base fictícia de clientes, transações e histórico de inadimplência.  

O fluxo foi estruturado para refletir as **boas práticas de modelagem de risco de crédito**, abordando os processos de ***feature engineering*, consolidação da ABT (Analytical Base Table), seleção de variáveis e a modelagem em si**.  

- **Desenvolvimento do Projeto:** [Projeto PD](notebooks/case_PD.ipynb)  

---

## Estrutura do projeto

```bash
├── data/                   # Dados brutos e processados 
│   ├── raw/                # Bases originais (ex: clientes_case.csv, transacoes_case.csv)
│   └── processed/          # ABTs finais prontas para modelagem        
│
├── features/                             # Scripts modulares para criação 
│   ├── features_clientes_transacional.py # das *features* utilizadas na modelagem 
│   ├── features_clientes.py
│   ├── features_flags.py
│   ├── features_quantidade.py
│   ├── features_tempo.py
│   ├── features_valor.py
│
├── notebooks/              # Desenvolvimento do modelo. Contém análises exploratórias e
│   └── case_PD.ipynb       # modelagem
│
├── pipeline/               # Scripts modulares para execução do pipeline. 
│   ├── carregar_dados.py
│   ├── criar_abt.py
│   ├── preprocess.py
│   └── utils.py            # Funções auxiliares para o estudo
│
├── requirements.txt        # Dependências do projeto
```
---

## Entendimento do contexto

O objetivo é estimar a **probabilidade de inadimplência (default)** em determinado horizonte de tempo, considerando o perfil cadastral, histórico transacional e comportamento prévio dos clientes.  

**DEFAULT DA CARTEIRA DE CLIENTES: 10%**

**KPIs de avaliação:**  
- AUC 
- KS
- *Precision* e *Recall*

---

## Pré-Processamento dos Dados  

1. Padronização e consistência de colunas.  
2. Construção da **ABT (Analytical Base Table)** com janelas de observação e performance (`mes_safra`).  
   - Opção por utilizar apenas informações **de M-1** com objetivo de evitar vazamento de informações de safras ainda não maturadas, mas que por algum motivo ainda tinham marcação de *default*.  
3. Tratamento de valores faltantes e variáveis categóricas.  

---

## Pré-seleção de variáveis

Antes da modelagem, foi realizada uma etapa de pré-seleção, tendo como critério:  

- **Correlação**: remoção de variáveis com correlação > 0.8, mantendo aquelas com maior IV.  

---

## Modelagem  

Modelos testados:

- **Decision Trees** 
- **Regressão Logística** 
- **CatBoost** 

### Estratégia de Treino  
- Divisão treino/validação de forma aleatória;
- Tunig de hiperparâmetros com **RandomizedSearchCV**

### 📊 Resultados  

- Baixo poder das variáveis de discriminar o público bom do mau (KS=)
- AUC no patamar de 50%, evidenciando o desbalanceamento como principal problema a ser superado.

---

## 📈 Conclusão Estratégica  

- A abordagem confirmou qinicialmente que **variáveis transacionais e de recência** têm maior relevância na explicação do default
- O desbalanceamento se mostrou a maior questão a ser resolvida, afetando todos os modelos
- É preciso adicionar novas informações, seja com novos fornecedores de dados ou disponibilização de mais dados já existentes
---

## Próximos Passos  

1. **Feature Engineering mais profunda** – incluir variáveis provenientes de novas fontes, acrescentar efeitos de interação e efeitos de sazonalidade que possam explicar melhor o *target* 
2. **Refazer modelos** - com a inclusão de variáveis mais relevantes, ao mesmo tempo em que se trata o desbalanceamento das classes abordagens como alteração do parâmetro *class_weights* nos modelos
4. **Versionamento** – Integrar MLflow para versionar modelos e garantir uma reprodutibilidade mais robusta
3. **Deploy** – Implantar o modelo como API para realizar predições em *batch* ou em tempo real

