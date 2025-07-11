# 🧠 Desafio de Previsão do IBOVESPA

Este projeto foi desenvolvido como parte de um desafio técnico em Ciência de Dados, com o objetivo de prever a **tendência diária do índice IBOVESPA** — se o fechamento do dia seguinte será de alta ou baixa — com base em seus dados históricos.

A proposta simula um cenário real dentro de um grande fundo de investimentos, onde a solução seria utilizada por analistas quantitativos como suporte em dashboards de decisão. O foco está em aplicar técnicas de análise de dados e machine learning para construir um modelo preditivo com **mínimo de 75% de acurácia** no conjunto de teste.

Para isso, foram utilizados dados históricos do índice IBOVESPA, com período diário e abrangência de dez anos. Todo o pipeline foi construído desde a coleta, limpeza e preparação dos dados até a avaliação do modelo.

---

## 🚀 **Instalação**

Antes de instalar os requisitos do projeto, você precisa ter as seguintes ferramentas instaladas no seu sistema:

* Python 3.11
* Git

Após instalar os pré-requisitos, siga os passos abaixo para configurar o projeto:

1.  **Clone o repositório:**

    ```bash
    git clone https://github.com/JacksonvBarbosa/Analise_Indice_Ibovespa/tree/sofia
    cd Analise_Indice_Ibovespa
    ```

2.  **Crie e ative um ambiente virtual:**

    ```bash
    python -m venv venv
    # No Windows:
    .\venv\Scripts\activate
    # No macOS/Linux:
    source venv/bin/activate
    ```

3.  **Instale as dependências do projeto:**

    ```bash
    pip install -r requirements.txt
    ```

---

## 📊 **Dados**

Os dados utilizados são históricos do índice **IBOVESPA**, disponíveis publicamente em:  
🔗 [Investing.com – BOVESPA Dados Históricos](https://br.investing.com/indices/bovespa-historical-data)

---

## 💻 **Tecnologias utilizadas**

* **Coleta e manipulação de dados:** `pandas`
* **Indicadores técnicos:** `pandas_ta`
* **Visualização:** `matplotlib`, `seaborn`, `plotly`
* **Modelagem:** `scikit-learn`
