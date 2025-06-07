## 🧠 Desafio de Previsão do IBOVESPA

Este projeto foi desenvolvido como parte de um desafio técnico em Ciência de Dados, com o objetivo de prever a **tendência diária do índice IBOVESPA** — se o fechamento do dia seguinte será de alta ou baixa — com base em seus dados históricos.

A proposta simula um cenário real dentro de um grande fundo de investimentos, onde a solução seria utilizada por analistas quantitativos como suporte em dashboards de decisão. O foco está em aplicar técnicas de análise de dados e machine learning para construir um modelo preditivo com **mínimo de 75% de acurácia** no conjunto de teste.

Para isso, foram utilizados dados históricos do índice IBOVESPA, com período diário e abrangência mínima de dois anos. Todo o pipeline foi construído desde a coleta, limpeza e preparação dos dados até a avaliação do modelo.

O objetivo principal é detectar padrões relevantes que possam antecipar o comportamento do mercado no curto prazo, servindo como ponto de partida para análises quantitativas mais robustas.

---

### 📊 **Dados**

Os dados utilizados são históricos do índice **IBOVESPA**, disponíveis publicamente em:  
🔗 [Investing.com – BOVESPA Dados Históricos](https://br.investing.com/indices/bovespa-historical-data)

> Recomendação: selecionar o período **“diário”** e baixar pelo menos **2 anos de dados** para garantir a robustez do modelo.

---

## 💻 Tecnologias Utilizadas

- **Python** – Linguagem principal do projeto  
- **Pandas** – Manipulação e análise de dados tabulares  
- **NumPy** – Operações numéricas e vetoriais  
- **Scikit-learn (sklearn)** – Modelos de machine learning e métricas de avaliação  
- **Matplotlib** – Geração de gráficos e visualizações  
- **Seaborn** – Visualização estatística avançada  
- **Jupyter Notebook** – Ambiente de desenvolvimento interativo  
- **Google Colab** – Execução em nuvem e compartilhamento de notebooks
