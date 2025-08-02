# %%

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def wmape(y_true, y_pred):
    """
    Calcula o Weighted Mean Absolute Percentage Error (WMAPE).

    Args:
        y_true (array-like): Valores reais.
        y_pred (array-like): Valores previstos.

    Returns:
        float: O valor do WMAPE. Retorna np.nan se a soma dos valores reais absolutos for zero
               para evitar divisão por zero.
    """
    # Garante que y_true e y_pred são arrays numpy para operações elemento a elemento
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calcula o erro absoluto
    absolute_error = np.abs(y_true - y_pred)

    # Soma dos valores reais absolutos
    sum_abs_y_true = np.sum(np.abs(y_true))

    # Evita divisão por zero
    if sum_abs_y_true != 0:
        wmape_value = np.sum(absolute_error) / sum_abs_y_true
    else:
        wmape_value = np.nan # Retorna NaN se a soma dos valores reais absolutos for zero

    return wmape_value
# %%

input_path = '../data/processed/dados_historicos_ibovespa_2015-2025_processed.csv'
# Carregar o dataset
df = pd.read_csv(input_path, index_col='ds', parse_dates=['ds'])

# Converter 'ds' para datetime e ordenar
'''df['ds'] = pd.to_datetime(df['ds'])
df = df.sort_values(by='ds').reset_index(drop=True)'''

df
# %%

# 1. Escolha da Variável Alvo (y)
target = 'close' # Ou 'daily_return'

# 2. Seleção de Features (X)
# Adicionar lag do 'close' como feature
df['close_lag1'] = df['close'].shift(1)
df['close_lag3'] = df['close'].shift(3)
df['close_lag5'] = df['close'].shift(5)
df['close_lag7'] = df['close'].shift(7) # Exemplo: preço de fechamento de 7 dias atrás

# Lista de features candidatas
features = [
    'close_lag1', 'close_lag3', 'close_lag5','close_lag7', # Adicionando lags do target
    'daily_return', # Se o target for 'close', podemos usar o daily_return D-1 como feature, ou o daily_return_lag_1
    'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_4', 'return_lag_5',
    'momentum_5', 'momentum_21', 'momentum_63',
    'sma_21', 'ema_50', 'rsi_14', 'atr_14', 'obv',
    'day_of_week', 'month', # day_of_month pode ter muita variação
]

# Excluir linhas com NaN (resultantes dos lags)
df_clean = df.dropna(subset=features + [target])

# Definir X e y
X = df_clean[features]
y = df_clean[target]

# %%

# Dividir dados em treino e teste (respeitando a ordem temporal)
# Exemplo: 80% treino, 20% teste
train_size = int(len(df_clean) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Opcional: Escalar features (XGBoost geralmente não exige, mas pode ajudar)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)


# %%

# 3. Modelo de Regressão: XGBoost
model = xgb.XGBRegressor(
    objective='reg:squarederror', # Objetivo para regressão
    n_estimators=300,            # Número de árvores
    learning_rate=0.05,          # Taxa de aprendizado
    max_depth=5,                 # Profundidade máxima das árvores
    subsample=0.8,               # Fração de amostras por árvore
    colsample_bytree=0.8,        # Fração de features por árvore
    random_state=42,             # Para reprodutibilidade
    n_jobs=-1                    # Usar todos os núcleos da CPU
)

# Treinamento do modelo
print("Treinando o modelo...")
model.fit(X_train, y_train) # Use X_train_scaled se estiver escalando

# Previsão
print("Fazendo previsões...")
predictions = model.predict(X_test) # Use X_test_scaled se estiver escalando

# %%

# Avaliação do Modelo
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\n--- Avaliação do XGBoost ---")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R2 Score: {r2:.4f}")

# Exemplo de WMAPE (você já tem a função wmape definida)
# Certifique-se de que a função wmape está definida no seu ambiente
wmape_score = wmape(y_test.values, predictions)
print(f"WMAPE: {wmape_score:.2%}")

# %%

# Visualização das previsões
plt.figure(figsize=(15, 7))
plt.plot(y_train.index, y_train, label='Dados de Treino (Real)')
plt.plot(y_test.index, y_test, label='Dados de Teste (Real)')
plt.plot(y_test.index, predictions, label='Previsões (XGBoost)', linestyle='--')
plt.title(f'Previsão do Ibovespa - {target}')
plt.xlabel('Índice da Amostra (Ordenado por Data)')
plt.ylabel(target)
plt.legend()
plt.grid(True)
plt.show()

# %%

# Importância das Features
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
feat_importances.nlargest(15).plot(kind='barh')
plt.title('Importância das Features (Top 15)')
plt.xlabel('Importância Relativa')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
# %%
