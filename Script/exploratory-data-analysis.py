# %%
import pandas as pd
import numpy as np
from scipy import stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %%
# Configurar paleta de cores e estilo

# Matplotlib
mpl.rcParams["figure.figsize"] = (20, 7)
mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["font.size"] = 12
mpl.rcParams["figure.titlesize"] = 25
mpl.rcParams["figure.dpi"] = 100

# Seaborn
sns.set_style('whitegrid', {"grid.color": ".8", "grid.linestyle": "--"})
sns.set_palette(palette='mako')
# %%

input_path = '../data/processed/arquivo-exploratorio.csv'

df_ex = pd.read_csv(input_path, thousands='.', decimal=',', parse_dates=['ds'], date_format='%d.%m.%Y', index_col='ds')
df_ex.index = pd.to_datetime(df_ex.index)
df_ex


# %%
df_ex.info()
# %%
df_ex.describe()
# %%
df_ex.columns
# %%
# Distribuição do retorno diário
sns.histplot(df_ex['delta'], kde=True)

# Título e rótulos dos eixos
plt.title('Distribuição do retorno diário')
plt.xlabel('Variação diária (%)')
plt.ylabel('Contagem')

plt.show()
# %%
# Teste de hipótese da distribuição do retorno diário
stat, p = stats.shapiro(df_ex['delta'])
print(f'Estatística do teste: {stat:.4f}\nValor-p: {p}')

# Nível de significância de 95%
alpha = 0.05
if p > alpha:
    print("\nConclusão: Não há evidência suficiente para rejeitar a hipótese nula. Os dados seguem uma distribuição normal.")
else:
    print("\nConclusão: Rejeitamos a hipótese nula. Os dados NÃO seguem uma distribuição normal.")
# %%

# correlação 1

# mapa de calor de correlação das variáveis
columns_corr1 = ['target', 'close', 'delta', 'return', 'return_1d', 'return_2d',
                'return_3d', 'return_5d', 'momentum_aceel', 'momentum_consistency',
                'vol_3d', 'vol_10d', 'vol_regime', 'open', 'high', 'low', 'rsi_9',
                'rsi_14', 'rsi_divergence', 'macd', 'macd_signal']

corr = df_ex[columns_corr1].corr(numeric_only=True)

mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, mask=mask, annot=True, fmt='.2f')
plt.title('Mapa de calor de correlações 1ª metade dos dados')
plt.grid(False)
plt.show()

# %%

# correlação 2

columns_corr2 = ['target', 'close',
                'stoch_k', 'stoch_d', 'volatilidade', 'volatilidade_relativa', 'ema_5',
                'ema_10', 'ema_cross', 'target_lag1', 'target_lag2', 'target_lag3',
                'target_lag4', 'target_lag5', 'target_lag6', 'target_lag7', 'macd_hist',
                'close_lag1', 'close_lag3', 'close_lag5', 'close_lag7', 'daily_return'
]

corr = df_ex[columns_corr2].corr(numeric_only=True)

mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, mask=mask, annot=True, fmt='.2f')
plt.title('Mapa de calor de correlações 2ª metade dos dados')
plt.grid(False)
plt.show()


# %%

plt.plot(df_ex.index, df_ex['close'],color= 'orange')
plt.xlabel('Data')
plt.ylabel('Pontos')
plt.grid(False)
plt.show()

# %%
