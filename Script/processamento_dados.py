# %%
# Classe funções processadas
import funcoes_processed as fp

import pandas as pd
import yfinance as yf
import numpy as np
#import pandas_ta as ta
#import talib as ta
#from finta import TA

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# %%

input_path = '../data/raw/dados_historicos_ibovespa_2008-2025.csv'

df = pd.read_csv(input_path, thousands='.', decimal=',', parse_dates=['Data'], date_format='%d.%m.%Y', index_col='Data')
df = df.rename_axis('ds').sort_index()
df.tail()

# %%
df.info()
# %%
# renomeando as colunas para os nomes padrões utilizados no mercado financeiro
colunas = {
  'Último': 'close',              # fechamento da negociação diária
  'Abertura': 'open',             # início da negociação diária
  'Máxima': 'high',               # valor máximo do dia
  'Mínima': 'low',                # valor mínimo do dia
  'Vol.': 'volume',               # volume de negociação diária
  'Var%': 'daily_return'          # variação percentual diária
}

df.rename(columns=colunas, inplace=True)
# %%
# conferindo se há valores nulos
df.isna().sum().sort_values(ascending=False)
# %%

df['volume'] = df['volume'].apply(fp.converter_volume)
# %%

# substituir o volume nulo pela média do volume anterior e posterior daquela data
df['volume'] = df['volume'].interpolate()
# %%
# ajustando a coluna variação percentual diária, que contém o pct_change() do fechamento
df['daily_return'] = df['daily_return'].str.replace('%', '').str.replace(',', '.')
df['daily_return'] = round(df['daily_return'].astype(float) / 100, 4)
df.head()
# %%
# Cria um DataFrame limpo
df_prep = pd.DataFrame()

# %%
# DATAFRAME PARA RODAR OS MODELOS
# Chama função
fp.cria_variaveis_proximo_dia(df_prep, df)
# Remove valores nulos
df_prep = df_prep.dropna(axis=0)
df_prep
# %%
n_lags = 7

# Chama função cria colunas com lags
df_prep = fp.make_n_lags(df_prep, n_lags, "target", 1)
df_prep = fp.make_n_lags(df_prep, n_lags, "close", 2)
# %%

df_model_data = df_prep.copy()
df_model_data['daily_return'] = df['daily_return'].copy()
df_model_data['daily_return']
df_model_data = df_model_data[df_model_data.index >= '2015-06-17'].dropna().copy()

# %%
# Exclui tabela Close que não vai ser usada
df_prep.drop('close', axis=1, inplace=True)
# Data Frame pronto para o modelo doprando linhas com valores na
df_model = df_prep.dropna().copy()

# %%
# Salva arquivo removendo o index
df_model.to_csv('../data/processed/arquivo-modelo.csv', index=False)

# %%

# Salva arquivo mantendo o index
df_model_data.to_csv('../data/processed/arquivo-exploratorio.csv', index=True)

# %%
