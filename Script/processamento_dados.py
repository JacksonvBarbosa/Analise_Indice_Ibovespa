# %%

import pandas as pd
import yfinance as yf
import numpy as np
#import pandas_ta as ta
#import talib as ta
from finta import TA

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# %%

input_path = '../data/raw/dados_historicos_ibovespa_2008-2025.csv'

df = pd.read_csv(input_path, thousands='.', decimal=',', parse_dates=['Data'], date_format='%d.%m.%Y', index_col='Data')
df = df.rename_axis('ds').sort_index()
df.tail()

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
def converter_volume(vol: str | float) -> float:
    """
    Converte uma string de volume com sufixos (K, M, B) para um número float.
    
    Parâmetro:
        vol (string | float): o valor a ser convertido (ex: '8,3M'). Pode ser uma string ou um np.nan (que é float).
        
    Retorna:
        float: o valor convertido ou np.nan caso não haja um valor.
    """
    if not isinstance(vol, str):
        return vol

    multiplicadores = {'K': 1e3, 'M': 1e6, 'B': 1e9}
    vol = vol.upper().replace(',', '.').strip()
    sufixo = vol[-1]

    if sufixo in multiplicadores:
        return float(vol[:-1]) * multiplicadores[sufixo]
    else:
        return float(vol)

df['volume'] = df['volume'].apply(converter_volume)
# %%
# substituir o volume nulo pela média do volume anterior e posterior daquela data
df['volume'] = df['volume'].interpolate()
# %%
# ajustando a coluna variação percentual diária, que contém o pct_change() do fechamento
df['daily_return'] = df['daily_return'].str.replace('%', '').str.replace(',', '.')
df['daily_return'] = round(df['daily_return'].astype(float) / 100, 4)
df.head()
# %%
# Cria um Data Frame limpo
df_prep = pd.DataFrame()

# %%
# Delta(diferença d1 - d2) e Return (variação diaria) adiantado um 1
# Momentum recente

def create_next_day_features(df_prep, df):
    # Diferença de fechamento e variações desconsiderando 0.5%
    # Variáveis Returns
    
    df_prep['close'] = df['close'].copy()

    delta = df_prep['close'].diff() # Tira a diferença de um dia para o outro (d1 - d2)
    threshold = 0.005 # desconsidera variações menos que 0.5%
    df_prep['target'] = np.where(
        delta > threshold, 1, np.where(delta < -threshold, 0, np.nan)
    )

    df_prep['delta'] = delta.shift(1)
    df_prep['return'] = df_prep['close'].pct_change().shift(1)
    df_prep['return_1d'] = df_prep['close'].pct_change(1).shift()
    df_prep['return_2d'] = df_prep['close'].pct_change(2).shift()
    df_prep['return_3d'] = df_prep['close'].pct_change(3).shift()
    df_prep['return_5d'] = df_prep['close'].pct_change(5).shift()
    
    # Aceleração de momentum
    df_prep['momentum_aceel'] = df_prep['return_1d'] - df_prep['return_2d']
    df_prep['momentum_consistency'] = (
        (df_prep['return_1d'] > 0) &
        (df_prep['return_2d'] > 0) &
        (df_prep['return_3d'] > 0)
    ).astype(int)
    
    # Volatilidade recente vs histórica
    df_prep['vol_3d'] = df_prep['return_1d'].rolling(3).std()
    df_prep['vol_10d'] = df_prep['return_1d'].rolling(10).std()
    df_prep['vol_regime'] = df_prep['vol_3d'] / df_prep['vol_10d']

    # open, high, low
    df_prep['open'] = df['open'].copy()
    df_prep['high'] = df['high'].shift(1).copy()
    df_prep['low'] = df['low'].shift(1).copy()
    
    # RSI
    df_prep['rsi_9'] = TA.RSI(df_prep, period=9)
    df_prep['rsi_14'] = TA.RSI(df_prep, period=14)
    df_prep['rsi_divergence'] = df_prep['rsi_14'] - df_prep['rsi_9']
    
    # MACD rápido
    try:
        # Tenta passar o DataFrame inteiro primeiro
        macd_df = TA.MACD(df_prep, column='close')
        df_prep['macd'] = macd_df['MACD']
        df_prep['macd_signal'] = macd_df['SIGNAL']
        df_prep['macd_hist'] = df_prep['macd'] - df_prep['macd_signal']
    except:
        # Retorno: calculo manual do MACD
        ema_12 = df_prep['close'].ewm(span=12).mean()
        ema_26 = df_prep['close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        
        df_prep['macd'] = macd
        df_prep['macd_signal'] = macd_signal
        df_prep['macd_hist'] = macd - macd_signal
    # Stochastic rápido
    low_min = df_prep['low'].copy().rolling(window=5).min()
    high_max = df_prep['high'].copy().rolling(window=5).max()
    df_prep['stoch_k'] = 100 * (df_prep['close'] - low_min) / (high_max - low_min)
    df_prep['stoch_d'] = df_prep['stoch_k'].rolling(window=3).mean()
    # Colunas criada algumas com defasagem de uma dia
    df_prep['volatilidade'] = df_prep['high'].copy() - df_prep['low'].copy()
    df_prep['volatilidade_relativa'] = df_prep['high'].copy() / df_prep['low'].copy()
    # MÉDIAS MÓVEIS E TENDÊNCIA (Peso: 10%)
    df_prep['ema_5'] = df_prep['close'].ewm(span=5).mean()
    df_prep['ema_10'] = df_prep['close'].ewm(span=10).mean()
    df_prep['ema_cross'] = (df_prep['ema_5'] > df_prep['ema_10']).astype(int)

    return df_prep


# %%
# Chama função
create_next_day_features(df_prep, df)
# Remove valores nulos
df_prep = df_prep.dropna(axis=0)
df_prep
# %%
# Função lags de Series
def make_lags(series: pd.Series, n_lags):
    return series.shift(n_lags)

# Função cria colunas defasadas
def make_n_lags(df, n_lags, column, step):
    for i in range(1, n_lags + 1, step):
        df[f"{column}_lag{i}"] = df[column].shift(i)
    return df
# %%
n_lags = 7

# Chama função cria colunas com lags
df_prep = make_n_lags(df_prep, n_lags, "target", 1)
df_prep = make_n_lags(df_prep, n_lags, "close", 2)

# Exclui tabela Close que não vai ser usada
df_prep.drop('close', axis=1, inplace=True)
# Data Frame pronto para o modelo doprando linhas com valores na
df_model = df_prep.dropna().copy()
df_model
# %%
df_model.isna().sum().sort_values()

df_model.to_csv('../data/processed/arquivo-modelo.csv', index=False)
