# %%
# FUNÇÕES DE PROCESSAMENTO DE DADOS
import pandas as pd

# Função de co-variáveis para o DataFrame dos modelos
def cria_variaveis_proximo_dia(df_input, df):
    # modulos
    import numpy as np
    from finta import TA

    # Diferença de fechamento e variações desconsiderando 0.5%
    # Variáveis Returns
    df_input['close'] = df['close'].copy()

    delta = df_input['close'].diff().copy() # Tira a diferença de um dia para o outro (d1 - d2)
    threshold = 0.005 # desconsidera variações menos que 0.5%
    df_input['target'] = np.where(
        delta > threshold, 1, np.where(delta < -threshold, 0, np.nan)
    )

    df_input['delta'] = delta.shift(1)
    df_input['return'] = df_input['close'].pct_change().shift(1)
    df_input['return_1d'] = df_input['close'].pct_change(1).shift()
    df_input['return_2d'] = df_input['close'].pct_change(2).shift()
    df_input['return_3d'] = df_input['close'].pct_change(3).shift()
    df_input['return_5d'] = df_input['close'].pct_change(5).shift()
    
    # Aceleração de momentum
    df_input['momentum_aceel'] = df_input['return_1d'] - df_input['return_2d']
    df_input['momentum_consistency'] = (
        (df_input['return_1d'] > 0) &
        (df_input['return_2d'] > 0) &
        (df_input['return_3d'] > 0)
    ).astype(int)
    
    # Volatilidade recente vs histórica
    df_input['vol_3d'] = df_input['return_1d'].rolling(3).std()
    df_input['vol_10d'] = df_input['return_1d'].rolling(10).std()
    df_input['vol_regime'] = df_input['vol_3d'] / df_input['vol_10d']

    # open, high, low
    df_input['open'] = df['open'].copy()
    df_input['high'] = df['high'].shift(1).copy()
    df_input['low'] = df['low'].shift(1).copy()
    
    # RSI
    df_input['rsi_9'] = TA.RSI(df_input, period=9).shift(1)
    df_input['rsi_14'] = TA.RSI(df_input, period=14).shift(1)
    df_input['rsi_divergence'] = df_input['rsi_14'] - df_input['rsi_9']
    
    # MACD rápido
    try:
        # Tenta passar o DataFrame inteiro primeiro
        macd_df = TA.MACD(df_input, column='close').shift(1)
        df_input['macd'] = macd_df['MACD']
        df_input['macd_signal'] = macd_df['SIGNAL']
        df_input['macd_hist'] = df_input['macd'] - df_input['macd_signal']
    except:
        # Retorno: calculo manual do MACD
        ema_12 = df_input['close'].ewm(span=12).mean().shift(1)
        ema_26 = df_input['close'].ewm(span=26).mean().shift(1)
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        
        df_input['macd'] = macd
        df_input['macd_signal'] = macd_signal
        df_input['macd_hist'] = macd - macd_signal
    # Stochastic rápido
    low_min = df_input['low'].copy().rolling(window=5).min()
    high_max = df_input['high'].copy().rolling(window=5).max()
    df_input['stoch_k'] = 100 * (df_input['close'] - low_min) / (high_max - low_min)
    df_input['stoch_d'] = df_input['stoch_k'].rolling(window=3).mean()
    # Colunas criada algumas com defasagem de uma dia
    df_input['volatilidade'] = df_input['high'].copy() - df_input['low'].copy()
    df_input['volatilidade_relativa'] = df_input['high'].copy() / df_input['low'].copy()
    # MÉDIAS MÓVEIS E TENDÊNCIA (Peso: 10%)
    df_input['ema_5'] = df_input['close'].ewm(span=5).mean()
    df_input['ema_10'] = df_input['close'].ewm(span=10).mean()
    df_input['ema_cross'] = (df_input['ema_5'] > df_input['ema_10']).astype(int)

    return df_input

# Função converter volume de string para float
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
    
# Função lags de Series
def make_lags(series: pd.Series, n_lags):
    return series.shift(n_lags)

# Função cria colunas defasadas
def make_n_lags(df, n_lags, column, step):
    for i in range(1, n_lags + 1, step):
        df[f"{column}_lag{i}"] = df[column].shift(i)
    return df
# %%
