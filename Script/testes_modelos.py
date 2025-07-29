# %%
import pandas as pd

# %%

input_path = '../data/processed/arquivo-modelo.csv'

df_model = pd.read_csv(input_path)

# %%

# Caça sistemática ao vazamento de dados

def hunt_data_leakage(df_model, target='target'):
    """
    Identifica sistematicamente quais features estão vazando dados
    """
    
    print("🕵️ CAÇA AO VAZAMENTO DE DADOS")
    print("="*50)
    
    # Excluir target das features
    features = [col for col in df_model.columns if col != target]
    
    # 1. ANÁLISE DE CORRELAÇÕES SUSPEITAS
    print(f"\n1️⃣ CORRELAÇÕES MAIS SUSPEITAS:")
    correlations = df_model[features].corrwith(df_model[target]).abs().sort_values(ascending=False)
    
    print("Top 15 correlações com target:")
    for i, (feature, corr) in enumerate(correlations.head(15).items()):
        status = "🚨" if corr > 0.5 else "⚠️" if corr > 0.3 else "📊"
        print(f"   {i+1:2d}. {feature:<25} {corr:.3f} {status}")
    
    # Features muito suspeitas
    very_suspicious = correlations[correlations > 0.5]
    if len(very_suspicious) > 0:
        print(f"\n🚨 FEATURES ALTAMENTE SUSPEITAS (>0.5):")
        for feature, corr in very_suspicious.items():
            print(f"   {feature}: {corr:.3f}")
    
    return correlations

correlacao = hunt_data_leakage(df_model)
hunt_data_leakage(df_model)
# %%
def test_feature_groups(df_model, target='target'):
    """
    Testa grupos de features para identificar qual grupo está vazando
    """
    
    print(f"\n2️⃣ TESTE POR GRUPOS DE FEATURES:")
    print("-" * 40)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    
    # Definir grupos de features
    feature_groups = {
        'returns': ['return', 'return_1d', 'return_2d', 'return_3d', 'return_5d', 'delta'],
        'momentum': ['momentum_aceel', 'momentum_consistency'],
        'volatility': ['vol_3d', 'vol_10d', 'vol_regime', 'volatilidade', 'volatilidade_relativa'],
        'ohlc': ['open', 'high', 'low'],
        'technical': ['rsi_9', 'rsi_14', 'rsi_divergence', 'macd', 'macd_signal', 'macd_hist'],
        'stochastic': ['stoch_k', 'stoch_d'],
        'moving_avg': ['ema_5', 'ema_10', 'price_vs_ema5', 'ema_cross'],
        'lags': ['close_lag1', 'close_lag3', 'close_lag5', 'close_lag7']
    }
    
    # Separação dos dados
    y = df_model[target].dropna()
    y_train, y_test = y.iloc[:-30], y.iloc[-30:]
    
    results = {}
    
    for group_name, group_features in feature_groups.items():
        # Verificar se features existem
        available_features = [f for f in group_features if f in df_model.columns]
        if not available_features:
            continue
            
        print(f"\n📊 Testando grupo '{group_name}': {available_features}")
        
        try:
            X = df_model[available_features].loc[y.index].dropna()
            if len(X) < 50:  # Dados insuficientes
                print(f"   ⚠️ Dados insuficientes ({len(X)} amostras)")
                continue
                
            # Alinhar y com X
            y_aligned = y.loc[X.index]
            X_train = X.iloc[:-30]
            X_test = X.iloc[-30:]
            y_train_aligned = y_aligned.iloc[:-30]
            y_test_aligned = y_aligned.iloc[-30:]
            
            # Treinar modelo
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train_aligned)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test_aligned, y_pred)
            f1 = f1_score(y_test_aligned, y_pred, average='weighted')
            
            results[group_name] = {'accuracy': acc, 'f1': f1, 'features': available_features}
            
            status = "🚨 SUSPEITO!" if acc > 0.7 or f1 > 0.7 else "✅ Normal"
            print(f"   Accuracy: {acc:.3f} | F1: {f1:.3f} {status}")
            
        except Exception as e:
            print(f"   ❌ Erro: {str(e)}")
    
    # Ranking dos grupos mais suspeitos
    print(f"\n🏆 RANKING DOS GRUPOS MAIS SUSPEITOS:")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for i, (group, metrics) in enumerate(sorted_results):
        acc = metrics['accuracy']
        f1 = metrics['f1']
        status = "🚨" if acc > 0.7 else "⚠️" if acc > 0.6 else "✅"
        print(f"   {i+1}. {group:<15} Acc:{acc:.3f} F1:{f1:.3f} {status}")
    
    return results
test_feature_groups(df_model)

# %%
def test_individual_suspicious_features(df_model, correlations, target='target', threshold=0.3):
    """
    Testa individualmente as features mais suspeitas
    """
    
    print(f"\n3️⃣ TESTE INDIVIDUAL DAS FEATURES SUSPEITAS:")
    print("-" * 50)
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    
    suspicious_features = correlations[correlations > threshold].index.tolist()
    
    if not suspicious_features:
        print("✅ Nenhuma feature com correlação suspeita encontrada")
        return
    
    print(f"Testando {len(suspicious_features)} features com correlação > {threshold}")
    
    y = df_model[target].dropna()
    individual_results = {}
    
    for feature in suspicious_features[:10]:  # Top 10
        if feature not in df_model.columns:
            continue
            
        try:
            X = df_model[[feature]].loc[y.index].dropna()
            y_aligned = y.loc[X.index]
            
            X_train = X.iloc[:-30]
            X_test = X.iloc[-30:]
            y_train_aligned = y_aligned.iloc[:-30]
            y_test_aligned = y_aligned.iloc[-30:]
            
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train_aligned)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test_aligned, y_pred)
            f1 = f1_score(y_test_aligned, y_pred, average='weighted')
            
            individual_results[feature] = {'accuracy': acc, 'f1': f1}
            
            status = "🚨 VAZAMENTO!" if acc > 0.8 else "⚠️ Suspeito" if acc > 0.65 else "✅ Ok"
            print(f"   {feature:<25} Acc:{acc:.3f} F1:{f1:.3f} {status}")
            
        except Exception as e:
            print(f"   {feature:<25} ❌ Erro: {str(e)}")
    
    return individual_results

test_individual_suspicious_features(df_model, correlacao)