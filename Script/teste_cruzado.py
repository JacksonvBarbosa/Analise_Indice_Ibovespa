# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint, uniform
from sklearn import tree
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

# %%
# Carregamento
input_path = '../data/processed/arquivo-modelo.csv'
df_model = pd.read_csv(input_path)

# %%
# Features
best_features1 = ['close_lag3', 'close_lag5', 'close_lag7', 'delta', 'ema_10', 'ema_cross', 'macd', 'macd_hist',
                  'macd_signal', 'momentum_aceel', 'return_2d', 'return_3d', 'return_5d', 'rsi_14', 'rsi_9', 'stoch_d',
                  'vol_10d', 'vol_3d', 'volatilidade', 'volatilidade_relativa']
target = 'target'

X, y = df_model[best_features1], df_model[target]
X_train, X_test = X.iloc[:-30], X.iloc[-30:]
y_train, y_test = y.iloc[:-30], y.iloc[-30:]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# %%
# Estatísticas simples
print('Taxa variável resposta geral:', y.mean())
print('Taxa variável resposta treino:', y_train.mean())
print('Taxa variável resposta teste:', y_test.mean())

# %%
# Objeto de validação cruzada temporal
tscv = TimeSeriesSplit(n_splits=5)

# %%
# Funções auxiliares

def construir_pipeline(nome_modelo, modelo):
    return Pipeline([
        ('scaler', StandardScaler()),
        (nome_modelo, modelo)
    ])

def executar_random_search(pipeline, param_grid, X_train, y_train, cv, n_iter=10, scoring='f1_weighted'):
    busca = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    busca.fit(X_train, y_train)
    return busca

def avaliar_cross_validation(modelo, X_train, y_train, cv, scoring='f1_weighted'):
    scores = cross_val_score(modelo.best_estimator_, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    print(f"\nValidação Cruzada ({scoring}):")
    print(f"Média: {scores.mean():.4f} | Desvio Padrão: {scores.std():.4f}\n")

def avaliar_modelo(modelo_treinado, X_test, y_test):
    y_pred = modelo_treinado.predict(X_test)
    print("Melhores parâmetros:", modelo_treinado.best_params_)
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

    conf_mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.show()

# %%
# XGBoost
pipeline_xgb = construir_pipeline('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))

param_grid_xgb = {
    'xgb__n_estimators': randint(100, 150),
    'xgb__max_depth': randint(2, 5),
    'xgb__learning_rate': uniform(0.03, 0.09),
    'xgb__subsample': uniform(0.5, 0.5),
    'xgb__colsample_bytree': uniform(0.5, 0.5),
    'xgb__colsample_bylevel': uniform(0.5, 0.5),
    'xgb__reg_alpha': uniform(0.1, 0.9),
    'xgb__reg_lambda': uniform(0.5, 0.9),
    'xgb__min_child_weight': randint(3, 5),
    'xgb__gamma': uniform(0.1, 2),
}

modelo_xgb = executar_random_search(pipeline_xgb, param_grid_xgb, X_train, y_train, tscv)
avaliar_cross_validation(modelo_xgb, X_train, y_train, tscv)
avaliar_modelo(modelo_xgb, X_test, y_test)

# %%
# CatBoost
pipeline_cat = construir_pipeline('cat', CatBoostClassifier(verbose=0))

param_grid_cat = {
    'cat__iterations': randint(300, 400),
    'cat__learning_rate': uniform(0.01, 0.10),
    'cat__depth': randint(4, 7),
    'cat__l2_leaf_reg': uniform(1, 4),
    'cat__subsample': uniform(0.7, 0.3),
    'cat__random_strength': uniform(0.5, 1.5),
    'cat__bagging_temperature': uniform(0.5, 0.5),
    'cat__border_count': [64, 128, 254],
}

modelo_cat = executar_random_search(pipeline_cat, param_grid_cat, X_train, y_train, tscv)
avaliar_cross_validation(modelo_cat, X_train, y_train, tscv)
avaliar_modelo(modelo_cat, X_test, y_test)

# %%
# RandomForest
pipeline_rfc = construir_pipeline('rfc', RandomForestClassifier(random_state=42, n_jobs=-1))

param_grid_rfc = {
    'rfc__n_estimators': randint(100, 150),
    'rfc__max_depth': randint(5, 10),
    'rfc__min_samples_split': randint(2, 10),
    'rfc__min_samples_leaf': randint(1, 10),
    'rfc__class_weight': ['balanced', None]
}

modelo_rfc = executar_random_search(pipeline_rfc, param_grid_rfc, X_train, y_train, tscv)
avaliar_cross_validation(modelo_rfc, X_train, y_train, tscv)
avaliar_modelo(modelo_rfc, X_test, y_test)
