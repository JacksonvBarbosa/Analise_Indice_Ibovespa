# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint, uniform
from sklearn import tree
# Separa dados para uma respostamais confiavel e não embara os dados
from sklearn.model_selection import TimeSeriesSplit
# importe do Pipeline de dados e dos modelos de teste
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

# %%

input_path = '../data/processed/arquivo-modelo.csv'

df_model = pd.read_csv(input_path)
df_model
# %%

# melhores co-varáveis
best_features = ['delta', 'return_1d', 'return_2d', 'return_3d', 'return_5d', 'momentum_aceel',  'close_lag7',
                'momentum_consistency', 'vol_3d', 'vol_10d', 'open', 'high', 'low', 'rsi_9', 'rsi_14',
                'macd', 'macd_signal', 'macd_hist', 'stoch_d', 'volatilidade', 'volatilidade_relativa',
                'ema_5', 'ema_10', 'ema_cross', 'target_lag7', 'close_lag1', 'close_lag3', 'close_lag5'
    ]

best_features1 = ['close_lag3', 'close_lag5', 'close_lag7', 'delta', 'ema_10', 'ema_cross', 'macd', 'macd_hist',
                'macd_signal', 'momentum_aceel', 'return_2d', 'return_3d', 'return_5d', 'rsi_14', 'rsi_9', 'stoch_d',
                'vol_10d', 'vol_3d', 'volatilidade', 'volatilidade_relativa'
]

features = df_model.columns[1:]
# Variáveis resposta
target = 'target'

# Separação das features
X, y = df_model[best_features1], df_model[target]

# Separação de treino e teste
X_train, X_test = X.iloc[:-30], X.iloc[-30:]
y_train, y_test = y.iloc[:-30], y.iloc[-30:]

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# %%
# Média das variáveis de treino e teste
print('Taxa variável resposta geral',y.mean())
print('Taxa variável resposta treino',y_train.mean())
print('Taxa variável resposta teste',y_test.mean())


# %%
# Objeto para válidação cruzada de series temporais
tscv = TimeSeriesSplit(n_splits=5)

# %%

# FUNÇÕES E PIPELINES

# Pipelines dos Modelos
def construir_pipeline(nome_modelo, modelo):
    return Pipeline([
        ('scaler', StandardScaler()),
        (nome_modelo, modelo)
    ])

# Função executa o RandomizedSearchCV
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

# Função de avaliação dos modelos
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
# EXECUTANDO FUNÇÕES

# XGBoost
pipeline_xgb = construir_pipeline('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))

# parametros para o XGBOOST
param_grid_xgb = {
    'xgb__n_estimators': randint(100, 150),             # Menos árvores
    'xgb__max_depth': randint(2, 5),                    # Árvores mais rasas
    'xgb__learning_rate': uniform(0.03, 0.09),          # Learning rate menor
    'xgb__subsample': uniform(0.5, 0.5),                # Entre 0.5 e 0.9
    'xgb__colsample_bytree': uniform(0.5, 0.5),         # Entre 0.5 e 0.9
    'xgb__colsample_bylevel': uniform(0.5, 0.5),        # Entre 0.5 e 0.9
    'xgb__reg_alpha': uniform(0.1, 0.9),                # Regularização L1 maior
    'xgb__reg_lambda': uniform(0.5, 0.9),               # Regularização L2 maior
    'xgb__min_child_weight': randint(3, 5),             # Peso mínimo maior
    'xgb__gamma': uniform(0.1, 2),                      # Poda mais agressiva
}

#grid_xgb = GridSearchCV(pipe_xgb, param_grid_xgb, cv=tscv)
modelo_xgb = executar_random_search(pipeline_xgb, param_grid_xgb, X_train, y_train, tscv)
avaliar_modelo(modelo_xgb, X_test, y_test)

# %%
# CatBoost
pipeline_cat = construir_pipeline('cat', CatBoostClassifier(verbose=0))

# parametros para o CATBOOST
param_grid_cat = {
    'cat__iterations': randint(300, 400),               # Menos iterações
    'cat__learning_rate': uniform(0.01, 0.10),          # [0.05, 0.15] - range menor
    'cat__depth': randint(4, 7),                        # [4, 5, 6] - depths moderados
    'cat__l2_leaf_reg': uniform(1, 4),                  # [1, 5] - regularização moderada0
    'cat__subsample': uniform(0.7, 0.3),                # [0.7, 1.0] - mais conservador
    'cat__random_strength': uniform(0.5, 1.5),          # [0.5, 2.0] - evita 0
    'cat__bagging_temperature': uniform(0.5, 0.5),      # [0.5, 1.0] - evita 0
    'cat__border_count': [64, 128, 254],                # Valores fixos seguros
}

#grid_xgb = GridSearchCV(pipe_xgb, param_grid_xgb, cv=tscv)
modelo_cat = executar_random_search(pipeline_cat, param_grid_cat, X_train, y_train, tscv)
avaliar_modelo(modelo_cat, X_test, y_test)

# %%
# RandomForest
pipeline_rfc = construir_pipeline('rfc', RandomForestClassifier(random_state=42, n_jobs=-1))

# parametros para o RandomForest
param_grid_rfc = {
    'rfc__n_estimators': randint(100, 150),         # Número de árvores
    'rfc__max_depth': randint(5, 10),               # Profundidade máxima
    'rfc__min_samples_split': randint(2, 10),       # Mínimo de amostras para split
    'rfc__min_samples_leaf': randint(1, 10),        # Mínimo de amostras na folha
    'rfc__class_weight': ['balanced', None]         # Tratar desbalanceamento
    # 'rcf__max_features': uniform(0.5, 1.0)        # Exemplo de parâmetro float
}

modelo_rfc = executar_random_search(pipeline_rfc, param_grid_rfc, X_train, y_train, tscv)
avaliar_modelo(modelo_rfc, X_test, y_test)

# %%
# Associação de cada features com suas importaâncias
features_importance = (pd.Series(pipeline_xgb.best_estimator_,
                                 index=X_train.columns)
                                 .sort_values(ascending=False).
                                 reset_index()
                                 )

# Uma estratégia (Mais tem várias que possa ser utilizada) é utilizar a acumulação da porcentagem da importância
# E por exemplos selecionar as que vão até 95% 
features_importance['acum.'] = features_importance[0].cumsum() # Acumular com o cumsum()
features_importance[features_importance['acum.'] < 0.96].sort_values(by='acum.',ascending=False)

# %%

best_features1 = features_importance[features_importance['acum.'] < 0.96]['index'].sort_values().to_list()
best_features1
