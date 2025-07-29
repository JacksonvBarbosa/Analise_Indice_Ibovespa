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
best_features = [
        'delta', 'return_1d', 'return_2d', 'return_3d',
        'return_5d', 'momentum_aceel', 'momentum_consistency', 'vol_3d',
        'vol_10d', 'open', 'high', 'low', 'rsi_9', 'rsi_14',
        'macd', 'macd_signal', 'macd_hist', 'rsi_divergence',
        'stoch_d', 'volatilidade', 'volatilidade_relativa', 'ema_5', 'ema_10',
        'ema_cross',
        
        'target_lag7', 'close_lag1', 'close_lag3', 'close_lag5', 'close_lag7'
    ]

features = df_model.columns[1:]
target = 'target'

X, y = df_model[best_features], df_model[target]

display(X)

display(y)
# Separação dos dados
X_train, X_test = X.iloc[:-30], X.iloc[-30:]
y_train, y_test = y.iloc[:-30], y.iloc[-30:]

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# %%
# Utilizando o train_test_split
''' X, y = df_model[features], df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False # importante para manter ordem temporal
)'''
# %%

print('Taxa variável resposta geral',y.mean())
print('Taxa variável resposta treino',y_train.mean())
print('Taxa variável resposta teste',y_test.mean())


# %%

tscv = TimeSeriesSplit(n_splits=5)
# %%
pipe_xgb = Pipeline(
    [
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier()),
    ]
)


param_grid_xgb = {
    'xgb__n_estimators': randint(100, 150),              # Menos árvores
    'xgb__max_depth': randint(2, 5),                    # Árvores mais rasas
    'xgb__learning_rate': uniform(0.03, 0.09),          # Learning rate menor
    'xgb__subsample': uniform(0.5, 0.9),                # Entre 0.5 e 0.9
    'xgb__colsample_bytree': uniform(0.5, 0.9),         # Entre 0.5 e 0.9
    'xgb__colsample_bylevel': uniform(0.5, 0.9),        # Entre 0.5 e 0.9
    'xgb__reg_alpha': uniform(0.1, 0.9),                  # Regularização L1 maior
    'xgb__reg_lambda': uniform(0.5, 0.9),                 # Regularização L2 maior
    'xgb__min_child_weight': randint(2, 5),            # Peso mínimo maior
    'xgb__gamma': uniform(0.1, 2),                      # Poda mais agressiva
}                 # Entre 0 e 2


#grid_xgb = GridSearchCV(pipe_xgb, param_grid_xgb, cv=tscv)
grid_xgb = RandomizedSearchCV(pipe_xgb, param_grid_xgb, cv=tscv) # Ele tira mostras aleatórias dos dados ai ele não passa por todos os estados
grid_xgb.fit(X_train, y_train)


# %%

print(grid_xgb.best_params_)
print(grid_xgb.feature_names_in_)
print(grid_xgb.best_score_)
# %%

print("Logistic Regression:")
print("Melhores parâmetros:", grid_xgb.best_params_) # Salva o melhor modelo nesse best_params_
print("Acurácia:", accuracy_score(y_test, grid_xgb.predict(X_test)))
print(classification_report(y_test, grid_xgb.predict(X_test)))
# %%

pipe_cat = Pipeline(
    [
        ('scaler', StandardScaler()),
        ('cat', CatBoostClassifier()),
    ]
)

param_grid_cat = {
    'cat__iterations': randint(300, 1200),              # Entre 300 e 1199
    'cat__learning_rate': uniform(0.01, 0.19),          # Entre 0.01 e 0.2
    'cat__depth': randint(3, 8),                        # Entre 3 e 7
    'cat__l2_leaf_reg': uniform(0.1, 9.9),             # Entre 0.1 e 10.0
    'cat__subsample': uniform(0.6, 0.4),               # Entre 0.6 e 1.0
    'cat__random_strength': uniform(0, 2),              # Entre 0 e 2 (para regularização adicional)
    'cat__bagging_temperature': uniform(0, 1),          # Entre 0 e 1
    'cat__border_count': randint(32, 256),              # Entre 32 e 255 (discretização)
}

#grid_xgb = GridSearchCV(pipe_xgb, param_grid_xgb, cv=tscv)
grid_cat = RandomizedSearchCV(pipe_cat, param_grid_cat, cv=tscv) # Ele tira mostras aleatórias dos dados ai ele não passa por todos os estados
grid_cat.fit(X_train, y_train)

# %%

print("Logistic Regression:")
print("Melhores parâmetros:", grid_cat.best_params_) # Salva o melhor modelo nesse best_params_
print("Acurácia:", accuracy_score(y_test, grid_cat.predict(X_test)))
print(classification_report(y_test, grid_cat.predict(X_test)))

# %%

pipe_forest = Pipeline([
    ('scaler', StandardScaler()),  # pode ser omitido no RandomForest, mas ajuda em outros modelos
    ('model', RandomForestClassifier(
        n_estimators=200,       # número de árvores
        max_depth=10,           # profundidade controlada (evita overfitting)
        min_samples_split=10,   # exige mais exemplos para split
        min_samples_leaf=5,     # folhas menos sensíveis a ruídos
        random_state=42,
        class_weight='balanced' # ajuda com desbalanceamento entre classes 0 e 1
    ))
])

# Treinar modelo
pipe_forest.fit(X_train, y_train)
y_pred = pipe_forest.predict(X_test)

# %%

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

print("Matriz de Confusão:")
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

# %%

# %% 
# Importância das Features
model = pipe_forest.named_steps['model']
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(20).plot(kind='barh', figsize=(10, 8))
plt.title('Top 20 Features mais importantes')
plt.xlabel('Importância')
plt.gca().invert_yaxis()
plt.show()

# %%
# Associação de cada features com suas importaâncias
features_importance = (pd.Series(model.feature_importances_,
                                 index=X_train.columns)
                                 .sort_values(ascending=False).
                                 reset_index()
                                 )

# Uma estratégia (Mais tem várias que possa ser utilizada) é utilizar a acumulação da porcentagem da importância
# E por exemplos selecionar as que vão até 95% 
features_importance['acum.'] = features_importance[0].cumsum() # Acumular com o cumsum()
features_importance[features_importance['acum.'] < 0.96].sort_values(by='acum.',ascending=False)
