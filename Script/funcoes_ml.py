# %%
# FUNÇÕES E PIPELINES

# Pipelines dos Modelos
def construir_pipeline(nome_modelo, modelo):
    # modulo
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ('scaler', StandardScaler()),
        (nome_modelo, modelo)
    ])

# Função executa validação cruzada com RandomizedSearchCV
def executar_random_search(pipeline, param_grid, X_train, y_train, cv, n_iter=10, scoring='f1_weighted'):
    # modulo 
    from sklearn.model_selection import RandomizedSearchCV

    busca = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        verbose=5,
        random_state=42,
        n_jobs=-1
    )
    busca.fit(X_train, y_train)
    return busca
# Função executa validação cruzada com croo_val_score 
def avaliar_cross_validation(modelo, X_train, y_train, cv, scoring='f1_weighted'):
    # modulos
    from sklearn.model_selection import cross_val_score
    
    scores = cross_val_score(modelo.best_estimator_, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    print(f"\nValidação Cruzada ({scoring}):")
    print(f"Média: {scores.mean():.4f} | Desvio Padrão: {scores.std():.4f}\n")

# Função de avaliação dos modelos
def avaliar_modelo(modelo_treinado, X_test, y_test):
    # modulo
    from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

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

# Função estatística msnemar
def stat_msnemar(modelo1, modelo2, X_test, y_test):
    import numpy as np
    from statsmodels.stats.contingency_tables import mcnemar

    y_pred1 = modelo1.predict(X_test)
    y_pred2 = modelo2.predict(X_test)
    b = 0 # XGB acertou, RFC errou
    c = 0 # XGB errou, RFC acertou 

    for yt, px, pr in zip(y_test, y_pred1, y_pred2):
        if px == yt and pr != yt:
            b += 1
        elif px != yt and pr == yt:
            c += 1

    contingencia = np.array([[0, b],
                            [c, 0]])
    print("Tabela de contigência que o Mcnemar exige:")
    print(contingencia)
    resultado = mcnemar(contingencia, exact=False, correction=True)
    print(f'Estatística: {resultado.statistic}')
    print(f'p-valor: {resultado.pvalue:.4f}')

    if resultado.pvalue < 0.05:
        print("💥 Existe diferença estatisticamente significativa entre os modelos.")
    else:
        print("✅ Não há diferença estatisticamente significativa entre os modelos.")

# %%
