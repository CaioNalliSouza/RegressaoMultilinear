"""
Módulo para treinamento e gestão do modelo de Machine Learning.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def treinar_modelo(df):
    """
    Prepara os dados e treina o modelo de regressão linear múltipla para previsão de faturamento.
    Inclui engenharia de features para maximizar R².
    
    Args:
        df (pd.DataFrame): DataFrame com dados históricos
    
    Returns:
        tuple: (modelo, feature_names, metricas, X_train, X_test, y_train, y_test)
    """
    
    # Criar variável de faturamento do mês anterior
    df_modelo = df.copy()
    df_modelo['Faturamento_Mes_Ant'] = df_modelo['Faturamento'].shift(1)
    df_modelo['Sinistralidade_Mes_Ant'] = df_modelo['Sinistralidade_Realizada'].shift(1)
    
    # Remover primeira linha (sem mês anterior)
    df_modelo = df_modelo.dropna()
    
    # ==== ENGENHARIA DE FEATURES OTIMIZADA ====
    
    # Features de interação mais importantes (apenas as que têm sentido lógico forte)
    df_modelo['Volume_x_Ticket'] = df_modelo['Qtd_Atendimentos'] * df_modelo['Ticket_Medio']
    
    # Tendência temporal simples
    df_modelo['Tendencia'] = np.arange(len(df_modelo))
    
    # Selecionar features principais
    features_base = [
        'Faturamento_Mes_Ant', 'Qtd_Atendimentos', 'Ticket_Medio', 
        'Perc_Atend_Com_Pecas', 'Tempo_Medio_Atend_Horas', 
        'Taxa_Reincidencia', 'NPS', 'Sinistralidade_Mes_Ant',
        'Taxa_Juros', 'Indice_Acidentes', 'Tendencia'
    ]
    
    # Feature engenheirada crítica
    features_engenheiradas = ['Volume_x_Ticket']
    
    all_features = features_base + features_engenheiradas
    
    # Criar variáveis dummy para os meses (sazonalidade adicional)
    df_modelo_dummies = pd.get_dummies(df_modelo, columns=['Mes'], prefix='Mes', drop_first=True)
    mes_features = [col for col in df_modelo_dummies.columns if col.startswith('Mes_')]
    all_features.extend(mes_features)
    
    # Separar features e target
    X = df_modelo_dummies[all_features]
    y = df_modelo_dummies['Faturamento']
    
    # Dividir em treino e teste (80% treino, 20% teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # Treinar modelo
    modelo = LinearRegression(fit_intercept=True, copy_X=True)
    modelo.fit(X_train, y_train)
    
    # Calcular métricas no conjunto de teste
    y_pred_test = modelo.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    # Calcular também métricas no treino
    y_pred_train = modelo.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)
    
    # Calcular MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    
    metricas = {
        'r2': r2,
        'r2_train': r2_train,
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }
    
    return modelo, all_features, metricas, X_train, X_test, y_train, y_test


def fazer_previsao(modelo, feature_names, inputs):
    """
    Realiza previsão de faturamento com base nos inputs fornecidos.
    Inclui cálculo de features engenheiradas.
    
    Args:
        modelo: Modelo treinado
        feature_names (list): Lista de nomes das features
        inputs (dict): Dicionário com valores das features
    
    Returns:
        float: Valor previsto de faturamento
    """
    
    # Preparar o input para o modelo com todas as features em zero
    X_pred = pd.DataFrame(0.0, index=[0], columns=feature_names)
    
    # Preencher as variáveis diretas
    for key, value in inputs.items():
        if key in X_pred.columns:
            X_pred[key] = value
    
    # ==== CALCULAR FEATURES ENGENHEIRADAS ====
    
    # 1. Interação Volume x Ticket
    if 'Volume_x_Ticket' in X_pred.columns:
        X_pred['Volume_x_Ticket'] = inputs.get('Qtd_Atendimentos', 0) * inputs.get('Ticket_Medio', 0)
    
    # 2. Tendência temporal (usar valor médio)
    if 'Tendencia' in X_pred.columns:
        X_pred['Tendencia'] = 24
    
    # Preencher a variável Dummy do Mês (sazonalidade categórica)
    mes_prev = inputs.get('mes_prev', 1)
    if mes_prev > 1:
        coluna_mes_dummy = f'Mes_{mes_prev}'
        if coluna_mes_dummy in X_pred.columns:
            X_pred[coluna_mes_dummy] = 1
    
    # Realizar Previsão
    predicao = modelo.predict(X_pred)[0]
    
    # Garantir que a previsão seja razoável
    predicao = np.clip(predicao, 100000, 2000000)
    
    return predicao
