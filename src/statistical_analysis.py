"""
Módulo de Análises Estatísticas Avançadas
Fornece análises profundas para suporte à decisão comercial
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any


def calcular_correlacoes(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Calcula matriz de correlação entre todas as variáveis
    """
    correlacoes = df[features].corr()
    return correlacoes


def identificar_correlacoes_fortes(df: pd.DataFrame, features: List[str], threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Identifica correlações fortes e retorna insights
    """
    correlacoes = df[features].corr()
    insights = []
    
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            corr_value = correlacoes.iloc[i, j]
            if abs(corr_value) >= threshold:
                var1 = features[i]
                var2 = features[j]
                
                # Teste de significância
                _, p_value = stats.pearsonr(df[var1], df[var2])
                
                insights.append({
                    'variavel_1': var1,
                    'variavel_2': var2,
                    'correlacao': corr_value,
                    'p_value': p_value,
                    'significancia': 'Alta' if p_value < 0.01 else 'Média' if p_value < 0.05 else 'Baixa',
                    'tipo': 'Positiva' if corr_value > 0 else 'Negativa',
                    'forca': 'Forte' if abs(corr_value) >= 0.7 else 'Moderada'
                })
    
    return sorted(insights, key=lambda x: abs(x['correlacao']), reverse=True)


def analise_tendencia_temporal(df: pd.DataFrame, coluna: str) -> Dict[str, Any]:
    """
    Analisa tendência temporal de uma variável usando regressão linear
    """
    x = np.arange(len(df))
    y = df[coluna].values
    
    # Regressão linear
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Tendência
    if p_value < 0.05:
        if slope > 0:
            tendencia = "Crescente"
            interpretacao = "aumentando"
        else:
            tendencia = "Decrescente"
            interpretacao = "diminuindo"
    else:
        tendencia = "Estável"
        interpretacao = "mantendo-se estável"
    
    # Variação percentual
    variacao_pct = ((y[-1] - y[0]) / y[0]) * 100 if y[0] != 0 else 0
    
    return {
        'tendencia': tendencia,
        'slope': slope,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'variacao_percentual': variacao_pct,
        'interpretacao': interpretacao,
        'significante': p_value < 0.05
    }


def calcular_intervalo_confianca(valores: np.ndarray, confianca: float = 0.95) -> Tuple[float, float, float]:
    """
    Calcula intervalo de confiança para uma série de valores
    """
    media = np.mean(valores)
    sem = stats.sem(valores)
    intervalo = stats.t.interval(confianca, len(valores) - 1, loc=media, scale=sem)
    
    return media, intervalo[0], intervalo[1]


def analise_distribuicao(valores: np.ndarray) -> Dict[str, Any]:
    """
    Analisa a distribuição estatística dos valores
    """
    # Estatísticas descritivas
    media = np.mean(valores)
    mediana = np.median(valores)
    desvio_padrao = np.std(valores)
    cv = (desvio_padrao / media * 100) if media != 0 else 0  # Coeficiente de variação
    
    # Teste de normalidade
    _, p_normalidade = stats.shapiro(valores) if len(valores) < 5000 else stats.kstest(valores, 'norm')
    
    # Quartis
    q1 = np.percentile(valores, 25)
    q3 = np.percentile(valores, 75)
    iqr = q3 - q1
    
    # Outliers
    limite_inf = q1 - 1.5 * iqr
    limite_sup = q3 + 1.5 * iqr
    outliers = valores[(valores < limite_inf) | (valores > limite_sup)]
    
    return {
        'media': media,
        'mediana': mediana,
        'desvio_padrao': desvio_padrao,
        'coeficiente_variacao': cv,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'minimo': np.min(valores),
        'maximo': np.max(valores),
        'num_outliers': len(outliers),
        'pct_outliers': (len(outliers) / len(valores)) * 100,
        'normal': p_normalidade > 0.05,
        'p_normalidade': p_normalidade
    }


def analise_comparativa_periodos(df: pd.DataFrame, coluna: str, n_meses_recentes: int = 6) -> Dict[str, Any]:
    """
    Compara período recente com período anterior
    """
    total_meses = len(df)
    
    # Dividir em dois períodos
    periodo_anterior = df[coluna].iloc[:total_meses - n_meses_recentes]
    periodo_recente = df[coluna].iloc[-n_meses_recentes:]
    
    # Estatísticas de cada período
    media_anterior = np.mean(periodo_anterior)
    media_recente = np.mean(periodo_recente)
    
    # Teste t para diferença de médias
    t_stat, p_value = stats.ttest_ind(periodo_anterior, periodo_recente)
    
    # Variação percentual
    variacao_pct = ((media_recente - media_anterior) / media_anterior * 100) if media_anterior != 0 else 0
    
    # Interpretação
    if p_value < 0.05:
        if media_recente > media_anterior:
            interpretacao = f"aumento significativo de {abs(variacao_pct):.1f}%"
            status = "melhora" if coluna in ['Faturamento', 'NPS', 'Qtd_Atendimentos'] else "piora"
        else:
            interpretacao = f"redução significativa de {abs(variacao_pct):.1f}%"
            status = "piora" if coluna in ['Faturamento', 'NPS', 'Qtd_Atendimentos'] else "melhora"
    else:
        interpretacao = "sem mudança significativa"
        status = "estável"
    
    return {
        'media_anterior': media_anterior,
        'media_recente': media_recente,
        'variacao_percentual': variacao_pct,
        'p_value': p_value,
        'significante': p_value < 0.05,
        'interpretacao': interpretacao,
        'status': status,
        't_statistic': t_stat
    }


def calcular_capacidade_processo(valores: np.ndarray, limite_inferior: float, limite_superior: float) -> Dict[str, Any]:
    """
    Calcula índices de capacidade do processo (Cp, Cpk)
    """
    media = np.mean(valores)
    desvio = np.std(valores, ddof=1)
    
    if desvio == 0:
        return {
            'cp': np.inf,
            'cpk': np.inf,
            'interpretacao': 'Processo sem variação'
        }
    
    # Cp - Capacidade potencial
    cp = (limite_superior - limite_inferior) / (6 * desvio)
    
    # Cpk - Capacidade real (considera centralização)
    cpu = (limite_superior - media) / (3 * desvio)
    cpl = (media - limite_inferior) / (3 * desvio)
    cpk = min(cpu, cpl)
    
    # Interpretação
    if cpk >= 1.33:
        interpretacao = "Processo capaz (excelente)"
        status = "✅ Ótimo"
    elif cpk >= 1.0:
        interpretacao = "Processo adequado (bom)"
        status = "✓ Bom"
    elif cpk >= 0.67:
        interpretacao = "Processo marginalmente capaz (atenção)"
        status = "⚠️ Atenção"
    else:
        interpretacao = "Processo incapaz (crítico)"
        status = "❌ Crítico"
    
    return {
        'cp': cp,
        'cpk': cpk,
        'cpu': cpu,
        'cpl': cpl,
        'interpretacao': interpretacao,
        'status': status,
        'dentro_limites': np.sum((valores >= limite_inferior) & (valores <= limite_superior)) / len(valores) * 100
    }


def gerar_insights_comerciais(df: pd.DataFrame, modelo, features: List[str]) -> List[Dict[str, str]]:
    """
    Gera insights comerciais baseados em análise estatística
    """
    insights = []
    
    # 1. Análise de Sinistralidade
    sin_stats = analise_distribuicao(df['Sinistralidade_Realizada'].values)
    if sin_stats['media'] > 50:
        insights.append({
            'tipo': '🚨 Alerta',
            'titulo': 'Sinistralidade acima da meta',
            'descricao': f"Média de {sin_stats['media']:.1f}% está {sin_stats['media'] - 50:.1f} pontos acima da meta de 50%",
            'impacto': 'Alto',
            'acao': 'Revisar processos operacionais e negociar com fornecedores'
        })
    else:
        insights.append({
            'tipo': '✅ Sucesso',
            'titulo': 'Sinistralidade controlada',
            'descricao': f"Média de {sin_stats['media']:.1f}% está dentro da meta",
            'impacto': 'Positivo',
            'acao': 'Manter estratégias atuais'
        })
    
    # 2. Análise de Tendência de Faturamento
    fat_tendencia = analise_tendencia_temporal(df, 'Faturamento')
    if fat_tendencia['tendencia'] == 'Crescente':
        insights.append({
            'tipo': '📈 Oportunidade',
            'titulo': 'Crescimento de Faturamento',
            'descricao': f"Crescimento de {fat_tendencia['variacao_percentual']:.1f}% no período",
            'impacto': 'Alto',
            'acao': 'Investir em expansão e captação de novos clientes'
        })
    elif fat_tendencia['tendencia'] == 'Decrescente':
        insights.append({
            'tipo': '⚠️ Atenção',
            'titulo': 'Queda no Faturamento',
            'descricao': f"Redução de {abs(fat_tendencia['variacao_percentual']):.1f}% no período",
            'impacto': 'Alto',
            'acao': 'Implementar estratégias de recuperação e retenção'
        })
    
    # 3. Análise de NPS
    nps_stats = analise_distribuicao(df['NPS'].values)
    if nps_stats['media'] < 70:
        insights.append({
            'tipo': '🎯 Melhoria',
            'titulo': 'Satisfação do Cliente',
            'descricao': f"NPS médio de {nps_stats['media']:.1f} abaixo da meta de 70",
            'impacto': 'Médio',
            'acao': 'Implementar programa de melhoria da experiência do cliente'
        })
    
    # 4. Análise de Eficiência Operacional
    tempo_stats = analise_distribuicao(df['Tempo_Medio_Atend_Horas'].values)
    if tempo_stats['media'] > 3.0:
        insights.append({
            'tipo': '⏱️ Eficiência',
            'titulo': 'Tempo de Atendimento',
            'descricao': f"Tempo médio de {tempo_stats['media']:.1f}h acima da meta de 3h",
            'impacto': 'Médio',
            'acao': 'Otimizar processos e aumentar capacidade operacional'
        })
    
    # 5. Análise de Ticket Médio
    ticket_tendencia = analise_tendencia_temporal(df, 'Ticket_Medio')
    if ticket_tendencia['tendencia'] == 'Crescente' and ticket_tendencia['variacao_percentual'] > 5:
        insights.append({
            'tipo': '💰 Receita',
            'titulo': 'Aumento do Ticket Médio',
            'descricao': f"Crescimento de {ticket_tendencia['variacao_percentual']:.1f}%",
            'impacto': 'Positivo',
            'acao': 'Analisar oportunidades de upselling e cross-selling'
        })
    
    return insights


def calcular_previsao_com_intervalo(modelo, features: List[str], inputs: Dict[str, float], 
                                    df_historico: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula previsão com intervalo de confiança
    """
    # Preparar dados de entrada
    X_pred = pd.DataFrame([inputs])[features]
    
    # Previsão pontual
    previsao = modelo.predict(X_pred)[0]
    
    # Calcular erro padrão residual do modelo
    X_train = df_historico[features]
    y_train = df_historico['Sinistralidade_Realizada']
    y_pred_train = modelo.predict(X_train)
    residuos = y_train - y_pred_train
    erro_padrao = np.std(residuos)
    
    # Intervalo de confiança (95%)
    margem_erro = 1.96 * erro_padrao
    ic_inferior = previsao - margem_erro
    ic_superior = previsao + margem_erro
    
    return {
        'previsao': previsao,
        'ic_inferior': max(0, ic_inferior),
        'ic_superior': min(100, ic_superior),
        'erro_padrao': erro_padrao,
        'confianca': 95
    }
