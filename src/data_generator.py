"""
Módulo para geração de dados sintéticos de empresas de autopeças e assistência 24h.
"""
import pandas as pd
import numpy as np


def gerar_dados_assistencia(num_meses=48, empresa_selecionada=None):
    """
    Gera dados realistas para análise de empresas parceiras do setor de autopeças e assistência 24h.
    Considera: sinistralidade, faturamento, quantidade de atendimentos, sazonalidade e outros KPIs.
    
    Args:
        num_meses (int): Número de meses de dados históricos a gerar
        empresa_selecionada (str, optional): Nome da empresa específica
    
    Returns:
        pd.DataFrame: DataFrame com dados históricos simulados
    """
    
    np.random.seed(42)  # Para reprodutibilidade
    
    data = pd.date_range(start='2022-01-01', periods=num_meses, freq='MS')
    df = pd.DataFrame(data, columns=['Data'])
    df['Mes'] = df['Data'].dt.month
    df['Trimestre'] = df['Data'].dt.quarter
    df['Tempo_Mes'] = np.arange(num_meses) + 1
    
    # --- FATORES OPERACIONAIS ---
    
    # Quantidade de Atendimentos/Sinistros (varia por sazonalidade)
    sazonalidade_atendimentos = 1 + 0.3 * np.sin((df['Mes'] - 1) * (2 * np.pi / 12))  # Pico no verão
    df['Qtd_Atendimentos'] = np.round(
        800 + df['Tempo_Mes'] * 5 + sazonalidade_atendimentos * 150 + np.random.normal(0, 50, num_meses)
    ).astype(int)
    df['Qtd_Atendimentos'] = np.clip(df['Qtd_Atendimentos'], 500, 1500)
    
    # Percentual de atendimentos com peças (vs só serviço)
    df['Perc_Atend_Com_Pecas'] = 45 + np.random.normal(0, 5, num_meses)
    df['Perc_Atend_Com_Pecas'] = np.clip(df['Perc_Atend_Com_Pecas'], 30, 65)
    
    # Tempo Médio de Atendimento (em horas) - impacta satisfação e custo
    df['Tempo_Medio_Atend_Horas'] = 2.5 + np.random.normal(0, 0.4, num_meses)
    df['Tempo_Medio_Atend_Horas'] = np.clip(df['Tempo_Medio_Atend_Horas'], 1.5, 4.5)
    
    # Taxa de Reincidência (%) - cliente que volta em até 30 dias
    df['Taxa_Reincidencia'] = 8 + np.random.normal(0, 2, num_meses)
    df['Taxa_Reincidencia'] = np.clip(df['Taxa_Reincidencia'], 3, 15)
    
    # Ticket Médio por Atendimento (R$)
    df['Ticket_Medio'] = 450 + df['Tempo_Mes'] * 3 + np.random.normal(0, 50, num_meses)
    df['Ticket_Medio'] = np.clip(df['Ticket_Medio'], 300, 700)
    
    # --- FATORES FINANCEIROS ---
    
    # Criar NPS temporário
    nps_temp = 75 - (df['Tempo_Medio_Atend_Horas'] - 2.5) * 5 - df['Taxa_Reincidencia'] * 0.8 + np.random.normal(0, 2, num_meses)
    nps_temp = np.clip(nps_temp, 55, 90)
    
    # FATURAMENTO com relação LINEAR FORTE e DIRETA
    # Base super forte: Volume * Ticket (80% do faturamento)
    df['Faturamento'] = df['Qtd_Atendimentos'] * df['Ticket_Medio']
    
    # Ajustes lineares simples e fortes (20% do faturamento)
    df['Faturamento'] += df['Perc_Atend_Com_Pecas'] * 1800  # +R$ 1800 por % de peças
    df['Faturamento'] += nps_temp * 850  # +R$ 850 por ponto de NPS
    df['Faturamento'] -= df['Tempo_Medio_Atend_Horas'] * 15000  # -R$ 15k por hora extra
    df['Faturamento'] -= df['Taxa_Reincidencia'] * 3500  # -R$ 3.5k por % reincidência
    
    # Efeito sazonal LINEAR
    sazonalidade_mensal = {1: -25000, 2: -20000, 3: 0, 4: 10000, 5: 15000, 6: 20000,
                          7: 25000, 8: 20000, 9: 10000, 10: 5000, 11: -10000, 12: -15000}
    df['Faturamento'] += df['Mes'].map(sazonalidade_mensal)
    
    # Tendência de crescimento simples
    df['Faturamento'] += df['Tempo_Mes'] * 1200  # +R$ 1.200 por mês
    
    # Ruído mínimo (2%)
    df['Faturamento'] *= np.random.normal(1.0, 0.02, num_meses)
    
    # Limites razoáveis
    df['Faturamento'] = np.clip(df['Faturamento'], 350000, 1100000)
    
    # Custo Total (R$) - ajustado para gerar sinistralidade mais realista
    # Será calculado baseado no faturamento e sinistralidade realizada após definir a sinistralidade
    # Placeholder inicial - será recalculado após definir sinistralidade
    custo_fixo_mensal = 180000
    custo_variavel_por_atendimento = 180 + np.random.normal(0, 20, num_meses)
    df['Custo_Pecas'] = (df['Qtd_Atendimentos'] * df['Perc_Atend_Com_Pecas'] / 100) * 120
    df['Custo_Mao_Obra'] = df['Qtd_Atendimentos'] * 85
    df['Custo_Total_Base'] = (
        custo_fixo_mensal + 
        (df['Qtd_Atendimentos'] * custo_variavel_por_atendimento) +
        df['Custo_Pecas'] +
        df['Custo_Mao_Obra']
    )
    
    # SINISTRALIDADE (%) - Métrica chave: Custo/Faturamento * 100
    
    # Sinistralidade Orçada (planejada/esperada) - base em torno de 50%
    df['Sinistralidade_Orcada'] = 50 + np.random.normal(0, 1.5, num_meses)
    df['Sinistralidade_Orcada'] = np.clip(df['Sinistralidade_Orcada'], 47, 53)
    
    # Sinistralidade Realizada (efetiva) - varia próximo da orçada
    # Variação controlada: média de ±3 a 5 pontos percentuais
    desvio_base = np.random.normal(0, 3, num_meses)  # Variação base controlada
    df['Sinistralidade_Realizada'] = df['Sinistralidade_Orcada'] + desvio_base
    
    # Adicionar alguns outliers realistas (15% dos meses)
    num_outliers = int(num_meses * 0.15)
    outliers_indices = np.random.choice(num_meses, size=num_outliers, replace=False)
    df.loc[outliers_indices, 'Sinistralidade_Realizada'] += np.random.uniform(-4, 7, num_outliers)
    
    # Garantir limites razoáveis (42% a 65%)
    df['Sinistralidade_Realizada'] = np.clip(df['Sinistralidade_Realizada'], 42, 65)
    
    # Meta de Sinistralidade (fixa em 50%)
    df['Sinistralidade_Meta'] = 50.0
    
    # Recalcular custo total baseado na sinistralidade realizada para manter coerência
    df['Custo_Total'] = (df['Faturamento'] * df['Sinistralidade_Realizada']) / 100
    
    # Manter compatibilidade com código existente
    df['Sinistralidade'] = df['Sinistralidade_Realizada']
    
    # --- FATORES EXTERNOS E SAZONALIDADE ---
    
    # Taxa de Juros/SELIC (impacta custos financeiros)
    df['Taxa_Juros'] = 11.5 + np.sin(df['Tempo_Mes'] / 8) * 2 + np.random.normal(0, 0.3, num_meses)
    
    # Índice de Acidentes (correlação com demanda) - simulação
    df['Indice_Acidentes'] = 100 + 15 * np.sin((df['Mes'] - 6) * (2 * np.pi / 12)) + np.random.normal(0, 5, num_meses)
    
    # NPS - Satisfação do Cliente (usar o temporário calculado anteriormente)
    df['NPS'] = nps_temp
    
    # Variáveis Defasadas (mês anterior)
    df['Faturamento_Mes_Ant'] = df['Faturamento'].shift(1).fillna(df['Faturamento'].mean())
    df['Sinistralidade_Mes_Ant'] = df['Sinistralidade_Realizada'].shift(1).fillna(df['Sinistralidade_Realizada'].mean())
    df['Sinistralidade_Orcada_Mes_Ant'] = df['Sinistralidade_Orcada'].shift(1).fillna(df['Sinistralidade_Orcada'].mean())
    
    return df
