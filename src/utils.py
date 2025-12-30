"""
Módulo com funções utilitárias e auxiliares.
"""
import pandas as pd


def calcular_metricas_derivadas(predicao, sinistralidade_ant, qtd_atendimentos, ticket_medio):
    """
    Calcula métricas derivadas da previsão.
    
    Args:
        predicao (float): Faturamento previsto
        sinistralidade_ant (float): Sinistralidade do mês anterior
        qtd_atendimentos (int): Quantidade de atendimentos
        ticket_medio (float): Ticket médio
    
    Returns:
        dict: Dicionário com métricas calculadas
    """
    margem_bruta = predicao - (predicao * sinistralidade_ant / 100)
    ticket_real = predicao / qtd_atendimentos if qtd_atendimentos > 0 else 0
    
    return {
        'margem_bruta': margem_bruta,
        'ticket_real': ticket_real
    }


def gerar_recomendacoes(tempo_atend, taxa_reincidencia, nps, sinistralidade_ant, 
                        sinistralidade_orcada, perc_pecas):
    """
    Gera recomendações baseadas nos indicadores.
    
    Returns:
        list: Lista de recomendações
    """
    recomendacoes = []
    
    if tempo_atend > 3.0:
        recomendacoes.append("🔧 Reduzir tempo de atendimento (meta: < 3h) para melhorar satisfação e reduzir custos operacionais")
    
    if taxa_reincidencia > 10:
        recomendacoes.append("🔄 Investigar causas de reincidência elevada - pode indicar problemas de qualidade no serviço")
    
    if nps < 70:
        recomendacoes.append("😊 Melhorar NPS através de treinamento da equipe e redução de tempo de resposta")
    
    if sinistralidade_ant > 50:
        recomendacoes.append(f"💰 Sinistralidade {sinistralidade_ant:.1f}% acima da meta (50%) - Revisar precificação ou renegociar contratos")
    
    if sinistralidade_ant > sinistralidade_orcada + 5:
        recomendacoes.append(f"📉 Sinistralidade realizada ({sinistralidade_ant:.1f}%) está {(sinistralidade_ant - sinistralidade_orcada):.1f}% acima do orçado ({sinistralidade_orcada:.1f}%) - Investigar causas")
    
    if perc_pecas < 40:
        recomendacoes.append("📦 Avaliar oportunidades de venda de peças - percentual abaixo do potencial")
    
    return recomendacoes


def determinar_status_sinistralidade(sinistralidade_ant):
    """
    Determina o status da sinistralidade baseado na meta de 50%.
    
    Returns:
        tuple: (mensagem, cor)
    """
    if sinistralidade_ant <= 50:
        return "✅ **Excelente** - Sinistralidade dentro da meta (≤ 50%)", "success"
    elif sinistralidade_ant <= 60:
        return "⚠️ **Atenção** - Sinistralidade acima da meta, mas aceitável (50-60%)", "warning"
    else:
        return "🚨 **Crítico** - Sinistralidade elevada, requer ação imediata (> 60%)", "error"


def preparar_dados_sazonalidade(dados):
    """Prepara dados para análise de sazonalidade."""
    dados_sazon = dados.groupby('Mes').agg({
        'Faturamento': 'mean',
        'Qtd_Atendimentos': 'mean',
        'Sinistralidade': 'mean'
    }).reset_index()
    
    dados_sazon['Mes_Nome'] = dados_sazon['Mes'].apply(
        lambda x: pd.to_datetime(str(x), format='%m').strftime('%B')
    )
    
    return dados_sazon


def calcular_estatisticas_sinistralidade(dados):
    """Calcula estatísticas de sinistralidade."""
    meses_dentro_meta = (dados['Sinistralidade_Realizada'] <= 50).sum()
    perc_dentro_meta = (meses_dentro_meta / len(dados)) * 100
    desvio_medio = (dados['Sinistralidade_Realizada'] - dados['Sinistralidade_Orcada']).mean()
    
    return {
        'meses_dentro_meta': meses_dentro_meta,
        'perc_dentro_meta': perc_dentro_meta,
        'desvio_medio': desvio_medio,
        'media_realizada': dados['Sinistralidade_Realizada'].mean(),
        'media_orcada': dados['Sinistralidade_Orcada'].mean(),
        'min_realizada': dados['Sinistralidade_Realizada'].min(),
        'max_realizada': dados['Sinistralidade_Realizada'].max()
    }
