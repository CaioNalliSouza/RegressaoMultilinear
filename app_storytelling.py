"""
Sistema de Análise Preditiva com Storytelling
Autopeças & Assistência 24h - Apresentação Comercial
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_generator import gerar_dados_assistencia
from model import treinar_modelo, fazer_previsao
from visualizations import *
from utils import *
from config import *
from statistical_analysis import *

# Configuração da página
st.set_page_config(
    page_title="Análise Preditiva - Storytelling",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para storytelling
st.markdown("""
<style>
    .big-metric {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .insight-box {
        background-color: #f0f8ff;
        border-left: 5px solid #1f77b4;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .alert-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .story-title {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 20px;
    }
    .story-subtitle {
        font-size: 1.3rem;
        color: #34495e;
        margin-top: 10px;
    }
    .stat-badge {
        background-color: #e7f3ff;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- Carregamento e Cache de Dados ---
@st.cache_data
def carregar_dados():
    return gerar_dados_assistencia(NUM_MESES_HISTORICO)

@st.cache_resource
def carregar_modelo(dados):
    return treinar_modelo(dados)

@st.cache_data
def calcular_analises_estatisticas(dados, feature_names):
    """Cache de todas as análises estatísticas"""
    return {
        'correlacoes': identificar_correlacoes_fortes(dados, feature_names, threshold=0.4),
        'tendencia_faturamento': analise_tendencia_temporal(dados, 'Faturamento'),
        'tendencia_sinistralidade': analise_tendencia_temporal(dados, 'Sinistralidade_Realizada'),
        'tendencia_nps': analise_tendencia_temporal(dados, 'NPS'),
        'dist_sinistralidade': analise_distribuicao(dados['Sinistralidade_Realizada'].values),
        'comparacao_sinistralidade': analise_comparativa_periodos(dados, 'Sinistralidade_Realizada', 6),
        'comparacao_faturamento': analise_comparativa_periodos(dados, 'Faturamento', 6),
        'capacidade_sinistralidade': calcular_capacidade_processo(
            dados['Sinistralidade_Realizada'].values, 0, 50
        )
    }

# Carregar dados e modelo
dados = carregar_dados()
modelo, feature_names, metricas, X_train, X_test, y_train, y_test = carregar_modelo(dados)
analises = calcular_analises_estatisticas(dados, feature_names)
insights_comerciais = gerar_insights_comerciais(dados, modelo, feature_names)

# Header principal com narrativa
st.markdown('<p class="story-title">📊 Análise Preditiva de Performance Operacional</p>', unsafe_allow_html=True)
st.markdown('<p class="story-subtitle">Autopeças & Assistência 24h - Inteligência de Dados para Decisões Estratégicas</p>', unsafe_allow_html=True)

# Sidebar com KPIs rápidos
with st.sidebar:
    st.header("⚡ Visão Rápida")
    
    # KPIs principais
    ultimo_mes = dados.iloc[-1]
    st.metric(
        "Sinistralidade Atual", 
        f"{ultimo_mes['Sinistralidade_Realizada']:.1f}%",
        delta=f"{ultimo_mes['Sinistralidade_Realizada'] - 50:.1f}% vs Meta",
        delta_color="inverse"
    )
    st.metric(
        "Faturamento Atual", 
        f"R$ {ultimo_mes['Faturamento']/1000:.0f}K",
        delta=f"{((ultimo_mes['Faturamento'] / dados.iloc[-2]['Faturamento']) - 1) * 100:.1f}%"
    )
    st.metric(
        "NPS", 
        f"{ultimo_mes['NPS']:.0f}",
        delta=f"{ultimo_mes['NPS'] - 70:.0f} vs Meta"
    )
    
    st.divider()
    
    # Performance do modelo
    st.markdown("### 🎯 Qualidade Preditiva")
    st.metric("Acurácia (R²)", f"{metricas['r2']:.1%}")
    st.caption(f"MAE: R$ {metricas['mae']:,.0f} | RMSE: R$ {metricas['rmse']:,.0f}")
    
    st.divider()
    
    # Período analisado
    st.markdown("### 📅 Base Histórica")
    st.info(f"""
    **{len(dados)} meses** de dados
    
    {dados['Data'].min().strftime('%b/%Y')} → {dados['Data'].max().strftime('%b/%Y')}
    """)

# Tabs com storytelling
tabs = st.tabs([
    "📖 Sumário Executivo",
    "📊 Análise Estatística Detalhada", 
    "🔮 Simulador Preditivo",
    "🎯 Insights & Recomendações",
    "📈 Evolução Temporal"
])

# ============================================================
# TAB 1: SUMÁRIO EXECUTIVO (STORYTELLING)
# ============================================================
with tabs[0]:
    st.markdown("## 📖 A História dos Números")
    st.markdown("*Uma jornada analítica pelos últimos 48 meses de operação*")
    
    st.divider()
    
    # Seção 1: Contexto do Negócio
    st.markdown("### 🏢 Contexto do Negócio")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="big-metric">{len(dados)}</div>', unsafe_allow_html=True)
        st.caption("Meses Analisados")
    with col2:
        st.markdown(f'<div class="big-metric">{dados["Qtd_Atendimentos"].sum():,.0f}</div>', unsafe_allow_html=True)
        st.caption("Atendimentos Realizados")
    with col3:
        st.markdown(f'<div class="big-metric">R$ {dados["Faturamento"].sum()/1000000:.1f}M</div>', unsafe_allow_html=True)
        st.caption("Faturamento Acumulado")
    with col4:
        st.markdown(f'<div class="big-metric">{analises["dist_sinistralidade"]["media"]:.1f}%</div>', unsafe_allow_html=True)
        st.caption("Sinistralidade Média")
    
    st.markdown("""
    Nossa análise compreende **4 anos completos de operação**, período no qual a empresa processou 
    milhares de atendimentos e consolidou sua presença no mercado de autopeças e assistência 24h. 
    Durante este tempo, observamos padrões claros de comportamento operacional e oportunidades 
    significativas de otimização.
    """)
    
    # Seção 2: O Desafio da Sinistralidade
    st.divider()
    st.markdown("### 🎯 O Desafio Central: Sinistralidade")
    
    sin_stats = analises['dist_sinistralidade']
    capacidade = analises['capacidade_sinistralidade']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_sin_hist = criar_grafico_sinistralidade(dados)
        st.plotly_chart(fig_sin_hist, use_container_width=True)
    
    with col2:
        st.markdown(f"**Média Histórica:** {sin_stats['media']:.1f}%")
        st.markdown(f"**Meta Estabelecida:** 50%")
        st.markdown(f"**Desvio da Meta:** {sin_stats['media'] - 50:.1f} pontos")
        st.markdown(f"**Variabilidade (CV):** {sin_stats['coeficiente_variacao']:.1f}%")
        
        st.markdown("---")
        st.markdown(f"**Capacidade do Processo:** {capacidade['status']}")
        st.caption(f"Cpk = {capacidade['cpk']:.2f}")
        st.caption(f"{capacidade['dentro_limites']:.1f}% dos meses dentro da meta")
    
    # Interpretação comercial
    if sin_stats['media'] > 50:
        st.markdown(f"""
        <div class="alert-box">
        <b>⚠️ Análise Crítica:</b> A sinistralidade média de <b>{sin_stats['media']:.1f}%</b> está 
        <b>{sin_stats['media'] - 50:.1f} pontos percentuais acima da meta</b>. Isso representa uma 
        oportunidade significativa de melhoria que pode impactar diretamente a rentabilidade. 
        Uma redução de apenas 5 pontos percentuais representaria uma economia aproximada de 
        <b>R$ {(dados['Faturamento'].mean() * 0.05)/1000:.0f}K por mês</b>.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="success-box">
        <b>✅ Gestão Eficiente:</b> A sinistralidade média de <b>{sin_stats['media']:.1f}%</b> está 
        <b>dentro da meta estabelecida</b>, demonstrando eficiência operacional e controle adequado 
        dos custos. Esta performance sustentável é um diferencial competitivo importante.
        </div>
        """, unsafe_allow_html=True)
    
    # Seção 3: Tendências e Projeções
    st.divider()
    st.markdown("### 📈 Tendências Identificadas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💰 Faturamento")
        tend_fat = analises['tendencia_faturamento']
        comp_fat = analises['comparacao_faturamento']
        
        fig_fat = criar_grafico_evolucao_faturamento(dados)
        st.plotly_chart(fig_fat, use_container_width=True)
        
        if tend_fat['tendencia'] == 'Crescente':
            st.markdown(f"""
            <div class="success-box">
            <b>📈 Crescimento Sustentado:</b> O faturamento apresenta tendência de crescimento 
            estatisticamente significativa (p < 0.05), com variação de <b>{tend_fat['variacao_percentual']:.1f}%</b> 
            no período. Os últimos 6 meses mostram {comp_fat['interpretacao']}.
            </div>
            """, unsafe_allow_html=True)
        elif tend_fat['tendencia'] == 'Decrescente':
            st.markdown(f"""
            <div class="alert-box">
            <b>📉 Atenção Necessária:</b> O faturamento apresenta tendência de queda de 
            <b>{abs(tend_fat['variacao_percentual']):.1f}%</b>. É fundamental implementar 
            estratégias de recuperação e retenção de clientes.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="insight-box">
            <b>➡️ Estabilidade:</b> O faturamento mantém-se estável, sem tendência clara de 
            crescimento ou queda. Momento ideal para investir em iniciativas de crescimento.
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 😊 Satisfação do Cliente (NPS)")
        tend_nps = analises['tendencia_nps']
        
        fig_nps = criar_grafico_nps(dados)
        st.plotly_chart(fig_nps, use_container_width=True)
        
        nps_atual = dados['NPS'].iloc[-1]
        if nps_atual >= 70:
            st.markdown(f"""
            <div class="success-box">
            <b>🌟 Excelência no Atendimento:</b> NPS atual de <b>{nps_atual:.0f}</b> indica 
            alta satisfação dos clientes. Manter este nível é estratégico para fidelização.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-box">
            <b>⚠️ Oportunidade de Melhoria:</b> NPS de <b>{nps_atual:.0f}</b> está abaixo 
            da meta de 70. Investir em experiência do cliente pode gerar resultados significativos.
            </div>
            """, unsafe_allow_html=True)
    
    # Seção 4: Insights Principais
    st.divider()
    st.markdown("### 💡 Principais Insights Comerciais")
    
    for i, insight in enumerate(insights_comerciais[:5], 1):
        tipo_class = "success-box" if insight['tipo'].startswith('✅') else "alert-box" if insight['tipo'].startswith('🚨') else "insight-box"
        st.markdown(f"""
        <div class="{tipo_class}">
        <b>{insight['tipo']} {insight['titulo']}</b><br>
        {insight['descricao']}<br>
        <span class="stat-badge">Impacto: {insight['impacto']}</span>
        <span class="stat-badge">Ação: {insight['acao']}</span>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# TAB 2: ANÁLISE ESTATÍSTICA DETALHADA
# ============================================================
with tabs[1]:
    st.markdown("## 📊 Análise Estatística Profunda")
    st.markdown("*Rigor estatístico para fundamentar decisões estratégicas*")
    
    # Seção 1: Correlações entre Variáveis
    st.divider()
    st.markdown("### 🔗 Análise de Correlações")
    
    st.markdown("""
    Utilizando o **coeficiente de correlação de Pearson**, identificamos as relações mais 
    fortes entre as variáveis operacionais. Correlações significativas (p < 0.05) indicam 
    relações estatisticamente válidas que podem ser exploradas estrategicamente.
    """)
    
    # Matriz de correlação visual
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_corr = criar_heatmap_correlacao(dados, feature_names)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        st.markdown("**Correlações Identificadas:**")
        
        correlacoes_fortes = analises['correlacoes'][:5]
        for corr in correlacoes_fortes:
            st.markdown(f"""
            **{corr['variavel_1']}** ↔️ **{corr['variavel_2']}**
            - Correlação: `{corr['correlacao']:.3f}` ({corr['forca']})
            - Significância: {corr['significancia']} (p={corr['p_value']:.4f})
            - Tipo: {corr['tipo']}
            """)
            st.markdown("---")
    
    # Interpretação das correlações mais relevantes
    st.markdown("#### 🎯 Interpretação Comercial das Correlações")
    
    for corr in correlacoes_fortes[:3]:
        st.markdown(f"""
        <div class="insight-box">
        <b>Relação: {corr['variavel_1']} × {corr['variavel_2']}</b><br>
        Com correlação {corr['tipo'].lower()} de <b>{abs(corr['correlacao']):.2f}</b> e 
        significância {corr['significancia'].lower()}, esta relação indica que mudanças em 
        <b>{corr['variavel_1']}</b> tendem a estar associadas a mudanças 
        {'no mesmo sentido' if corr['tipo'] == 'Positiva' else 'em sentido oposto'} em 
        <b>{corr['variavel_2']}</b>.
        </div>
        """, unsafe_allow_html=True)
    
    # Seção 2: Distribuições Estatísticas
    st.divider()
    st.markdown("### 📊 Análise de Distribuições")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Sinistralidade Realizada")
        sin_dist = analises['dist_sinistralidade']
        
        # Box plot
        fig_box_sin = criar_boxplot_sinistralidade(dados)
        st.plotly_chart(fig_box_sin, use_container_width=True)
        
        st.markdown(f"""
        **Estatísticas Descritivas:**
        - **Média:** {sin_dist['media']:.2f}%
        - **Mediana:** {sin_dist['mediana']:.2f}%
        - **Desvio Padrão:** {sin_dist['desvio_padrao']:.2f}%
        - **Coef. Variação:** {sin_dist['coeficiente_variacao']:.1f}%
        
        **Quartis:**
        - Q1 (25%): {sin_dist['q1']:.2f}%
        - Q3 (75%): {sin_dist['q3']:.2f}%
        - IQR: {sin_dist['iqr']:.2f}%
        
        **Outliers:** {sin_dist['num_outliers']} meses ({sin_dist['pct_outliers']:.1f}%)
        """)
    
    with col2:
        st.markdown("#### Faturamento")
        fat_dist = analise_distribuicao(dados['Faturamento'].values)
        
        fig_box_fat = criar_boxplot_faturamento(dados)
        st.plotly_chart(fig_box_fat, use_container_width=True)
        
        st.markdown(f"""
        **Estatísticas Descritivas:**
        - **Média:** R$ {fat_dist['media']:,.2f}
        - **Mediana:** R$ {fat_dist['mediana']:,.2f}
        - **Desvio Padrão:** R$ {fat_dist['desvio_padrao']:,.2f}
        - **Coef. Variação:** {fat_dist['coeficiente_variacao']:.1f}%
        
        **Quartis:**
        - Q1 (25%): R$ {fat_dist['q1']:,.2f}
        - Q3 (75%): R$ {fat_dist['q3']:,.2f}
        - IQR: R$ {fat_dist['iqr']:,.2f}
        
        **Outliers:** {fat_dist['num_outliers']} meses ({fat_dist['pct_outliers']:.1f}%)
        """)
    
    # Seção 3: Testes de Hipótese
    st.divider()
    st.markdown("### 🔬 Testes de Hipótese")
    
    st.markdown("""
    Comparamos o desempenho dos **últimos 6 meses** com o período anterior para identificar 
    mudanças estatisticamente significativas (teste t de Student, α = 0.05).
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        comp_sin = analises['comparacao_sinistralidade']
        st.markdown("#### Sinistralidade: Últimos 6 Meses vs Anterior")
        
        st.metric(
            "Período Anterior",
            f"{comp_sin['media_anterior']:.2f}%"
        )
        st.metric(
            "Últimos 6 Meses",
            f"{comp_sin['media_recente']:.2f}%",
            delta=f"{comp_sin['variacao_percentual']:.1f}%"
        )
        
        if comp_sin['significante']:
            st.markdown(f"""
            <div class="{'success-box' if comp_sin['status'] == 'melhora' else 'alert-box'}">
            <b>Resultado Significativo:</b> Observamos {comp_sin['interpretacao']} 
            (t = {comp_sin['t_statistic']:.2f}, p = {comp_sin['p_value']:.4f}).
            Status: <b>{comp_sin['status'].upper()}</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="insight-box">
            <b>Sem Mudança Significativa:</b> A diferença observada não é estatisticamente 
            significativa (p = {comp_sin['p_value']:.4f} > 0.05).
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        comp_fat = analises['comparacao_faturamento']
        st.markdown("#### Faturamento: Últimos 6 Meses vs Anterior")
        
        st.metric(
            "Período Anterior",
            f"R$ {comp_fat['media_anterior']/1000:.0f}K"
        )
        st.metric(
            "Últimos 6 Meses",
            f"R$ {comp_fat['media_recente']/1000:.0f}K",
            delta=f"{comp_fat['variacao_percentual']:.1f}%"
        )
        
        if comp_fat['significante']:
            st.markdown(f"""
            <div class="{'success-box' if comp_fat['status'] == 'melhora' else 'alert-box'}">
            <b>Resultado Significativo:</b> Observamos {comp_fat['interpretacao']} 
            (t = {comp_fat['t_statistic']:.2f}, p = {comp_fat['p_value']:.4f}).
            Status: <b>{comp_fat['status'].upper()}</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="insight-box">
            <b>Sem Mudança Significativa:</b> A diferença observada não é estatisticamente 
            significativa (p = {comp_fat['p_value']:.4f} > 0.05).
            </div>
            """, unsafe_allow_html=True)
    
    # Seção 4: Capacidade do Processo
    st.divider()
    st.markdown("### ⚙️ Análise de Capacidade do Processo")
    
    st.markdown("""
    A **análise de capacidade** avalia se o processo é capaz de atender as especificações estabelecidas 
    (sinistralidade ≤ 50%). Utilizamos os índices Cp (capacidade potencial) e Cpk (capacidade real).
    """)
    
    cap = analises['capacidade_sinistralidade']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Índice Cp", f"{cap['cp']:.3f}")
        st.caption("Capacidade Potencial")
    with col2:
        st.metric("Índice Cpk", f"{cap['cpk']:.3f}")
        st.caption("Capacidade Real")
    with col3:
        st.metric("Conformidade", f"{cap['dentro_limites']:.1f}%")
        st.caption("Meses dentro da meta")
    
    st.markdown(f"""
    <div class="insight-box">
    <b>Diagnóstico:</b> {cap['interpretacao']}<br>
    <b>Status:</b> {cap['status']}<br><br>
    
    <b>Interpretação dos Índices:</b><br>
    • Cpk ≥ 1.33: Processo capaz (excelente)<br>
    • Cpk ≥ 1.00: Processo adequado (bom)<br>
    • Cpk ≥ 0.67: Processo marginal (atenção)<br>
    • Cpk < 0.67: Processo incapaz (crítico)
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# TAB 3: SIMULADOR PREDITIVO
# ============================================================
with tabs[2]:
    st.markdown("## 🔮 Simulador de Cenários Preditivos")
    st.markdown("*Utilize o modelo de machine learning para projetar cenários futuros*")
    
    st.divider()
    
    st.markdown("""
    ### 🎯 Como Usar o Simulador
    
    1. **Ajuste os parâmetros** abaixo baseado em cenários reais ou hipotéticos
    2. **Observe a previsão** com intervalo de confiança de 95%
    3. **Analise o impacto** de cada variável na sinistralidade projetada
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("#### 💰 Financeiro")
        fat_ant = st.number_input(
            "Faturamento Mês Anterior (R$)", 
            min_value=100000.0, 
            value=float(dados['Faturamento'].iloc[-1]), 
            step=10000.0,
            help="Faturamento do mês imediatamente anterior"
        )
        sinistralidade_ant = st.slider(
            "Sinistralidade Mês Anterior (%)", 
            min_value=30.0, 
            max_value=80.0, 
            value=float(dados['Sinistralidade_Realizada'].iloc[-1]),
            step=0.5,
            help="Sinistralidade realizada no mês anterior"
        )
    
    with col2:
        st.markdown("#### 📦 Operacional")
        qtd_atend = st.number_input(
            "Quantidade de Atendimentos", 
            min_value=100, 
            value=int(dados['Qtd_Atendimentos'].iloc[-1]), 
            step=50,
            help="Volume esperado de atendimentos"
        )
        ticket_medio = st.number_input(
            "Ticket Médio (R$)", 
            min_value=100.0, 
            value=float(dados['Ticket_Medio'].iloc[-1]), 
            step=50.0,
            help="Valor médio por atendimento"
        )
        perc_pecas = st.slider(
            "% Atend. com Peças", 
            min_value=30.0, 
            max_value=100.0, 
            value=float(dados['Perc_Atend_Com_Pecas'].iloc[-1]),
            help="Percentual de atendimentos que necessitam peças"
        )
    
    with col3:
        st.markdown("#### ⏱️ Qualidade")
        tempo_atend = st.number_input(
            "Tempo Médio Atend. (horas)", 
            min_value=1.0, 
            max_value=10.0, 
            value=float(dados['Tempo_Medio_Atend_Horas'].iloc[-1]), 
            step=0.5,
            help="Tempo médio de resolução"
        )
        taxa_reincidencia = st.slider(
            "Taxa de Reincidência (%)", 
            min_value=0.0, 
            max_value=30.0, 
            value=float(dados['Taxa_Reincidencia'].iloc[-1]),
            help="% de casos que retornam"
        )
        nps = st.slider(
            "NPS", 
            min_value=0, 
            max_value=100, 
            value=int(dados['NPS'].iloc[-1]),
            help="Net Promoter Score"
        )
    
    with col4:
        st.markdown("#### 🌍 Externos")
        taxa_juros = st.number_input(
            "Taxa de Juros (%)", 
            min_value=5.0, 
            max_value=20.0, 
            value=float(dados['Taxa_Juros'].iloc[-1]), 
            step=0.5,
            help="Taxa Selic ou referencial"
        )
        indice_acidentes = st.slider(
            "Índice de Acidentes", 
            min_value=50.0, 
            max_value=150.0, 
            value=float(dados['Indice_Acidentes'].iloc[-1]),
            help="Índice de sinistralidade do mercado"
        )
    
    # Preparar inputs para previsão
    inputs_previsao = {
        'Faturamento_Mes_Ant': fat_ant,
        'Qtd_Atendimentos': qtd_atend,
        'Ticket_Medio': ticket_medio,
        'Perc_Atend_Com_Pecas': perc_pecas,
        'Tempo_Medio_Atend_Horas': tempo_atend,
        'Taxa_Reincidencia': taxa_reincidencia,
        'Sinistralidade_Mes_Ant': sinistralidade_ant,
        'NPS': nps,
        'Taxa_Juros': taxa_juros,
        'Indice_Acidentes': indice_acidentes
    }
    
    # Fazer previsão com intervalo de confiança
    previsao_completa = calcular_previsao_com_intervalo(modelo, feature_names, inputs_previsao, dados)
    
    st.divider()
    
    # Mostrar resultado
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### 📊 Resultado da Previsão")
        
        fig_gauge = criar_gauge_sinistralidade(previsao_completa['previsao'])
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Previsão Pontual")
        st.markdown(f'<div class="big-metric">{previsao_completa["previsao"]:.1f}%</div>', unsafe_allow_html=True)
        
        delta_meta = previsao_completa['previsao'] - 50
        if delta_meta > 0:
            st.markdown(f"<span style='color:red; font-size:1.2rem;'>↑ {delta_meta:.1f}% acima da meta</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:green; font-size:1.2rem;'>↓ {abs(delta_meta):.1f}% abaixo da meta</span>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("### 📈 Intervalo de Confiança")
        st.markdown(f"**Inferior:** {previsao_completa['ic_inferior']:.1f}%")
        st.markdown(f"**Superior:** {previsao_completa['ic_superior']:.1f}%")
        st.caption(f"Confiança: {previsao_completa['confianca']}%")
        st.caption(f"Erro padrão: ±{previsao_completa['erro_padrao']:.2f}%")
    
    # Interpretação
    st.markdown("### 💬 Interpretação do Resultado")
    
    if previsao_completa['previsao'] <= 50:
        st.markdown(f"""
        <div class="success-box">
        <b>✅ Cenário Favorável:</b> A sinistralidade prevista de <b>{previsao_completa['previsao']:.1f}%</b> 
        está dentro da meta estabelecida. Com 95% de confiança, o valor real estará entre 
        <b>{previsao_completa['ic_inferior']:.1f}%</b> e <b>{previsao_completa['ic_superior']:.1f}%</b>.
        Este cenário indica gestão adequada dos custos operacionais.
        </div>
        """, unsafe_allow_html=True)
    elif previsao_completa['previsao'] <= 60:
        st.markdown(f"""
        <div class="alert-box">
        <b>⚠️ Atenção Necessária:</b> A sinistralidade prevista de <b>{previsao_completa['previsao']:.1f}%</b> 
        está acima da meta, mas ainda em nível controlável. Com 95% de confiança, o valor real estará entre 
        <b>{previsao_completa['ic_inferior']:.1f}%</b> e <b>{previsao_completa['ic_superior']:.1f}%</b>.
        Recomenda-se monitoramento próximo e ajustes operacionais.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-box" style="border-left-color: #dc3545;">
        <b>🚨 Situação Crítica:</b> A sinistralidade prevista de <b>{previsao_completa['previsao']:.1f}%</b> 
        está significativamente acima da meta. Com 95% de confiança, o valor real estará entre 
        <b>{previsao_completa['ic_inferior']:.1f}%</b> e <b>{previsao_completa['ic_superior']:.1f}%</b>.
        <b>Ação imediata é necessária</b> para reverter este cenário.
        </div>
        """, unsafe_allow_html=True)
    
    # Análise de sensibilidade
    st.divider()
    st.markdown("### 🎚️ Importância das Variáveis")
    
    fig_importancia = criar_grafico_importancia_features(modelo, feature_names)
    st.plotly_chart(fig_importancia, use_container_width=True)
    
    st.markdown("""
    Este gráfico mostra o **impacto relativo** de cada variável na previsão. Variáveis com maior 
    coeficiente (em valor absoluto) têm maior influência no resultado final.
    """)

# ============================================================
# TAB 4: INSIGHTS & RECOMENDAÇÕES
# ============================================================
with tabs[3]:
    st.markdown("## 🎯 Insights Estratégicos e Recomendações")
    st.markdown("*Do dado à ação: direcionamentos baseados em evidências*")
    
    st.divider()
    
    # Insights prioritários
    st.markdown("### 🔥 Prioridades de Ação")
    
    for i, insight in enumerate(insights_comerciais, 1):
        with st.expander(f"{insight['tipo']} {insight['titulo']}", expanded=(i <= 3)):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Descrição:** {insight['descricao']}")
                st.markdown(f"**Ação Recomendada:** {insight['acao']}")
            
            with col2:
                st.metric("Impacto", insight['impacto'])
    
    # Recomendações por área
    st.divider()
    st.markdown("### 📋 Plano de Ação por Área")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💰 Gestão Financeira")
        st.markdown("""
        <div class="insight-box">
        <b>Objetivo:</b> Manter sinistralidade ≤ 50%<br><br>
        
        <b>Ações Imediatas:</b>
        • Revisar contratos com fornecedores de peças<br>
        • Implementar negociação em lote para maiores volumes<br>
        • Analisar outliers de custo mês a mês<br><br>
        
        <b>Ações Médio Prazo:</b>
        • Desenvolver programa de fornecedores preferenciais<br>
        • Implementar sistema de cotação automática<br>
        • Criar fundo de reserva para eventos atípicos
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### ⚙️ Eficiência Operacional")
        st.markdown("""
        <div class="insight-box">
        <b>Objetivo:</b> Tempo médio ≤ 3h | Reincidência < 5%<br><br>
        
        <b>Ações Imediatas:</b>
        • Mapear gargalos no processo de atendimento<br>
        • Treinar equipe em procedimentos padronizados<br>
        • Implementar checklist de qualidade<br><br>
        
        <b>Ações Médio Prazo:</b>
        • Automatizar etapas de diagnóstico<br>
        • Criar base de conhecimento de soluções<br>
        • Implementar sistema de gestão de filas
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 😊 Experiência do Cliente")
        st.markdown("""
        <div class="insight-box">
        <b>Objetivo:</b> NPS ≥ 70<br><br>
        
        <b>Ações Imediatas:</b>
        • Implementar pesquisa pós-atendimento<br>
        • Criar canal de feedback direto<br>
        • Treinar equipe em atendimento humanizado<br><br>
        
        <b>Ações Médio Prazo:</b>
        • Programa de fidelização de clientes<br>
        • Sistema de acompanhamento proativo<br>
        • Benefícios para clientes recorrentes
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📈 Crescimento Comercial")
        st.markdown("""
        <div class="insight-box">
        <b>Objetivo:</b> Crescimento sustentável de 10-15% a.a.<br><br>
        
        <b>Ações Imediatas:</b>
        • Identificar clientes de alto potencial<br>
        • Desenvolver propostas personalizadas<br>
        • Intensificar ações de marketing<br><br>
        
        <b>Ações Médio Prazo:</b>
        • Expandir para novas regiões geográficas<br>
        • Desenvolver novos produtos/serviços<br>
        • Parcerias estratégicas com seguradoras
        </div>
        """, unsafe_allow_html=True)
    
    # Monitoramento
    st.divider()
    st.markdown("### 📡 Sistema de Monitoramento Contínuo")
    
    st.markdown("""
    <div class="success-box">
    <b>🎯 KPIs para Acompanhamento Mensal:</b><br><br>
    
    <b>Críticos (Revisão Semanal):</b><br>
    • Sinistralidade Realizada vs Orçada<br>
    • Faturamento Acumulado vs Meta<br>
    • NPS Médio<br><br>
    
    <b>Importantes (Revisão Mensal):</b><br>
    • Volume de Atendimentos<br>
    • Ticket Médio<br>
    • Taxa de Reincidência<br>
    • Tempo Médio de Atendimento<br><br>
    
    <b>Estratégicos (Revisão Trimestral):</b><br>
    • Tendência de Crescimento<br>
    • Capacidade do Processo (Cpk)<br>
    • Correlações entre Variáveis<br>
    • ROI de Iniciativas Implementadas
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# TAB 5: EVOLUÇÃO TEMPORAL
# ============================================================
with tabs[4]:
    st.markdown("## 📈 Evolução Temporal Completa")
    st.markdown("*Análise detalhada da série histórica*")
    
    st.divider()
    
    # Gráficos de série temporal
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💰 Evolução do Faturamento")
        fig_fat_tempo = criar_grafico_evolucao_faturamento(dados)
        st.plotly_chart(fig_fat_tempo, use_container_width=True)
        
        tend_fat = analises['tendencia_faturamento']
        st.markdown(f"""
        **Tendência:** {tend_fat['tendencia']} ({tend_fat['interpretacao']})  
        **R²:** {tend_fat['r_squared']:.3f} | **p-valor:** {tend_fat['p_value']:.4f}  
        **Variação Total:** {tend_fat['variacao_percentual']:.1f}%
        """)
    
    with col2:
        st.markdown("### 📊 Evolução da Sinistralidade")
        fig_sin_tempo = criar_grafico_sinistralidade(dados)
        st.plotly_chart(fig_sin_tempo, use_container_width=True)
        
        tend_sin = analises['tendencia_sinistralidade']
        st.markdown(f"""
        **Tendência:** {tend_sin['tendencia']} ({tend_sin['interpretacao']})  
        **R²:** {tend_sin['r_squared']:.3f} | **p-valor:** {tend_sin['p_value']:.4f}  
        **Variação Total:** {tend_sin['variacao_percentual']:.1f}%
        """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 😊 Evolução do NPS")
        fig_nps_tempo = criar_grafico_nps(dados)
        st.plotly_chart(fig_nps_tempo, use_container_width=True)
        
        tend_nps = analises['tendencia_nps']
        st.markdown(f"""
        **Tendência:** {tend_nps['tendencia']} ({tend_nps['interpretacao']})  
        **R²:** {tend_nps['r_squared']:.3f} | **p-valor:** {tend_nps['p_value']:.4f}  
        **Variação Total:** {tend_nps['variacao_percentual']:.1f}%
        """)
    
    with col2:
        st.markdown("### 📦 Evolução do Volume")
        fig_vol = criar_grafico_atendimentos(dados)
        st.plotly_chart(fig_vol, use_container_width=True)
        
        tend_vol = analise_tendencia_temporal(dados, 'Qtd_Atendimentos')
        st.markdown(f"""
        **Tendência:** {tend_vol['tendencia']} ({tend_vol['interpretacao']})  
        **R²:** {tend_vol['r_squared']:.3f} | **p-valor:** {tend_vol['p_value']:.4f}  
        **Variação Total:** {tend_vol['variacao_percentual']:.1f}%
        """)
    
    # Análise de sazonalidade
    st.divider()
    st.markdown("### 📅 Análise de Sazonalidade")
    
    # Adicionar mês ao dataframe
    dados_sazon = dados.copy()
    dados_sazon['Mes'] = pd.to_datetime(dados_sazon['Data']).dt.month
    
    sazonalidade_fat = dados_sazon.groupby('Mes')['Faturamento'].mean().reset_index()
    sazonalidade_sin = dados_sazon.groupby('Mes')['Sinistralidade_Realizada'].mean().reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_sazon_fat = criar_grafico_sazonalidade(sazonalidade_fat, 'Faturamento', 'Faturamento Médio por Mês')
        st.plotly_chart(fig_sazon_fat, use_container_width=True)
    
    with col2:
        fig_sazon_sin = criar_grafico_sazonalidade(sazonalidade_sin, 'Sinistralidade_Realizada', 'Sinistralidade Média por Mês')
        st.plotly_chart(fig_sazon_sin, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
    <b>Interpretação da Sazonalidade:</b><br>
    Os gráficos acima mostram a média de cada métrica por mês do ano (agregando todos os anos da base). 
    Padrões consistentes indicam sazonalidade que deve ser considerada no planejamento orçamentário 
    e na alocação de recursos.
    </div>
    """, unsafe_allow_html=True)
    
    # Tabela resumo completa
    st.divider()
    st.markdown("### 📋 Resumo Estatístico Completo")
    
    resumo_stats = pd.DataFrame({
        'Métrica': [
            'Sinistralidade Realizada (%)',
            'Faturamento (R$)',
            'Qtd Atendimentos',
            'Ticket Médio (R$)',
            'NPS',
            'Tempo Atendimento (h)',
            'Taxa Reincidência (%)'
        ],
        'Média': [
            f"{dados['Sinistralidade_Realizada'].mean():.2f}",
            f"{dados['Faturamento'].mean():,.0f}",
            f"{dados['Qtd_Atendimentos'].mean():.0f}",
            f"{dados['Ticket_Medio'].mean():.2f}",
            f"{dados['NPS'].mean():.1f}",
            f"{dados['Tempo_Medio_Atend_Horas'].mean():.2f}",
            f"{dados['Taxa_Reincidencia'].mean():.2f}"
        ],
        'Mediana': [
            f"{dados['Sinistralidade_Realizada'].median():.2f}",
            f"{dados['Faturamento'].median():,.0f}",
            f"{dados['Qtd_Atendimentos'].median():.0f}",
            f"{dados['Ticket_Medio'].median():.2f}",
            f"{dados['NPS'].median():.1f}",
            f"{dados['Tempo_Medio_Atend_Horas'].median():.2f}",
            f"{dados['Taxa_Reincidencia'].median():.2f}"
        ],
        'Desvio Padrão': [
            f"{dados['Sinistralidade_Realizada'].std():.2f}",
            f"{dados['Faturamento'].std():,.0f}",
            f"{dados['Qtd_Atendimentos'].std():.0f}",
            f"{dados['Ticket_Medio'].std():.2f}",
            f"{dados['NPS'].std():.1f}",
            f"{dados['Tempo_Medio_Atend_Horas'].std():.2f}",
            f"{dados['Taxa_Reincidencia'].std():.2f}"
        ],
        'Mínimo': [
            f"{dados['Sinistralidade_Realizada'].min():.2f}",
            f"{dados['Faturamento'].min():,.0f}",
            f"{dados['Qtd_Atendimentos'].min():.0f}",
            f"{dados['Ticket_Medio'].min():.2f}",
            f"{dados['NPS'].min():.1f}",
            f"{dados['Tempo_Medio_Atend_Horas'].min():.2f}",
            f"{dados['Taxa_Reincidencia'].min():.2f}"
        ],
        'Máximo': [
            f"{dados['Sinistralidade_Realizada'].max():.2f}",
            f"{dados['Faturamento'].max():,.0f}",
            f"{dados['Qtd_Atendimentos'].max():.0f}",
            f"{dados['Ticket_Medio'].max():.2f}",
            f"{dados['NPS'].max():.1f}",
            f"{dados['Tempo_Medio_Atend_Horas'].max():.2f}",
            f"{dados['Taxa_Reincidencia'].max():.2f}"
        ]
    })
    
    st.dataframe(resumo_stats, use_container_width=True, hide_index=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <b>Sistema de Análise Preditiva</b> | Autopeças & Assistência 24h<br>
    Powered by Machine Learning & Statistical Analysis
</div>
""", unsafe_allow_html=True)
