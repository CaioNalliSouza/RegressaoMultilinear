"""
Aplicação principal do Sistema de Análise Preditiva.
Autopeças & Assistência 24h - Regressão Multifatorial
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_generator import gerar_dados_assistencia
from model import treinar_modelo, fazer_previsao
from visualizations import *
from utils import *
from config import *

# Configuração da página
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded"
)

# CSS Customizado para melhorar UX/UI
st.markdown("""
<style>
    /* Animações suaves */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Cards com destaque */
    .big-metric {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Caixas de insight */
    .insight-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-left: 5px solid #667eea;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        border-left: 5px solid #00c853;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .alert-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-left: 5px solid #ff6b6b;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Títulos estilizados */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    
    /* Botões */
    .stButton button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Cabeçalho com storytelling
st.title(f"{APP_ICON} {APP_TITLE}")

# Banner principal com contexto
st.markdown("""
<div class="insight-box">
    <h3>🚀 Inteligência de Negócios para Decisões Estratégicas</h3>
    <p style="font-size: 1.1rem; margin-top: 10px;">
        Transforme dados em resultados concretos! Este sistema utiliza <b>análise preditiva</b> 
        para projetar o faturamento da sua operação, considerando mais de <b>10 indicadores-chave</b> 
        que impactam diretamente seu negócio.
    </p>
    <p style="margin-top: 10px;">
        <b>📊 O que analisamos:</b> Histórico de atendimentos, satisfação do cliente, 
        eficiência operacional, tendências de mercado e padrões sazonais.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# --- Carregamento e Cache de Dados ---
@st.cache_data
def carregar_dados():
    return gerar_dados_assistencia(NUM_MESES_HISTORICO)

@st.cache_resource
def carregar_modelo(dados):
    return treinar_modelo(dados)

# Carregar dados e modelo
dados = carregar_dados()
modelo, feature_names, metricas, X_train, X_test, y_train, y_test = carregar_modelo(dados)

# Sidebar com informações do modelo
with st.sidebar:
    st.markdown("### 🎯 Confiabilidade da Previsão")
    
    # Explicar o R² de forma comercial
    r2_percentual = metricas['r2'] * 100
    if r2_percentual >= 90:
        emoji_confianca = "🌟"
        texto_confianca = "Excelente"
        cor_confianca = "#00c853"
    elif r2_percentual >= 75:
        emoji_confianca = "✅"
        texto_confianca = "Muito Boa"
        cor_confianca = "#64dd17"
    elif r2_percentual >= 60:
        emoji_confianca = "👍"
        texto_confianca = "Boa"
        cor_confianca = "#ffd600"
    else:
        emoji_confianca = "⚠️"
        texto_confianca = "Moderada"
        cor_confianca = "#ff6d00"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; color: white; text-align: center;">
        <h2 style="margin: 0; color: white;">{emoji_confianca} {r2_percentual:.1f}%</h2>
        <p style="margin: 5px 0 0 0; font-size: 1.1rem;"><b>{texto_confianca}</b></p>
        <p style="margin: 5px 0 0 0; font-size: 0.9rem;">Confiança nas Previsões</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    st.info(f"""
    💡 **O que isso significa?**
    
    Nosso modelo acerta **{r2_percentual:.1f}%** das variações de faturamento. 
    
    A margem de erro típica é de **± R$ {metricas['mae']:,.0f}** por mês.
    """)
    
    st.divider()
    st.markdown("### 📁 Base de Conhecimento")
    st.success(f"""
    **{len(dados)} meses** de histórico analisados
    
    📅 **Período:** {dados['Data'].min().strftime('%m/%Y')} até {dados['Data'].max().strftime('%m/%Y')}
    
    🔄 Atualização contínua dos padrões
    """)
    
    st.divider()
    st.markdown("### 🎯 Indicadores Analisados")
    st.markdown("""
    O sistema considera:
    
    **💰 Financeiros**
    - Histórico de faturamento
    - Custo de sinistros
    - Ticket médio de atendimento
    
    **📊 Operacionais**
    - Volume de atendimentos
    - Tempo de resposta
    - Taxa de reincidência
    
    **😊 Satisfação**
    - NPS (Net Promoter Score)
    - Qualidade do serviço
    
    **🌍 Mercado**
    - Sazonalidade
    - Taxa de juros
    - Índice de acidentes
    """)

# Tabs principais com navegação intuitiva
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Simulador de Cenários", 
    "📈 Histórico & Tendências", 
    "📊 Painel de Indicadores", 
    "💡 Insights do Modelo"
])

# --- TAB 1: PREVISÃO ---
with tab1:
    st.markdown("""
    <div class="insight-box">
        <h2 style="margin-top: 0;">🔮 Simulador de Cenários Futuro</h2>
        <p style="font-size: 1.1rem;">
            Ajuste os parâmetros abaixo para simular diferentes cenários e descobrir 
            o <b>faturamento projetado</b> para o próximo período. 
        </p>
        <p>
            💡 <b>Dica:</b> Experimente diferentes combinações para encontrar o cenário ideal!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎛️ Configure o Cenário")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("#### 💰 Indicadores Financeiros")
        fat_ant = st.number_input(
            "💵 Faturamento do Último Mês", 
            min_value=100000.0, 
            value=float(dados['Faturamento'].iloc[-1]), 
            step=10000.0, 
            format='%.2f',
            help="Quanto sua operação faturou no mês anterior?"
        )
        sinistralidade_ant = st.slider(
            "⚠️ Custo Real de Sinistros (%)", 
            min_value=30.0, 
            max_value=100.0, 
            value=float(dados['Sinistralidade_Realizada'].iloc[-1]),
            step=0.5,
            help="Percentual real dos custos com sinistros no último mês"
        )
        sinistralidade_orcada = st.slider(
            "🎯 Meta de Custo Planejada (%)", 
            min_value=30.0, 
            max_value=70.0, 
            value=float(dados['Sinistralidade_Orcada'].iloc[-1]),
            step=0.5,
            help="Quanto você planejou gastar com sinistros?"
        )
        
    with col2:
        st.markdown("#### 📦 Operação & Volume")
        qtd_atendimentos = st.number_input(
            "📞 Número de Atendimentos", 
            min_value=100, 
            value=int(dados['Qtd_Atendimentos'].iloc[-1]), 
            step=50,
            help="Quantos atendimentos você espera realizar?"
        )
        ticket_medio = st.number_input(
            "🎫 Valor Médio por Atendimento", 
            min_value=100.0, 
            value=float(dados['Ticket_Medio'].iloc[-1]), 
            step=10.0,
            format='%.2f',
            help="Valor médio que cada atendimento gera"
        )
        perc_pecas = st.slider(
            "🔧 Atendimentos que Usam Peças (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=float(dados['Perc_Atend_Com_Pecas'].iloc[-1]),
            step=1.0,
            help="Percentual de atendimentos que precisam de reposição de peças"
        )
        
    with col3:
        st.markdown("#### ⚙️ Qualidade & Eficiência")
        tempo_atend = st.slider(
            "⏱️ Tempo de Resolução (horas)", 
            min_value=0.5, 
            max_value=6.0, 
            value=float(dados['Tempo_Medio_Atend_Horas'].iloc[-1]),
            step=0.1,
            help="Quanto tempo em média leva cada atendimento?"
        )
        taxa_reincidencia = st.slider(
            "🔄 Taxa de Retorno do Cliente (%)", 
            min_value=0.0, 
            max_value=20.0, 
            value=float(dados['Taxa_Reincidencia'].iloc[-1]),
            step=0.5,
            help="Percentual de clientes que voltam em até 30 dias"
        )
        nps = st.slider(
            "😊 Satisfação dos Clientes (NPS)", 
            min_value=0, 
            max_value=100, 
            value=int(dados['NPS'].iloc[-1]),
            help="Net Promoter Score - quanto maior, mais satisfeitos estão seus clientes"
        )
        
    with col4:
        st.markdown("#### 🌍 Fatores Externos")
        mes_prev = st.selectbox(
            "📅 Mês da Simulação", 
            options=range(1, 13), 
            index=int(dados['Mes'].iloc[-1]) - 1,
            format_func=lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'),
            help="Escolha o mês para considerar sazonalidade"
        )
        taxa_juros = st.number_input(
            "📈 Taxa SELIC Atual (%)", 
            min_value=0.0, 
            max_value=30.0, 
            value=float(dados['Taxa_Juros'].iloc[-1]), 
            step=0.1,
            format='%.2f',
            help="Taxa básica de juros da economia"
        )
        indice_acidentes = st.number_input(
            "🚗 Índice de Acidentes", 
            min_value=50.0, 
            max_value=150.0, 
            value=float(dados['Indice_Acidentes'].iloc[-1]),
            step=1.0,
            format='%.1f',
            help="Índice de sinistralidade do mercado (Base 100 = média)"
        )

    st.divider()
    
    # Botão de Previsão com destaque
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        calcular_btn = st.button("🚀 CALCULAR PROJEÇÃO DE FATURAMENTO", type="primary", use_container_width=True)
    
    if calcular_btn:
        
        # Preparar inputs
        inputs = {
            'Faturamento_Mes_Ant': fat_ant,
            'Qtd_Atendimentos': qtd_atendimentos,
            'Ticket_Medio': ticket_medio,
            'Perc_Atend_Com_Pecas': perc_pecas,
            'Tempo_Medio_Atend_Horas': tempo_atend,
            'Taxa_Reincidencia': taxa_reincidencia,
            'Sinistralidade_Mes_Ant': sinistralidade_ant,
            'NPS': nps,
            'Taxa_Juros': taxa_juros,
            'Indice_Acidentes': indice_acidentes,
            'mes_prev': mes_prev
        }
        
        # Fazer previsão
        predicao = fazer_previsao(modelo, feature_names, inputs)
        
        # Calcular métricas derivadas
        metricas_derivadas = calcular_metricas_derivadas(predicao, sinistralidade_ant, qtd_atendimentos, ticket_medio)
        
        # Exibição dos Resultados com destaque visual
        st.divider()
        
        # Banner de resultado
        variacao_percentual = ((predicao - fat_ant) / fat_ant * 100) if fat_ant > 0 else 0
        emoji_resultado = "🎉" if variacao_percentual > 0 else "📉" if variacao_percentual < 0 else "➡️"
        
        st.markdown(f"""
        <div class="{'success-box' if variacao_percentual >= 0 else 'alert-box'}">
            <h2 style="margin-top: 0; text-align: center;">{emoji_resultado} Projeção de Faturamento</h2>
            <h1 style="text-align: center; font-size: 3.5rem; margin: 20px 0;">
                R$ {predicao:,.2f}
            </h1>
            <p style="text-align: center; font-size: 1.3rem; margin-bottom: 0;">
                <b>{variacao_percentual:+.1f}%</b> em relação ao mês anterior
                {"📈 Crescimento esperado!" if variacao_percentual > 0 else "📊 Estabilidade mantida" if variacao_percentual == 0 else "⚠️ Atenção necessária"}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Detalhamento dos Indicadores")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric(
                label="💰 Faturamento Projetado", 
                value=f"R$ {predicao:,.2f}",
                delta=f"R$ {(predicao - fat_ant):,.2f}" if fat_ant > 0 else "N/A",
                help="Valor total esperado de faturamento para o próximo período"
            )
        
        with col_m2:
            st.metric(
                label="💵 Margem Bruta Estimada", 
                value=f"R$ {metricas_derivadas['margem_bruta']:,.2f}",
                delta=f"{((metricas_derivadas['margem_bruta'] / predicao) * 100):.1f}%" if predicao > 0 else "0%",
                help="Lucro bruto esperado após deduzir custos de sinistros"
            )
        
        with col_m3:
            st.metric(
                label="🎫 Ticket Real por Atendimento", 
                value=f"R$ {metricas_derivadas['ticket_real']:,.2f}",
                delta=f"{((metricas_derivadas['ticket_real'] - ticket_medio) / ticket_medio * 100):.1f}%" if ticket_medio > 0 else "0%",
                help="Valor médio real que cada atendimento gerará"
            )
        
        with col_m4:
            variacao_meta = sinistralidade_ant - 50.0
            st.metric(
                label="⚠️ Custo de Sinistros", 
                value=f"{sinistralidade_ant:.1f}%",
                delta=f"{variacao_meta:+.1f}% vs Meta (50%)",
                delta_color="inverse",
                help="Percentual de custo com sinistros - meta ideal é 50%"
            )
        
        st.divider()
        
        # Análise detalhada com storytelling
        col_a1, col_a2 = st.columns([2, 1])
        
        with col_a1:
            st.markdown("### 💡 Análise Inteligente do Cenário")
            
            # Status da sinistralidade com linguagem comercial
            status_sin, cor_sin = determinar_status_sinistralidade(sinistralidade_ant)
            
            if cor_sin == "success":
                st.success(f"✅ **Excelente!** {status_sin}")
            elif cor_sin == "warning":
                st.warning(f"⚠️ **Atenção!** {status_sin}")
            else:
                st.error(f"🚨 **Crítico!** {status_sin}")
            
            # Storytelling do cenário
            st.markdown(f"""
            <div class="insight-box">
            <h4>📖 História do seu Cenário</h4>
            <p style="font-size: 1.05rem; line-height: 1.6;">
            No mês de <b>{pd.to_datetime(str(mes_prev), format='%m').strftime('%B')}</b>, 
            sua operação atenderá aproximadamente <b>{qtd_atendimentos:,} clientes</b>, 
            sendo que <b>{int(qtd_atendimentos * perc_pecas / 100):,} deles</b> ({perc_pecas:.0f}%) 
            precisarão de reposição de peças.
            </p>
            <p style="font-size: 1.05rem; line-height: 1.6;">
            Com um tempo médio de resolução de <b>{tempo_atend:.1f} horas</b> e uma satisfação 
            de <b>{nps} pontos no NPS</b>, o valor médio por atendimento será de 
            <b>R$ {metricas_derivadas['ticket_real']:,.2f}</b>.
            </p>
            <p style="font-size: 1.05rem; line-height: 1.6;">
            Os custos com sinistros representarão <b>{sinistralidade_ant:.1f}%</b> do faturamento 
            (meta: 50%), resultando em uma margem bruta de <b>R$ {metricas_derivadas['margem_bruta']:,.2f}</b>.
            </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🎯 Recomendações Estratégicas")
            
            # Gerar recomendações
            recomendacoes = gerar_recomendacoes(tempo_atend, taxa_reincidencia, nps, 
                                                sinistralidade_ant, sinistralidade_orcada, perc_pecas)
            
            if not recomendacoes:
                st.markdown("""
                <div class="success-box">
                    <h4>🌟 Parabéns! Operação Otimizada</h4>
                    <p>Todos os seus indicadores estão dentro dos padrões ideais. 
                    Continue monitorando para manter a excelência!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                for i, rec in enumerate(recomendacoes, 1):
                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; 
                         margin: 10px 0; border-left: 4px solid #667eea;">
                        <b>💡 Ação {i}:</b> {rec}
                    </div>
                    """, unsafe_allow_html=True)
        
        with col_a2:
            st.markdown("### 🎯 Indicador de Performance")
            
            # Gauge de sinistralidade
            fig_gauge = criar_gauge_sinistralidade(sinistralidade_ant)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown("""
            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px;">
                <h4>🏆 Metas de Excelência</h4>
                <ul style="line-height: 1.8;">
                    <li><b>Custo Sinistros:</b> ≤ 50% 🎯</li>
                    <li><b>Satisfação (NPS):</b> > 70 😊</li>
                    <li><b>Tempo Resposta:</b> < 3h ⏱️</li>
                    <li><b>Retorno Cliente:</b> < 10% 🔄</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# --- TAB 2: ANÁLISE HISTÓRICA ---
with tab2:
    st.markdown("""
    <div class="insight-box">
        <h2 style="margin-top: 0;">📈 Análise Histórica: A Jornada dos seus Resultados</h2>
        <p style="font-size: 1.1rem;">
            Entenda como sua operação evoluiu ao longo do tempo. 
            Identifique <b>padrões, tendências e oportunidades</b> de melhoria.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        fig_fat = criar_grafico_faturamento(dados)
        st.plotly_chart(fig_fat, use_container_width=True)
    
    with col_g2:
        fig_sin = criar_grafico_sinistralidade(dados)
        st.plotly_chart(fig_sin, use_container_width=True)
    
    col_g3, col_g4 = st.columns(2)
    
    with col_g3:
        fig_atend = criar_grafico_atendimentos(dados)
        st.plotly_chart(fig_atend, use_container_width=True)
    
    with col_g4:
        fig_ticket = criar_grafico_ticket_medio(dados)
        st.plotly_chart(fig_ticket, use_container_width=True)
    
    # Análise de sazonalidade
    st.markdown("### 📅 Entendendo a Sazonalidade do Negócio")
    st.info("💡 **Por que isso importa?** Saber quando seu negócio tem mais demanda ajuda a planejar estoque, equipe e campanhas de marketing!")
    
    dados_sazon = preparar_dados_sazonalidade(dados)
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        fig_sazon_fat = criar_grafico_sazonalidade(dados_sazon, tipo='faturamento')
        st.plotly_chart(fig_sazon_fat, use_container_width=True)
    
    with col_s2:
        fig_sazon_atend = criar_grafico_sazonalidade(dados_sazon, tipo='atendimentos')
        st.plotly_chart(fig_sazon_atend, use_container_width=True)

# --- TAB 3: DASHBOARD DE KPIs ---
with tab3:
    st.markdown("""
    <div class="insight-box">
        <h2 style="margin-top: 0;">📊 Painel de Indicadores: Sua Operação em Números</h2>
        <p style="font-size: 1.1rem;">
            Visão consolidada dos principais <b>indicadores de performance</b> do seu negócio. 
            Compare com metas e identifique áreas de destaque!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPIs principais
    col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
    
    with col_k1:
        st.metric(
            "💰 Faturamento Médio", 
            f"R$ {dados['Faturamento'].mean():,.2f}",
            delta=f"±{dados['Faturamento'].std():,.2f}"
        )
    
    with col_k2:
        st.metric(
            "📊 Sinistralidade Realizada Média", 
            f"{dados['Sinistralidade_Realizada'].mean():.1f}%",
            delta=f"{(dados['Sinistralidade_Realizada'].mean() - 50):.1f}% vs meta (50%)",
            delta_color="inverse"
        )
    
    with col_k3:
        st.metric(
            "📞 Atendimentos/Mês", 
            f"{dados['Qtd_Atendimentos'].mean():,.0f}",
            delta=f"±{dados['Qtd_Atendimentos'].std():,.0f}"
        )
    
    with col_k4:
        st.metric(
            "🎫 Ticket Médio", 
            f"R$ {dados['Ticket_Medio'].mean():,.2f}",
            delta=f"±{dados['Ticket_Medio'].std():,.2f}"
        )
    
    with col_k5:
        st.metric(
            "😊 NPS Médio", 
            f"{dados['NPS'].mean():.0f}/100",
            delta=f"{(dados['NPS'].mean() - 70):.0f} vs meta"
        )
    
    st.divider()
    
    # Análise comparativa de Sinistralidade
    st.markdown("### 💰 Controle de Custos: Planejado vs Realizado")
    st.info("📌 **Insight Importante:** O gráfico abaixo compara o que você planejou gastar com o que realmente gastou. Quanto mais próximos, melhor seu controle financeiro!")
    
    col_sin1, col_sin2 = st.columns(2)
    
    with col_sin1:
        fig_comp_sin = criar_grafico_comparativo_sinistralidade(dados)
        st.plotly_chart(fig_comp_sin, use_container_width=True)
    
    with col_sin2:
        st.markdown("#### 📈 Estatísticas de Sinistralidade")
        
        stats = calcular_estatisticas_sinistralidade(dados)
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            st.metric("Realizada Média", f"{stats['media_realizada']:.1f}%")
            st.metric("Orçada Média", f"{stats['media_orcada']:.1f}%")
        
        with col_stat2:
            st.metric("Realizada Mínima", f"{stats['min_realizada']:.1f}%")
            st.metric("Realizada Máxima", f"{stats['max_realizada']:.1f}%")
        
        with col_stat3:
            st.metric("Meses na Meta", f"{stats['meses_dentro_meta']}")
            st.metric("% na Meta", f"{stats['perc_dentro_meta']:.1f}%")
        
        if stats['desvio_medio'] > 0:
            st.warning(f"⚠️ Sinistralidade realizada está em média **{stats['desvio_medio']:.1f}%** acima do orçado")
        else:
            st.success(f"✅ Sinistralidade realizada está em média **{abs(stats['desvio_medio']):.1f}%** abaixo do orçado")
        
        # Distribuição
        fig_dist_sin = px.histogram(
            dados, 
            x='Sinistralidade_Realizada', 
            nbins=15,
            title='Distribuição da Sinistralidade Realizada',
            labels={'Sinistralidade_Realizada': 'Sinistralidade (%)'}
        )
        fig_dist_sin.add_vline(x=50, line_dash="dash", line_color="green", annotation_text="Meta: 50%")
        st.plotly_chart(fig_dist_sin, use_container_width=True)
    
    st.divider()
    
    # Correlações entre variáveis
    st.markdown("### 🔗 Conexões entre Indicadores")
    
    st.markdown("""
    <div class="insight-box">
        <h4>🧩 Como os Indicadores se Relacionam?</h4>
        <p style="font-size: 1.05rem;">
            Este mapa mostra quais indicadores andam juntos. 
            <b>Cores quentes (vermelho)</b> = quando um sobe, o outro também sobe. 
            <b>Cores frias (azul)</b> = quando um sobe, o outro desce.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    fig_corr = criar_matriz_correlacao(dados, VARIAVEIS_CORRELACAO)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Distribuições
    st.subheader("📊 Distribuição de Variáveis Chave")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        fig_hist_fat = px.histogram(
            dados, 
            x='Faturamento', 
            nbins=20,
            title='Distribuição do Faturamento',
            labels={'Faturamento': 'Faturamento (R$)'}
        )
        st.plotly_chart(fig_hist_fat, use_container_width=True)
    
    with col_d2:
        fig_hist_sin = px.histogram(
            dados, 
            x='Sinistralidade_Realizada', 
            nbins=20,
            title='Distribuição da Sinistralidade Realizada',
            labels={'Sinistralidade_Realizada': 'Sinistralidade (%)'}
        )
        fig_hist_sin.add_vline(x=50, line_dash="dash", line_color="green", annotation_text="Meta: 50%")
        st.plotly_chart(fig_hist_sin, use_container_width=True)

# --- TAB 4: INSIGHTS DO MODELO ---
with tab4:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
         padding: 40px; border-radius: 15px; color: white; text-align: center; margin-bottom: 30px;">
        <h1 style="margin: 0; color: white; font-size: 2.5rem;">🎯 Inteligência Estratégica</h1>
        <p style="font-size: 1.3rem; margin: 15px 0 0 0; opacity: 0.95;">
            Descubra as <b>alavancas de crescimento</b> do seu negócio
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Criar DataFrame com coeficientes
    df_coef = pd.DataFrame({
        'Variável': feature_names,
        'Coeficiente': modelo.coef_
    }).sort_values(by='Coeficiente', key=abs, ascending=False)
    
    # Separar variáveis de mês das outras
    df_coef_vars = df_coef[~df_coef['Variável'].str.startswith('Mes_')]
    df_coef_meses = df_coef[df_coef['Variável'].str.startswith('Mes_')]
    
    # Traduzir nomes das variáveis para linguagem comercial
    traducao_vars = {
        'Faturamento_Mes_Ant': 'Histórico de Faturamento',
        'Qtd_Atendimentos': 'Volume de Atendimentos',
        'Ticket_Medio': 'Valor Médio por Atendimento',
        'Perc_Atend_Com_Pecas': '% Atendimentos com Peças',
        'Tempo_Medio_Atend_Horas': 'Tempo de Resolução',
        'Taxa_Reincidencia': 'Taxa de Retorno do Cliente',
        'Sinistralidade_Mes_Ant': 'Custo de Sinistros',
        'NPS': 'Satisfação do Cliente (NPS)',
        'Taxa_Juros': 'Taxa SELIC',
        'Indice_Acidentes': 'Índice de Acidentes'
    }
    
    df_coef_vars = df_coef_vars.copy()
    df_coef_vars['Variável_Traduzida'] = df_coef_vars['Variável'].map(
        lambda x: traducao_vars.get(x, x.replace('_', ' '))
    )
    
    # ==== SEÇÃO 1: RESUMO EXECUTIVO ====
    st.markdown("## 📊 Resumo Executivo")
    
    col_exec1, col_exec2, col_exec3 = st.columns(3)
    
    total_positivos = len(df_coef_vars[df_coef_vars['Coeficiente'] > 0])
    total_negativos = len(df_coef_vars[df_coef_vars['Coeficiente'] < 0])
    impacto_total = df_coef_vars['Coeficiente'].abs().sum()
    
    with col_exec1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%); 
             padding: 25px; border-radius: 12px; text-align: center; height: 180px; display: flex; flex-direction: column; justify-content: center;">
            <h1 style="color: #2d5016; margin: 0; font-size: 3rem;">{total_positivos}</h1>
            <p style="color: #2d5016; font-size: 1.1rem; margin: 10px 0 0 0; font-weight: 600;">
                Fatores de Crescimento
            </p>
            <p style="color: #4a7c24; font-size: 0.9rem; margin: 5px 0 0 0;">
                Ações para aumentar
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_exec2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
             padding: 25px; border-radius: 12px; text-align: center; height: 180px; display: flex; flex-direction: column; justify-content: center;">
            <h1 style="color: #8b3a00; margin: 0; font-size: 3rem;">{total_negativos}</h1>
            <p style="color: #8b3a00; font-size: 1.1rem; margin: 10px 0 0 0; font-weight: 600;">
                Fatores de Risco
            </p>
            <p style="color: #b84c00; font-size: 0.9rem; margin: 5px 0 0 0;">
                Ações para reduzir
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_exec3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
             padding: 25px; border-radius: 12px; text-align: center; height: 180px; display: flex; flex-direction: column; justify-content: center;">
            <h1 style="color: #1a237e; margin: 0; font-size: 2.5rem;">R$ {impacto_total:,.0f}</h1>
            <p style="color: #1a237e; font-size: 1.1rem; margin: 10px 0 0 0; font-weight: 600;">
                Impacto Total
            </p>
            <p style="color: #3949ab; font-size: 0.9rem; margin: 5px 0 0 0;">
                Soma dos impactos
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # ==== SEÇÃO 2: TOP 3 PRIORIDADES ====
    st.markdown("## 🏆 Suas 3 Principais Prioridades")
    
    st.markdown("""
    <div class="insight-box">
        <p style="font-size: 1.15rem; margin: 0;">
            Concentre-se nestas <b>3 áreas</b> para gerar o <b>maior impacto</b> nos resultados. 
            São os fatores que, quando otimizados, trazem o melhor retorno sobre o esforço.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    top_3 = df_coef_vars.head(3)
    
    for i, row in enumerate(top_3.itertuples(), 1):
        emoji_medalha = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        eh_positivo = row.Coeficiente > 0
        cor_principal = "#00c853" if eh_positivo else "#ff6b6b"
        icone_acao = "📈" if eh_positivo else "📉"
        verbo = "aumentar" if eh_positivo else "reduzir"
        
        # Calcular impacto percentual relativo
        impacto_percentual = (abs(row.Coeficiente) / impacto_total) * 100
        
        col_p1, col_p2 = st.columns([2, 1])
        
        with col_p1:
            # Container com cor de fundo
            if eh_positivo:
                st.success(f"**{emoji_medalha} PRIORIDADE #{i}: {row.Variável_Traduzida}**")
            else:
                st.error(f"**{emoji_medalha} PRIORIDADE #{i}: {row.Variável_Traduzida}**")
            
            st.markdown(f"**Representa {impacto_percentual:.1f}%** do impacto total")
            
            st.markdown(f"""
            ---
            #### {icone_acao} Plano de Ação
            
            **Objetivo:** {verbo.upper()} este indicador  
            **Impacto:** Cada unidade que você {verbo} gera **R$ {abs(row.Coeficiente):,.2f}** de {"aumento" if eh_positivo else "redução"} no faturamento  
            **Potencial:** {"Alto potencial de crescimento 🚀" if eh_positivo else "Alto risco se não controlado ⚠️"}
            """)
        
        with col_p2:
            # Criar gauge visual do impacto
            fig_gauge_priority = go.Figure(go.Indicator(
                mode="gauge+number",
                value=impacto_percentual,
                title={'text': "Relevância", 'font': {'size': 16}},
                number={'suffix': "%", 'font': {'size': 28}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': cor_principal},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 33], 'color': '#e0e0e0'},
                        {'range': [33, 66], 'color': '#bdbdbd'},
                        {'range': [66, 100], 'color': '#9e9e9e'}
                    ]
                }
            ))
            fig_gauge_priority.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_gauge_priority, width='stretch')
            
            # Dica estratégica
            if eh_positivo:
                st.info("💡 **Dica:** Invista recursos para maximizar este fator. Pequenas melhorias aqui geram grandes resultados!")
            else:
                st.warning("💡 **Dica:** Monitore de perto e implemente controles. Reduções neste fator protegem sua margem!")
        
        st.divider()
    
    st.divider()
    
    # ==== SEÇÃO 3: RANKING COMPLETO INTERATIVO ====
    st.markdown("## 📋 Ranking Completo de Fatores")
    
    # Criar visualização mais atraente dos coeficientes
    df_coef_vars_display = df_coef_vars.copy()
    df_coef_vars_display['Impacto'] = df_coef_vars_display['Coeficiente'].apply(
        lambda x: '📈 Positivo' if x > 0 else '📉 Negativo'
    )
    df_coef_vars_display['Valor_Abs'] = df_coef_vars_display['Coeficiente'].abs()
    df_coef_vars_display['Prioridade'] = range(1, len(df_coef_vars_display) + 1)
    
    # Gráfico interativo
    fig_ranking = go.Figure()
    
    # Barras positivas
    positivos = df_coef_vars_display[df_coef_vars_display['Coeficiente'] > 0]
    fig_ranking.add_trace(go.Bar(
        y=positivos['Variável_Traduzida'],
        x=positivos['Coeficiente'],
        name='Fatores de Crescimento',
        orientation='h',
        marker=dict(
            color='#00c853',
            line=dict(color='#00a152', width=2)
        ),
        text=positivos['Coeficiente'].apply(lambda x: f'+R$ {x:,.0f}'),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Impacto: +R$ %{x:,.2f}<br><extra></extra>'
    ))
    
    # Barras negativas
    negativos = df_coef_vars_display[df_coef_vars_display['Coeficiente'] < 0]
    fig_ranking.add_trace(go.Bar(
        y=negativos['Variável_Traduzida'],
        x=negativos['Coeficiente'],
        name='Fatores de Risco',
        orientation='h',
        marker=dict(
            color='#ff6b6b',
            line=dict(color='#d32f2f', width=2)
        ),
        text=negativos['Coeficiente'].apply(lambda x: f'R$ {x:,.0f}'),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Impacto: R$ %{x:,.2f}<br><extra></extra>'
    ))
    
    fig_ranking.update_layout(
        title={
            'text': '🎯 Impacto de Cada Fator no Faturamento',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#333', 'family': 'Arial Black'}
        },
        xaxis_title='Impacto em R$ no Faturamento',
        yaxis_title='',
        barmode='relative',
        height=max(400, len(df_coef_vars_display) * 40),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    fig_ranking.add_vline(x=0, line_width=2, line_dash="solid", line_color="black")
    
    st.plotly_chart(fig_ranking, use_container_width=True)
    
    # Tabela estilizada
    st.markdown("### 📊 Tabela Detalhada")
    
    tabela_display = df_coef_vars_display[[
        'Prioridade', 'Variável_Traduzida', 'Coeficiente', 'Impacto'
    ]].copy()
    tabela_display.columns = ['#', 'Fator', 'Impacto (R$)', 'Tipo']
    
    st.dataframe(
        tabela_display.style.format({
            'Impacto (R$)': 'R$ {:,.2f}'
        }).background_gradient(
            subset=['Impacto (R$)'], 
            cmap='RdYlGn',
            vmin=-abs(df_coef_vars_display['Coeficiente']).max(),
            vmax=abs(df_coef_vars_display['Coeficiente']).max()
        ).map(
            lambda x: 'background-color: #e8f5e9' if x == '📈 Positivo' else 'background-color: #ffebee',
            subset=['Tipo']
        ),
        width=None,
        height=400
    )
    
    st.divider()
    
    # ==== SEÇÃO 4: SAZONALIDADE ====
    if len(df_coef_meses) > 0:
        st.markdown("## 📅 Efeito da Sazonalidade")
        
        st.info("""
        💡 **Entenda os meses:** Alguns meses naturalmente trazem mais ou menos faturamento. 
        Use isso para planejar campanhas, ajustar estoque e preparar a equipe!
        """)
        
        # Traduzir nomes dos meses
        meses_map = {
            'Mes_1': 'Janeiro', 'Mes_2': 'Fevereiro', 'Mes_3': 'Março',
            'Mes_4': 'Abril', 'Mes_5': 'Maio', 'Mes_6': 'Junho',
            'Mes_7': 'Julho', 'Mes_8': 'Agosto', 'Mes_9': 'Setembro',
            'Mes_10': 'Outubro', 'Mes_11': 'Novembro', 'Mes_12': 'Dezembro'
        }
        
        df_coef_meses = df_coef_meses.copy()
        df_coef_meses['Mês'] = df_coef_meses['Variável'].map(meses_map)
        df_coef_meses_sorted = df_coef_meses.sort_values('Coeficiente', ascending=False)
        
        # Identificar melhor e pior mês
        melhor_mes = df_coef_meses_sorted.iloc[0]
        pior_mes = df_coef_meses_sorted.iloc[-1]
        
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%); 
                 padding: 25px; border-radius: 12px; text-align: center;">
                <h1 style="font-size: 3rem; margin: 0;">🏆</h1>
                <h3 style="margin: 10px 0; color: #8b6914;">Melhor Mês</h3>
                <h2 style="margin: 5px 0; color: #5d4a0f;">{melhor_mes['Mês']}</h2>
                <p style="margin: 5px 0; color: #8b6914; font-size: 1.1rem;">
                    +R$ {melhor_mes['Coeficiente']:,.0f}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #c0c0c0 0%, #d9d9d9 100%); 
                 padding: 25px; border-radius: 12px; text-align: center;">
                <h1 style="font-size: 3rem; margin: 0;">📊</h1>
                <h3 style="margin: 10px 0; color: #5a5a5a;">Variação</h3>
                <h2 style="margin: 5px 0; color: #3d3d3d;">
                    {df_coef_meses['Coeficiente'].max() - df_coef_meses['Coeficiente'].min():.0f}%
                </h2>
                <p style="margin: 5px 0; color: #5a5a5a; font-size: 1.1rem;">
                    Entre meses
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m3:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #b0bec5 0%, #cfd8dc 100%); 
                 padding: 25px; border-radius: 12px; text-align: center;">
                <h1 style="font-size: 3rem; margin: 0;">⚠️</h1>
                <h3 style="margin: 10px 0; color: #37474f;">Mês Desafiador</h3>
                <h2 style="margin: 5px 0; color: #263238;">{pior_mes['Mês']}</h2>
                <p style="margin: 5px 0; color: #37474f; font-size: 1.1rem;">
                    R$ {pior_mes['Coeficiente']:,.0f}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Gráfico de sazonalidade
        fig_sazon = go.Figure()
        
        cores_sazon = ['#00c853' if x > 0 else '#ff6b6b' for x in df_coef_meses_sorted['Coeficiente']]
        
        fig_sazon.add_trace(go.Bar(
            x=df_coef_meses_sorted['Mês'],
            y=df_coef_meses_sorted['Coeficiente'],
            marker=dict(
                color=cores_sazon,
                line=dict(color='white', width=2)
            ),
            text=df_coef_meses_sorted['Coeficiente'].apply(lambda x: f'{"+" if x > 0 else ""}R$ {x:,.0f}'),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Impacto: R$ %{y:,.2f}<br><extra></extra>'
        ))
        
        fig_sazon.update_layout(
            title={
                'text': '📅 Impacto Sazonal por Mês',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title='Mês do Ano',
            yaxis_title='Impacto no Faturamento (R$)',
            height=450,
            showlegend=False,
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='white'
        )
        
        fig_sazon.add_hline(y=0, line_width=2, line_dash="solid", line_color="black")
        
        st.plotly_chart(fig_sazon, width='stretch')
    
    st.divider()
    
    # ==== SEÇÃO 5: PLANO DE AÇÃO ESTRATÉGICO ====
    st.markdown("## 🎯 Seu Plano de Ação Estratégico")
    
    st.markdown("""
    <div class="success-box">
        <h3 style="margin-top: 0;">📋 Resumo para Tomada de Decisão</h3>
        <p style="font-size: 1.1rem; line-height: 1.7;">
            Baseado na análise completa, aqui está seu <b>roteiro de ações prioritárias</b> 
            para os próximos meses:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_plano1, col_plano2 = st.columns(2)
    
    with col_plano1:
        st.markdown("### ✅ Ações de Curto Prazo (30 dias)")
        
        top_positivo = df_coef_vars[df_coef_vars['Coeficiente'] > 0].iloc[0]
        top_negativo = df_coef_vars[df_coef_vars['Coeficiente'] < 0].iloc[0]
        
        st.markdown(f"""
        <div style="background: white; padding: 20px; border-radius: 10px; border: 2px solid #00c853;">
            <h4 style="color: #00c853; margin-top: 0;">📈 Maximizar</h4>
            <p style="margin: 10px 0;"><b>{top_positivo['Variável_Traduzida']}</b></p>
            <p style="margin: 0; color: #666;">
                Foque em aumentar este indicador. Estabeleça uma meta de crescimento 
                de 5-10% e monitore semanalmente.
            </p>
        </div>
        
        <div style="background: white; padding: 20px; border-radius: 10px; border: 2px solid #ff6b6b; margin-top: 15px;">
            <h4 style="color: #ff6b6b; margin-top: 0;">📉 Controlar</h4>
            <p style="margin: 10px 0;"><b>{top_negativo['Variável_Traduzida']}</b></p>
            <p style="margin: 0; color: #666;">
                Implemente controles rígidos. Reduza este indicador em 3-5% 
                para proteger sua margem.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_plano2:
        st.markdown("### 🎯 Ações de Médio Prazo (90 dias)")
        
        segundo_positivo = df_coef_vars[df_coef_vars['Coeficiente'] > 0].iloc[1] if len(df_coef_vars[df_coef_vars['Coeficiente'] > 0]) > 1 else top_positivo
        
        st.markdown(f"""
        <div style="background: white; padding: 20px; border-radius: 10px; border: 2px solid #667eea;">
            <h4 style="color: #667eea; margin-top: 0;">🚀 Investir</h4>
            <p style="margin: 10px 0;"><b>{segundo_positivo['Variável_Traduzida']}</b></p>
            <p style="margin: 0; color: #666;">
                Aloque recursos para melhorar este fator. Treine equipe, 
                otimize processos e acompanhe resultados mensalmente.
            </p>
        </div>
        
        <div style="background: white; padding: 20px; border-radius: 10px; border: 2px solid #ffa726; margin-top: 15px;">
            <h4 style="color: #ffa726; margin-top: 0;">📊 Monitorar</h4>
            <p style="margin: 10px 0;"><b>Todos os Indicadores</b></p>
            <p style="margin: 0; color: #666;">
                Crie um dashboard de acompanhamento semanal. Compare real vs planejado 
                e ajuste estratégias conforme necessário.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Call to action final
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
         padding: 30px; border-radius: 15px; color: white; text-align: center; margin-top: 30px;">
        <h3 style="margin: 0 0 15px 0; color: white;">💡 Próximo Passo</h3>
        <p style="font-size: 1.2rem; margin: 0; line-height: 1.6;">
            Use o <b>Simulador de Cenários</b> (primeira aba) para testar diferentes combinações 
            destes fatores e encontrar a melhor estratégia para seu negócio!
        </p>
    </div>
    """, unsafe_allow_html=True)
