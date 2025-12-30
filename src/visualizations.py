"""
Módulo para criação de visualizações e gráficos.
Focado em UX e Storytelling Visual
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Paleta de cores consistente para storytelling
COLORS = {
    'primary': '#1f77b4',      # Azul principal
    'success': '#2ca02c',      # Verde sucesso
    'warning': '#ff7f0e',      # Laranja atenção
    'danger': '#d62728',       # Vermelho alerta
    'info': '#17becf',         # Azul info
    'secondary': '#7f7f7f',    # Cinza
    'purple': '#9467bd',       # Roxo
    'meta': '#2ca02c'          # Verde meta
}

# Configurações de estilo padrão
PLOT_BG = 'rgba(0,0,0,0)'
FONT_FAMILY = 'Arial, sans-serif'
GRID_COLOR = 'rgba(128,128,128,0.2)'

def aplicar_estilo_padrao(fig):
    """Aplica estilo padrão aos gráficos"""
    fig.update_layout(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PLOT_BG,
        font={'family': FONT_FAMILY, 'size': 12, 'color': '#2c3e50'},
        hoverlabel={'bgcolor': 'white', 'font_size': 13}
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=GRID_COLOR, showline=True, linewidth=2, linecolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=GRID_COLOR, showline=True, linewidth=2, linecolor=GRID_COLOR)
    return fig


def criar_grafico_faturamento(dados):
    """Cria gráfico de evolução do faturamento."""
    fig = px.line(
        dados, 
        x='Data', 
        y='Faturamento', 
        title='Evolução do Faturamento',
        labels={'Faturamento': 'Faturamento (R$)', 'Data': 'Período'}
    )
    fig.update_traces(line_color='#1f77b4', line_width=2)
    return fig


def criar_grafico_sinistralidade(dados):
    """Cria gráfico de sinistralidade com storytelling visual - Realizada, Orçada e Meta."""
    fig = go.Figure()
    
    # Área de zona segura (abaixo de 50%)
    fig.add_hrect(
        y0=0, y1=50,
        fillcolor=COLORS['success'],
        opacity=0.1,
        layer='below',
        line_width=0,
        annotation_text='Zona Segura',
        annotation_position='top left'
    )
    
    # Área de atenção (50-60%)
    fig.add_hrect(
        y0=50, y1=60,
        fillcolor=COLORS['warning'],
        opacity=0.1,
        layer='below',
        line_width=0,
        annotation_text='Atenção',
        annotation_position='top left'
    )
    
    # Área crítica (acima de 60%)
    fig.add_hrect(
        y0=60, y1=dados['Sinistralidade_Realizada'].max() + 5,
        fillcolor=COLORS['danger'],
        opacity=0.1,
        layer='below',
        line_width=0,
        annotation_text='Crítico',
        annotation_position='top left'
    )
    
    # Linha de Sinistralidade Orçada (primeira, mais sutil)
    fig.add_trace(go.Scatter(
        x=dados['Data'], 
        y=dados['Sinistralidade_Orcada'],
        mode='lines',
        name='📋 Orçada (Planejada)',
        line=dict(color=COLORS['secondary'], width=2, dash='dot'),
        hovertemplate='<b>Orçada</b>: %{y:.1f}%<extra></extra>'
    ))
    
    # Linha de Sinistralidade Realizada (destaque principal)
    fig.add_trace(go.Scatter(
        x=dados['Data'], 
        y=dados['Sinistralidade_Realizada'],
        mode='lines+markers',
        name='📈 Realizada (Efetiva)',
        line=dict(color=COLORS['danger'], width=3),
        marker=dict(size=4, color=COLORS['danger']),
        fill='tonexty',
        fillcolor='rgba(214, 39, 40, 0.1)',
        hovertemplate='<b>Realizada</b>: %{y:.1f}%<br><i>Desvio da meta: %{customdata:.1f} pontos</i><extra></extra>',
        customdata=dados['Sinistralidade_Realizada'] - 50
    ))
    
    # Linha da Meta (50%)
    fig.add_hline(
        y=50, 
        line_dash="dash", 
        line_color=COLORS['meta'],
        line_width=3,
        annotation_text="🎯 Meta: 50%",
        annotation_position="right",
        annotation=dict(font_size=14, font_color=COLORS['meta'], bgcolor='white')
    )
    
    fig.update_layout(
        title={
            'text': '📊 Jornada da Sinistralidade: Planejado vs Realizado',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2c3e50', 'family': 'Arial Black'}
        },
        xaxis_title='Período',
        yaxis_title='Sinistralidade (%)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        ),
        height=450
    )
    
    # Adicionar anotação do último valor
    ultimo_valor = dados['Sinistralidade_Realizada'].iloc[-1]
    ultima_data = dados['Data'].iloc[-1]
    
    fig.add_annotation(
        x=ultima_data,
        y=ultimo_valor,
        text=f'{ultimo_valor:.1f}%',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor=COLORS['danger'],
        ax=40,
        ay=-40,
        bgcolor='white',
        bordercolor=COLORS['danger'],
        borderwidth=2,
        font=dict(size=14, color=COLORS['danger'])
    )
    
    return fig


def criar_grafico_atendimentos(dados):
    """Cria gráfico de volume de atendimentos."""
    fig = px.bar(
        dados, 
        x='Data', 
        y='Qtd_Atendimentos', 
        title='Volume de Atendimentos Mensal',
        labels={'Qtd_Atendimentos': 'Quantidade', 'Data': 'Período'}
    )
    return fig


def criar_grafico_nps(dados):
    """Cria gráfico de evolução do NPS com zonas de classificação."""
    fig = go.Figure()
    
    # Zonas de classificação NPS
    fig.add_hrect(y0=0, y1=50, fillcolor='rgba(214, 39, 40, 0.1)', layer='below', line_width=0,
                  annotation_text='Zona Crítica', annotation_position='top left')
    fig.add_hrect(y0=50, y1=70, fillcolor='rgba(255, 127, 14, 0.1)', layer='below', line_width=0,
                  annotation_text='Zona de Aperfeiçoamento', annotation_position='top left')
    fig.add_hrect(y0=70, y1=100, fillcolor='rgba(44, 160, 44, 0.1)', layer='below', line_width=0,
                  annotation_text='Zona de Excelência', annotation_position='top left')
    
    # Linha do NPS com gradiente de cores
    cores_nps = [COLORS['danger'] if v < 50 else COLORS['warning'] if v < 70 else COLORS['success'] 
                 for v in dados['NPS']]
    
    fig.add_trace(go.Scatter(
        x=dados['Data'],
        y=dados['NPS'],
        mode='lines+markers',
        name='😊 NPS',
        line=dict(color=COLORS['purple'], width=3),
        marker=dict(size=8, color=cores_nps, line=dict(width=2, color='white')),
        hovertemplate='<b>NPS</b>: %{y:.0f}<br><i>%{customdata}</i><extra></extra>',
        customdata=['Crítico' if v < 50 else 'Aperfeiçoar' if v < 70 else 'Excelente' for v in dados['NPS']]
    ))
    
    # Meta
    fig.add_hline(
        y=70,
        line_dash='dash',
        line_color=COLORS['success'],
        line_width=3,
        annotation_text='🎯 Meta: 70',
        annotation_position='right',
        annotation=dict(font_size=14, bgcolor='white')
    )
    
    fig.update_layout(
        title={
            'text': '😊 Satisfação do Cliente (NPS): Jornada para a Excelência',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Período',
        yaxis_title='Net Promoter Score',
        height=450
    )
    
    # Destacar último NPS
    ultimo_nps = dados['NPS'].iloc[-1]
    ultima_data = dados['Data'].iloc[-1]
    cor_ultimo = COLORS['danger'] if ultimo_nps < 50 else COLORS['warning'] if ultimo_nps < 70 else COLORS['success']
    
    fig.add_annotation(
        x=ultima_data,
        y=ultimo_nps,
        text=f'{ultimo_nps:.0f}',
        showarrow=True,
        arrowhead=2,
        arrowcolor=cor_ultimo,
        ax=40,
        ay=-40,
        bgcolor='white',
        bordercolor=cor_ultimo,
        borderwidth=2,
        font=dict(size=16, color=cor_ultimo)
    )
    
    return fig


def criar_grafico_ticket_medio(dados):
    """Cria gráfico de evolução do ticket médio."""
    fig = px.line(
        dados, 
        x='Data', 
        y='Ticket_Medio', 
        title='Evolução do Ticket Médio (R$)',
        labels={'Ticket_Medio': 'Ticket Médio (R$)', 'Data': 'Período'}
    )
    fig.update_traces(line_color='#2ca02c', line_width=2)
    return fig


def criar_grafico_sazonalidade(dados_sazon, tipo='faturamento'):
    """Cria gráfico de sazonalidade."""
    if tipo == 'faturamento':
        fig = px.bar(
            dados_sazon, 
            x='Mes_Nome', 
            y='Faturamento', 
            title='Faturamento Médio por Mês',
            labels={'Faturamento': 'Faturamento Médio (R$)', 'Mes_Nome': 'Mês'}
        )
    else:
        fig = px.bar(
            dados_sazon, 
            x='Mes_Nome', 
            y='Qtd_Atendimentos', 
            title='Atendimentos Médios por Mês',
            labels={'Qtd_Atendimentos': 'Quantidade Média', 'Mes_Nome': 'Mês'},
            color='Qtd_Atendimentos',
            color_continuous_scale='Viridis'
        )
    return fig


def criar_grafico_comparativo_sinistralidade(dados):
    """Cria gráfico comparativo mensal de sinistralidade."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=dados['Data'],
        y=dados['Sinistralidade_Realizada'],
        name='Realizada',
        marker_color='#d62728'
    ))
    
    fig.add_trace(go.Scatter(
        x=dados['Data'],
        y=dados['Sinistralidade_Orcada'],
        name='Orçada',
        mode='lines+markers',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    fig.add_hline(
        y=50, 
        line_dash="dash", 
        line_color="green", 
        annotation_text="Meta: 50%"
    )
    
    fig.update_layout(
        title='Comparativo Mensal: Realizada vs Orçada',
        xaxis_title='Período',
        yaxis_title='Sinistralidade (%)',
        hovermode='x unified'
    )
    
    return fig


def criar_matriz_correlacao(dados, variaveis):
    """Cria matriz de correlação."""
    corr_matrix = dados[variaveis].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title='Correlação entre Variáveis Operacionais'
    )
    return fig


def criar_gauge_sinistralidade(valor):
    """Cria gauge impactante para sinistralidade."""
    
    # Determinar cor e status
    if valor <= 50:
        cor_barra = COLORS['success']
        status = '✅ Dentro da Meta'
        cor_status = COLORS['success']
    elif valor <= 60:
        cor_barra = COLORS['warning']
        status = '⚠️ Atenção Necessária'
        cor_status = COLORS['warning']
    else:
        cor_barra = COLORS['danger']
        status = '🚨 Situação Crítica'
        cor_status = COLORS['danger']
    
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode='gauge+number+delta',
        value=valor,
        title={
            'text': f'<b>Sinistralidade Prevista</b><br><span style="font-size:16px;color:{cor_status}">{status}</span>',
            'font': {'size': 20}
        },
        delta={
            'reference': 50,
            'increasing': {'color': COLORS['danger']},
            'decreasing': {'color': COLORS['success']},
            'suffix': ' pp',
            'font': {'size': 18}
        },
        number={
            'suffix': '%',
            'font': {'size': 48, 'color': cor_barra}
        },
        gauge={
            'axis': {
                'range': [None, 100],
                'tickwidth': 2,
                'tickcolor': 'darkgray',
                'tickfont': {'size': 14}
            },
            'bar': {'color': cor_barra, 'thickness': 0.8},
            'bgcolor': 'white',
            'borderwidth': 3,
            'bordercolor': 'gray',
            'steps': [
                {'range': [0, 50], 'color': 'rgba(44, 160, 44, 0.2)', 'name': 'Meta'},
                {'range': [50, 60], 'color': 'rgba(255, 127, 14, 0.2)', 'name': 'Atenção'},
                {'range': [60, 100], 'color': 'rgba(214, 39, 40, 0.2)', 'name': 'Crítico'}
            ],
            'threshold': {
                'line': {'color': COLORS['meta'], 'width': 6},
                'thickness': 0.8,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=80, b=20),
        font={'family': 'Arial'},
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def criar_grafico_coeficientes(df_coef):
    """Cria gráfico de barras dos coeficientes do modelo."""
    fig = px.bar(
        df_coef,
        x='Coeficiente',
        y='Variável',
        orientation='h',
        title='Impacto das Variáveis no Faturamento',
        color='Coeficiente',
        color_continuous_scale='RdYlGn'
    )
    fig.update_layout(height=500)
    return fig


def criar_grafico_evolucao_faturamento(dados):
    """Cria gráfico de evolução do faturamento com storytelling visual."""
    fig = go.Figure()
    
    # Calcular média móvel para tendência
    dados_plot = dados.copy()
    dados_plot['Media_Movel'] = dados_plot['Faturamento'].rolling(window=6, center=True).mean()
    
    # Área do faturamento
    fig.add_trace(go.Scatter(
        x=dados_plot['Data'],
        y=dados_plot['Faturamento'],
        mode='lines',
        name='💰 Faturamento Mensal',
        fill='tozeroy',
        line=dict(color=COLORS['primary'], width=2),
        fillcolor='rgba(31, 119, 180, 0.3)',
        hovertemplate='<b>Faturamento</b>: R$ %{y:,.0f}<extra></extra>'
    ))
    
    # Linha de tendência (média móvel)
    fig.add_trace(go.Scatter(
        x=dados_plot['Data'],
        y=dados_plot['Media_Movel'],
        mode='lines',
        name='📈 Tendência (6 meses)',
        line=dict(color=COLORS['warning'], width=3, dash='dash'),
        hovertemplate='<b>Tendência</b>: R$ %{y:,.0f}<extra></extra>'
    ))
    
    # Linha da média geral
    media_geral = dados['Faturamento'].mean()
    fig.add_hline(
        y=media_geral,
        line_dash='dot',
        line_color=COLORS['secondary'],
        annotation_text=f'Média: R$ {media_geral:,.0f}',
        annotation_position='left'
    )
    
    fig.update_layout(
        title={
            'text': '💰 Evolução do Faturamento: Crescimento ao Longo do Tempo',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Período',
        yaxis_title='Faturamento (R$)',
        height=450
    )
    
    # Destacar último valor
    ultimo_valor = dados['Faturamento'].iloc[-1]
    ultima_data = dados['Data'].iloc[-1]
    
    fig.add_annotation(
        x=ultima_data,
        y=ultimo_valor,
        text=f'R$ {ultimo_valor:,.0f}',
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS['primary'],
        ax=-50,
        ay=-40,
        bgcolor='white',
        bordercolor=COLORS['primary'],
        borderwidth=2,
        font=dict(size=14, color=COLORS['primary'])
    )
    
    return fig


def criar_heatmap_correlacao(dados, features):
    """Cria heatmap de correlação com storytelling visual."""
    corr_matrix = dados[features].corr()
    
    # Máscar triangular superior para evitar redundância
    mask = np.triu(np.ones_like(corr_matrix), k=1)
    corr_masked = corr_matrix.where(~mask.astype(bool))
    
    # Criar texto personalizado com interpretação
    text_display = []
    for i in range(len(corr_matrix)):
        row_text = []
        for j in range(len(corr_matrix)):
            val = corr_masked.iloc[i, j]
            if pd.isna(val):
                row_text.append('')
            elif abs(val) >= 0.7:
                row_text.append(f'<b>{val:.2f}</b><br>Forte')
            elif abs(val) >= 0.4:
                row_text.append(f'{val:.2f}<br>Moderada')
            else:
                row_text.append(f'{val:.2f}')
        text_display.append(row_text)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_masked.values,
        x=[col.replace('_', ' ') for col in corr_matrix.columns],
        y=[col.replace('_', ' ') for col in corr_matrix.columns],
        colorscale=[
            [0, COLORS['danger']],
            [0.25, '#ffcccc'],
            [0.5, 'white'],
            [0.75, '#ccffcc'],
            [1, COLORS['success']]
        ],
        zmid=0,
        text=text_display,
        texttemplate='%{text}',
        textfont={'size': 9},
        colorbar=dict(
            title='Correlação<br>de Pearson',
            tickmode='linear',
            tick0=-1,
            dtick=0.5
        ),
        hovertemplate='<b>%{y}</b> × <b>%{x}</b><br>Correlação: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': '🔗 Mapa de Relacionamentos: Como as Variáveis se Conectam',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis={'side': 'bottom', 'tickangle': -45},
        yaxis={'tickangle': 0},
        height=700,
        width=900
    )
    
    return fig


def criar_boxplot_sinistralidade(dados):
    """Cria boxplot storytelling da distribuição de sinistralidade."""
    fig = go.Figure()
    
    # Boxplot principal
    fig.add_trace(go.Box(
        y=dados['Sinistralidade_Realizada'],
        name='Sinistralidade',
        marker=dict(
            color=COLORS['danger'],
            size=8,
            line=dict(color='white', width=1)
        ),
        boxmean='sd',
        fillcolor='rgba(214, 39, 40, 0.3)',
        line=dict(color=COLORS['danger'], width=2),
        hovertemplate='<b>Estatísticas:</b><br>' +
                      'Máximo: %{y:.1f}%<br>' +
                      '<extra></extra>'
    ))
    
    # Adicionar zonas de referência
    media = dados['Sinistralidade_Realizada'].mean()
    q1 = dados['Sinistralidade_Realizada'].quantile(0.25)
    q3 = dados['Sinistralidade_Realizada'].quantile(0.75)
    
    # Meta
    fig.add_hline(
        y=50,
        line_dash='dash',
        line_color=COLORS['meta'],
        line_width=3,
        annotation_text=f'🎯 Meta: 50%',
        annotation=dict(font_size=13, bgcolor='white', bordercolor=COLORS['meta'], borderwidth=2)
    )
    
    # Média
    fig.add_hline(
        y=media,
        line_dash='dot',
        line_color=COLORS['warning'],
        annotation_text=f'Média: {media:.1f}%',
        annotation_position='left'
    )
    
    fig.update_layout(
        title={
            'text': '📏 Distribuição da Sinistralidade: Entendendo a Variabilidade',
            'x': 0.5,
            'xanchor': 'center'
        },
        yaxis_title='Sinistralidade (%)',
        showlegend=False,
        height=450
    )
    
    # Anotações explicativas
    fig.add_annotation(
        x=0.5, y=q3,
        text=f'75% dos meses<br>abaixo de {q3:.1f}%',
        showarrow=True,
        arrowhead=2,
        ax=80, ay=0,
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor=COLORS['info'],
        borderwidth=1
    )
    
    return fig


def criar_boxplot_faturamento(dados):
    """Cria boxplot storytelling do faturamento."""
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y=dados['Faturamento'],
        name='Faturamento',
        marker=dict(
            color=COLORS['primary'],
            size=8,
            line=dict(color='white', width=1)
        ),
        boxmean='sd',
        fillcolor='rgba(31, 119, 180, 0.3)',
        line=dict(color=COLORS['primary'], width=2)
    ))
    
    # Média e mediana
    media = dados['Faturamento'].mean()
    mediana = dados['Faturamento'].median()
    
    fig.add_hline(
        y=media,
        line_dash='dash',
        line_color=COLORS['warning'],
        annotation_text=f'Média: R$ {media:,.0f}',
        annotation_position='right'
    )
    
    fig.update_layout(
        title={
            'text': '📊 Distribuição do Faturamento: Amplitude e Consistência',
            'x': 0.5,
            'xanchor': 'center'
        },
        yaxis_title='Faturamento (R$)',
        showlegend=False,
        height=450
    )
    
    return fig


def criar_grafico_atendimentos(dados):
    """Cria gráfico storytelling de atendimentos."""
    fig = go.Figure()
    
    # Calcular média
    media = dados['Qtd_Atendimentos'].mean()
    
    # Cores baseadas em acima/abaixo da média
    cores = [COLORS['success'] if v >= media else COLORS['warning'] for v in dados['Qtd_Atendimentos']]
    
    fig.add_trace(go.Bar(
        x=dados['Data'],
        y=dados['Qtd_Atendimentos'],
        name='Atendimentos',
        marker=dict(
            color=cores,
            line=dict(color='white', width=1)
        ),
        hovertemplate='<b>Atendimentos</b>: %{y:,.0f}<br>' +
                      '<i>%{customdata}</i><extra></extra>',
        customdata=['Acima da média' if v >= media else 'Abaixo da média' 
                    for v in dados['Qtd_Atendimentos']]
    ))
    
    # Linha da média
    fig.add_hline(
        y=media,
        line_dash='dash',
        line_color=COLORS['secondary'],
        line_width=2,
        annotation_text=f'Média: {media:,.0f}',
        annotation_position='right'
    )
    
    fig.update_layout(
        title={
            'text': '📎 Volume de Atendimentos: Capacidade e Demanda',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Período',
        yaxis_title='Quantidade de Atendimentos',
        height=450,
        showlegend=False
    )
    
    return fig


def criar_grafico_comparativo_periodos(dados, metrica, n_meses=6):
    """Cria gráfico comparativo entre períodos recentes e anteriores."""
    total = len(dados)
    
    # Dividir em dois períodos
    periodo_ant = dados.iloc[:total-n_meses].copy()
    periodo_rec = dados.iloc[-n_meses:].copy()
    
    # Calcular médias
    media_ant = periodo_ant[metrica].mean()
    media_rec = periodo_rec[metrica].mean()
    variacao = ((media_rec - media_ant) / media_ant * 100) if media_ant != 0 else 0
    
    fig = go.Figure()
    
    # Barras comparativas
    fig.add_trace(go.Bar(
        name='Período Anterior',
        x=['Período Anterior', 'Últimos 6 Meses'],
        y=[media_ant, 0],
        marker=dict(color=COLORS['secondary'], line=dict(color='white', width=2)),
        text=[f'{media_ant:,.1f}', ''],
        textposition='auto',
        width=0.4,
        hovertemplate='<b>Período Anterior</b><br>Média: %{y:,.2f}<extra></extra>'
    ))
    
    cor_recente = COLORS['success'] if variacao < 0 else COLORS['danger']
    
    fig.add_trace(go.Bar(
        name='Últimos 6 Meses',
        x=['Período Anterior', 'Últimos 6 Meses'],
        y=[0, media_rec],
        marker=dict(color=cor_recente, line=dict(color='white', width=2)),
        text=['', f'{media_rec:,.1f}'],
        textposition='auto',
        width=0.4,
        hovertemplate='<b>Últimos 6 Meses</b><br>Média: %{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': f'📊 Comparação de Períodos: {metrica.replace("_", " ")}',
            'x': 0.5,
            'xanchor': 'center'
        },
        yaxis_title=metrica.replace('_', ' '),
        barmode='group',
        height=400,
        showlegend=True
    )
    
    # Adicionar anotação da variação
    fig.add_annotation(
        x=1,
        y=max(media_ant, media_rec) * 1.1,
        text=f'Variação: {variacao:+.1f}%',
        showarrow=False,
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor=cor_recente,
        borderwidth=2,
        font=dict(size=16, color=cor_recente)
    )
    
    return fig


def criar_grafico_kpi_cards(dados):
    """Cria visualização de cards KPI com storytelling."""
    ultimo_mes = dados.iloc[-1]
    
    kpis = [
        {
            'titulo': 'Sinistralidade',
            'valor': ultimo_mes['Sinistralidade_Realizada'],
            'meta': 50,
            'formato': '{:.1f}%',
            'cor': COLORS['danger'] if ultimo_mes['Sinistralidade_Realizada'] > 50 else COLORS['success']
        },
        {
            'titulo': 'Faturamento',
            'valor': ultimo_mes['Faturamento'],
            'meta': dados['Faturamento'].mean(),
            'formato': 'R$ {:.0f}',
            'cor': COLORS['primary']
        },
        {
            'titulo': 'NPS',
            'valor': ultimo_mes['NPS'],
            'meta': 70,
            'formato': '{:.0f}',
            'cor': COLORS['success'] if ultimo_mes['NPS'] >= 70 else COLORS['warning']
        },
        {
            'titulo': 'Atendimentos',
            'valor': ultimo_mes['Qtd_Atendimentos'],
            'meta': dados['Qtd_Atendimentos'].mean(),
            'formato': '{:.0f}',
            'cor': COLORS['info']
        }
    ]
    
    fig = go.Figure()
    
    for i, kpi in enumerate(kpis):
        fig.add_trace(go.Indicator(
            mode='number+delta+gauge',
            value=kpi['valor'],
            title={'text': kpi['titulo'], 'font': {'size': 16}},
            delta={
                'reference': kpi['meta'],
                'increasing': {'color': COLORS['success']},
                'decreasing': {'color': COLORS['danger']}
            },
            number={'font': {'size': 28, 'color': kpi['cor']}},
            domain={'row': 0, 'column': i}
        ))
    
    fig.update_layout(
        grid={'rows': 1, 'columns': 4, 'pattern': 'independent'},
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def criar_grafico_importancia_features(modelo, feature_names):
    """Cria gráfico storytelling de importância das features."""
    import pandas as pd
    
    coeficientes = pd.DataFrame({
        'Feature': feature_names,
        'Coeficiente': modelo.coef_
    })
    
    coeficientes['Abs_Coef'] = abs(coeficientes['Coeficiente'])
    coeficientes = coeficientes.sort_values('Abs_Coef', ascending=True)
    
    # Nomes amigáveis
    nomes_map = {
        'Faturamento_Mes_Ant': 'Faturamento Anterior',
        'Qtd_Atendimentos': 'Volume de Atendimentos',
        'Ticket_Medio': 'Ticket Médio',
        'Perc_Atend_Com_Pecas': '% Atend. com Peças',
        'Tempo_Medio_Atend_Horas': 'Tempo de Atendimento',
        'Taxa_Reincidencia': 'Taxa de Reincidência',
        'Sinistralidade_Mes_Ant': 'Sinistralidade Anterior',
        'NPS': 'Satisfação (NPS)',
        'Taxa_Juros': 'Taxa de Juros',
        'Indice_Acidentes': 'Índice de Acidentes'
    }
    
    coeficientes['Feature_Nome'] = coeficientes['Feature'].map(nomes_map)
    
    # Cores e emojis baseados no impacto
    cores = []
    emojis = []
    for val in coeficientes['Coeficiente']:
        if val > 0:
            cores.append(COLORS['danger'])  # Aumenta sinistralidade
            emojis.append('⬆️ ')
        else:
            cores.append(COLORS['success'])  # Reduz sinistralidade
            emojis.append('⬇️ ')
    
    coeficientes['Emoji'] = emojis
    coeficientes['Label'] = coeficientes['Emoji'] + coeficientes['Feature_Nome']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=coeficientes['Coeficiente'],
        y=coeficientes['Label'],
        orientation='h',
        marker=dict(
            color=cores,
            line=dict(color='white', width=1)
        ),
        text=[f'{v:.3f}' for v in coeficientes['Coeficiente']],
        textposition='outside',
        textfont=dict(size=12, color='black'),
        hovertemplate='<b>%{y}</b><br>Impacto: %{x:.4f}<br>' +
                      '<i>%{customdata}</i><extra></extra>',
        customdata=['Aumenta sinistralidade' if v > 0 else 'Reduz sinistralidade' 
                    for v in coeficientes['Coeficiente']]
    ))
    
    fig.update_layout(
        title={
            'text': '🎯 Alavancas de Controle: O que Mais Impacta a Sinistralidade',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Impacto no Modelo (Coeficiente)',
        yaxis_title='',
        height=500,
        showlegend=False
    )
    
    # Linha zero de referência
    fig.add_vline(
        x=0,
        line_dash='solid',
        line_color='black',
        line_width=2
    )
    
    # Anotações explicativas
    fig.add_annotation(
        x=max(coeficientes['Coeficiente']) * 0.7,
        y=len(coeficientes) - 0.5,
        text='⬆️ Aumenta<br>Sinistralidade',
        showarrow=False,
        bgcolor='rgba(214, 39, 40, 0.1)',
        bordercolor=COLORS['danger'],
        borderwidth=1,
        font=dict(size=11, color=COLORS['danger'])
    )
    
    fig.add_annotation(
        x=min(coeficientes['Coeficiente']) * 0.7,
        y=len(coeficientes) - 0.5,
        text='⬇️ Reduz<br>Sinistralidade',
        showarrow=False,
        bgcolor='rgba(44, 160, 44, 0.1)',
        bordercolor=COLORS['success'],
        borderwidth=1,
        font=dict(size=11, color=COLORS['success'])
    )
    
    return fig
