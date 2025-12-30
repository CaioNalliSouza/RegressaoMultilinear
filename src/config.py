"""
Configurações gerais do sistema.
"""

# Configurações da aplicação
APP_TITLE = "Sistema de Análise Preditiva - Autopeças & Assistência 24h"
APP_ICON = "🚗"
PAGE_LAYOUT = "wide"

# Configurações do modelo
NUM_MESES_HISTORICO = 48
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Metas e benchmarks
META_SINISTRALIDADE = 50.0
META_NPS = 70
META_TEMPO_ATENDIMENTO = 3.0
META_TAXA_REINCIDENCIA = 10.0

# Cores do tema
COR_SINISTRALIDADE_REALIZADA = '#d62728'
COR_SINISTRALIDADE_ORCADA = '#ff7f0e'
COR_FATURAMENTO = '#1f77b4'
COR_TICKET = '#2ca02c'

# Ranges de valores
RANGE_SINISTRALIDADE = {
    'excelente': (0, 50),
    'atencao': (50, 60),
    'critico': (60, 100)
}

# Features do modelo
FEATURES_MODELO = [
    'Faturamento_Mes_Ant',
    'Qtd_Atendimentos',
    'Ticket_Medio',
    'Perc_Atend_Com_Pecas',
    'Tempo_Medio_Atend_Horas',
    'Taxa_Reincidencia',
    'Sinistralidade_Mes_Ant',
    'NPS',
    'Taxa_Juros',
    'Indice_Acidentes'
]

# Variáveis para correlação
VARIAVEIS_CORRELACAO = [
    'Faturamento',
    'Sinistralidade_Realizada',
    'Sinistralidade_Orcada',
    'Qtd_Atendimentos',
    'Ticket_Medio',
    'NPS',
    'Tempo_Medio_Atend_Horas',
    'Taxa_Reincidencia'
]
