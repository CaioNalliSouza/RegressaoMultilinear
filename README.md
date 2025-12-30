# 🚗 Sistema de Análise Preditiva - Autopeças & Assistência 24h

## 📋 Visão Geral

Sistema profissional de análise preditiva baseado em **Regressão Multifatorial** para empresas intermediadoras de serviços de autopeças e assistência 24h. O sistema analiza o comportamento de empresas parceiras (seguradoras, frotas, etc.) e realiza previsões de faturamento considerando múltiplos fatores operacionais, financeiros e sazonais.

## 🎯 Objetivo

Prever o faturamento mensal de empresas parceiras considerando:
- **Sinistralidade** (Custo/Faturamento × 100%)
  - **Orçada**: Planejamento/expectativa
  - **Realizada**: Resultado efetivo
  - **Meta**: 50% (máximo aceitável)
- **Volume de atendimentos**
- **Ticket médio por atendimento**
- **Percentual de atendimentos com peças**
- **Qualidade do serviço** (NPS, tempo de atendimento, reincidência)
- **Sazonalidade**
- **Fatores econômicos externos**

## � Estrutura do Projeto

```
RegressãoMultifat/
│
├── app.py                      # Aplicação principal Streamlit (RECOMENDADA)
├── padrao.py                   # Versão legada (manter para compatibilidade)
├── requirements.txt            # Dependências do projeto
├── README.md                   # Documentação
├── .gitignore                  # Arquivos ignorados pelo Git
│
├── src/                        # Código-fonte modularizado
│   ├── __init__.py            # Inicializador do pacote
│   ├── config.py              # Configurações gerais
│   ├── data_generator.py      # Geração de dados sintéticos
│   ├── model.py               # Treinamento e predição ML
│   ├── visualizations.py      # Gráficos e visualizações
│   └── utils.py               # Funções utilitárias
│
└── .venv/                      # Ambiente virtual Python
```

## � Instalação

### Opção 1: Download sem Git (Mais Simples)

1. **Baixe o projeto:**
   - Acesse: https://github.com/CaioNalliSouza/RegressaoMultilinear
   - Clique no botão verde **"Code"** → **"Download ZIP"**
   - Extraia o arquivo em uma pasta de sua escolha

2. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute a aplicação:**
   ```bash
   streamlit run app.py
   ```

### Opção 2: Com Ambiente Virtual (Recomendado para Desenvolvedores)

**Por que usar ambiente virtual?**
- ✅ Isola as dependências deste projeto
- ✅ Evita conflitos com outros projetos Python
- ✅ Permite versões diferentes de bibliotecas em projetos distintos
- ✅ Mais organizado e profissional

**Quando NÃO usar:**
- Se for apenas testar rapidamente
- Se a máquina só terá este projeto Python
- Se você não desenvolve outros projetos Python

**Como instalar com ambiente virtual:**

1. **Clone ou baixe o repositório**

2. **Crie o ambiente virtual:**
   ```bash
   python -m venv .venv
   ```

3. **Ative o ambiente virtual:**
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

4. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Execute a aplicação:**
   ```bash
   streamlit run app.py
   ```

6. **Para desativar o ambiente virtual (quando terminar):**
   ```bash
   deactivate
   ```

O sistema abrirá automaticamente no navegador em `http://localhost:8501`

## 🚀 Como Executar

### Versão Principal (Recomendada)
```bash
streamlit run app.py
```

### Versão com Storytelling
```bash
streamlit run app_storytelling.py
```

## 📊 Funcionalidades

### 1. 🔮 Simulador de Cenários - Previsão de Faturamento

Interface interativa para simular diferentes cenários e obter previsões de faturamento:

**Inputs Financeiros:**
- Faturamento do mês anterior
- Sinistralidade do mês anterior (%)

**Inputs Operacionais:**
- Quantidade de atendimentos prevista
- Ticket médio por atendimento
- Percentual de atendimentos com peças

**Inputs de Qualidade:**
- Tempo médio de atendimento (horas)
- Taxa de reincidência (%)
- NPS - Satisfação do cliente

**Contexto Externo:**
- Mês da previsão (sazonalidade)
- Taxa de juros (SELIC)
- Índice de acidentes

**Outputs:**
- Faturamento previsto
- Margem bruta estimada
- Ticket médio efetivo
- Status da sinistralidade
- Recomendações personalizadas

### 2. 📈 Análise Histórica de Performance

Visualizações de tendências temporais:
- Evolução do faturamento
- Evolução da sinistralidade
- Volume de atendimentos mensal
- Evolução do ticket médio
- Análise de sazonalidade (padrões por mês)

### 3. 📊 Dashboard de KPIs

Painel consolidado com:
- Métricas principais (faturamento médio, sinistralidade, NPS, etc.)
- Matriz de correlação entre variáveis
- Distribuição de variáveis chave
- Benchmarks e metas

### 4. 🔍 Análise de Impacto das Variáveis

Análise detalhada dos coeficientes do modelo:
- Peso de cada variável na previsão
- Impacto de variáveis operacionais vs sazonais
- Top 3 variáveis mais impactantes
- Interpretação prática dos coeficientes

## 📈 Indicadores Considerados

### Financeiros
- **Faturamento**: Receita bruta mensal
- **Sinistralidade Orçada**: Planejamento/projeção de custos
- **Sinistralidade Realizada**: (Custo Total / Faturamento) × 100%
- **Sinistralidade Meta**: 50% (máximo aceitável)
- **Margem Bruta**: Faturamento - Custos

### Operacionais
- **Quantidade de Atendimentos**: Volume mensal de sinistros/chamados
- **Ticket Médio**: Valor médio por atendimento
- **% Atendimentos com Peças**: Proporção de atendimentos que incluem venda de peças
- **Custo de Peças**: Custo das peças utilizadas
- **Custo de Mão de Obra**: Custo operacional dos atendimentos

### Qualidade
- **NPS**: Net Promoter Score (0-100)
- **Tempo Médio de Atendimento**: Em horas
- **Taxa de Reincidência**: % de clientes que retornam em até 30 dias

### Externos
- **Sazonalidade**: Variação por mês do ano
- **Taxa de Juros**: SELIC/CDI (%)
- **Índice de Acidentes**: Proxy para demanda (base 100)

## 🎯 Metas e Benchmarks

- **Sinistralidade**: ≤ 50% (META OBRIGATÓRIA)
  - **Sinistralidade Orçada**: Planejamento mensal (tipicamente 45-55%)
  - **Sinistralidade Realizada**: Resultado efetivo do mês
  - **Sinistralidade Meta**: 50% (fixo)
- **NPS**: > 70 (satisfatório)
- **Tempo de Atendimento**: < 3 horas (meta)
- **Taxa de Reincidência**: < 10% (aceitável)

## 📊 Modelo de Machine Learning

**Algoritmo**: Regressão Linear Múltipla (sklearn.LinearRegression)

**Features utilizadas:**
- Faturamento do mês anterior
- Quantidade de atendimentos
- Ticket médio
- % de atendimentos com peças
- Tempo médio de atendimento
- Taxa de reincidência
- Sinistralidade do mês anterior
- NPS
- Taxa de juros
- Índice de acidentes
- Variáveis dummy de mês (sazonalidade)

**Métricas de Performance:**
- R² Score: Qualidade do ajuste
- MAE (Mean Absolute Error): Erro médio em R$
- RMSE (Root Mean Squared Error): Erro quadrático médio

## 🏗️ Arquitetura do Código

### Módulos

**`src/config.py`**
- Configurações centralizadas
- Constantes e parâmetros
- Metas e benchmarks

**`src/data_generator.py`**
- Geração de dados sintéticos realistas
- Simulação de 48 meses de operação
- Cálculos de sinistralidade

**`src/model.py`**
- Treinamento do modelo de ML
- Função de previsão
- Validação e métricas

**`src/visualizations.py`**
- Todos os gráficos Plotly
- Visualizações interativas
- Dashboards

**`src/utils.py`**
- Funções auxiliares
- Cálculos de métricas
- Geração de recomendações

**`app.py`**
- Interface Streamlit
- Orquestração dos módulos
- Lógica de apresentação

## 🔄 Fluxo de Dados

1. **Geração de Dados Históricos**: Simulação realista de 48 meses de operação
2. **Preparação**: Normalização e criação de variáveis dummy
3. **Treinamento**: Split 80/20 (treino/teste) com validação
4. **Previsão**: Input manual de cenário → Predição do faturamento
5. **Análise**: Visualizações e recomendações baseadas nos resultados

## 💡 Melhorias Implementadas

✅ Código modularizado e organizado
✅ Separação de responsabilidades (SRP)
✅ Configurações centralizadas
✅ Funções reutilizáveis
✅ Cache otimizado do Streamlit
✅ Documentação inline completa
✅ Type hints nos parâmetros
✅ Fácil manutenção e extensão
✅ Estrutura profissional

## 💡 Recomendações do Sistema

O sistema gera recomendações automáticas baseadas nos inputs:

- **Tempo de Atendimento Alto**: Sugestão de melhoria de processos
- **Reincidência Elevada**: Alerta de problemas de qualidade
- **NPS Baixo**: Necessidade de treinamento e melhoria de atendimento
- **Sinistralidade Alta**: Revisão de precificação ou renegociação de contratos
- **Baixo % de Peças**: Oportunidades de upsell

## � Próximas Melhorias

- [ ] Upload de dados reais via CSV
- [ ] Múltiplos modelos (Random Forest, XGBoost)
- [ ] Análise comparativa entre empresas
- [ ] Exportação de relatórios em PDF
- [ ] API REST para integração
- [ ] Alertas automáticos via email
- [ ] Previsão multi-período (3, 6, 12 meses)
- [ ] Testes unitários
- [ ] CI/CD Pipeline
- [ ] Dockerização

## 🔧 Manutenção

### Atualizar Dependências
```bash
pip install --upgrade -r requirements.txt
```

### Adicionar Novas Features
1. Adicione a lógica em `src/`
2. Importe no `app.py`
3. Integre na interface

### Executar Testes
```bash
# A implementar
pytest tests/
```

## 👥 Uso

**Público-alvo**: Gestores, analistas e tomadores de decisão em empresas intermediadoras de serviços automotivos.

**Casos de uso**:
1. Projeção de receita mensal
2. Análise de viabilidade de novos contratos
3. Identificação de padrões sazonais
4. Monitoramento de KPIs operacionais
5. Avaliação de ações corretivas

## 📞 Suporte

Para dúvidas ou sugestões sobre o sistema, consulte a documentação interna ou contate a equipe de desenvolvimento.

---

**Versão**: 2.0.0 (Refatorada)  
**Última atualização**: Dezembro 2025  
**Tecnologias**: Python, Streamlit, Scikit-learn, Plotly, Pandas, NumPy

## 📄 Licença

Uso interno da empresa. Todos os direitos reservados.
