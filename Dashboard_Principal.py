# Dashboard_Principal.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List

# --- Assuming these modules exist and function as in your original code ---
# mi_logica_original.py, dropbox_connector.py, kpis_y_analisis.py, analisis_adicional.py
from mi_logica_original import procesar_archivo_excel, generate_financial_statement, to_excel_buffer, COL_CONFIG
from dropbox_connector import get_dropbox_client, find_financial_files, load_excel_from_dropbox
from kpis_y_analisis import calcular_kpis_periodo, preparar_datos_tendencia, generar_analisis_avanzado_ia, generar_analisis_tendencia_ia
from analisis_adicional import calcular_analisis_vertical, calcular_analisis_horizontal, construir_flujo_de_caja

# ==============================================================================
# ⭐️ CORRECTED AND FINAL ACCOUNTING NOTATION ⭐️
# ==============================================================================
# The entire analysis assumes the following MIXED LOGIC:
# 1. INCOME STATEMENT (P&L): Standard financial logic.
#    - REVENUES are POSITIVE (+).
#    - EXPENSES and COSTS are NEGATIVE (-).
#    - A Net Profit > 0 is a GAIN.
# 2. BALANCE SHEET (BS): Accounting system logic.
#    - ASSETS are POSITIVE (+).
#    - LIABILITIES and EQUITY are NEGATIVE (-).
# The analysis and visualization functions are designed to interpret this logic.
# ==============================================================================


# ==============================================================================
#                      NEW ADVANCED ANALYSIS FUNCTIONS
# ==============================================================================

def plot_enhanced_sparkline(data: pd.Series, title: str, is_percent: bool = False, lower_is_better: bool = False):
    """
    Creates an enhanced sparkline chart that includes change and consistency.
    """
    if data.empty or len(data.dropna()) < 2:
        fig = go.Figure().update_layout(width=200, height=70, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig.add_annotation(text="N/A", showarrow=False)
        return fig

    last_val = data.iloc[-1]
    prev_val = data.iloc[-2] if len(data) > 1 else data.iloc[0]
    
    # Coefficient of variation to measure consistency (lower is more consistent)
    # A small value is added to avoid division by zero if the mean is 0
    consistency = data.std() / (data.mean() + 1e-6) 
    
    delta = last_val - prev_val
    delta_is_good = (not lower_is_better and delta >= 0) or (lower_is_better and delta <= 0)
    
    color = '#28a745' if delta_is_good else '#dc3545'

    fig = go.Figure(go.Scatter(
        x=list(range(len(data))), y=data, mode='lines',
        line=dict(color=color, width=3),
        fill='tozeroy',
        fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)"
    ))
    
    fig.update_layout(
        width=200, height=70,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=5, r=5, t=5, b=5),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False
    )
    return fig

def plot_growth_profitability_matrix(df_tendencia: pd.DataFrame):
    """
    Visualizes the relationship between revenue growth and profit growth.
    """
    df = df_tendencia.copy()
    df['crecimiento_ingresos_pct'] = df['ingresos'].pct_change() * 100
    df['crecimiento_utilidad_pct'] = df['utilidad_neta'].pct_change() * 100
    df.dropna(inplace=True)

    if df.empty:
        return go.Figure().add_annotation(text="Insufficient data for growth matrix", showarrow=False)

    fig = px.scatter(
        df,
        x='crecimiento_ingresos_pct',
        y='crecimiento_utilidad_pct',
        text='periodo',
        title="Growth vs. Profitability Matrix (Period over Period)",
        labels={'crecimiento_ingresos_pct': 'Revenue Growth (%)', 'crecimiento_utilidad_pct': 'Net Profit Growth (%)'},
        size_max=60
    )
    
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")

    fig.add_annotation(x=5, y=5, text="Ideal Growth", showarrow=False, xanchor="left", yanchor="bottom", font=dict(color="green"))
    fig.add_annotation(x=-5, y=5, text="Efficiency Focus", showarrow=False, xanchor="right", yanchor="bottom", font=dict(color="orange"))
    fig.add_annotation(x=5, y=-5, text="Growth at a Cost", showarrow=False, xanchor="left", yanchor="top", font=dict(color="orange"))
    fig.add_annotation(x=-5, y=-5, text="Danger Zone", showarrow=False, xanchor="right", yanchor="top", font=dict(color="red"))
    
    fig.update_traces(textposition='top center')
    fig.update_layout(height=500)
    return fig

def plot_roe_decomposition_waterfall(df_tendencia: pd.DataFrame):
    """
    Creates a waterfall chart that breaks down the change in ROE over time.
    """
    if len(df_tendencia) < 2:
        return go.Figure().add_annotation(text="At least two periods are needed to decompose ROE.", showarrow=False)
        
    start_roe = df_tendencia['roe'].iloc[0]
    end_roe = df_tendencia['roe'].iloc[-1]

    start_margin = df_tendencia['margen_neto'].iloc[0]
    end_margin = df_tendencia['margen_neto'].iloc[-1]
    
    start_turnover = df_tendencia['rotacion_activos'].iloc[0]
    end_turnover = df_tendencia['rotacion_activos'].iloc[-1]

    start_leverage = df_tendencia['apalancamiento'].iloc[0]
    end_leverage = df_tendencia['apalancamiento'].iloc[-1]

    # Contribution of each component. This is an approximation, but very illustrative.
    # Formula: ΔROE ≈ (ΔMargin * T * L) + (M * ΔTurnover * L) + (M * T * ΔLeverage)
    contrib_margin = (end_margin - start_margin) * start_turnover * start_leverage
    contrib_turnover = end_margin * (end_turnover - start_turnover) * start_leverage
    contrib_leverage = end_margin * end_turnover * (end_leverage - start_leverage)

    fig = go.Figure(go.Waterfall(
        name="ROE Decomposition",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["Initial ROE", "Margin Impact", "Asset Turnover Impact", "Leverage Impact", "Final ROE"],
        y=[start_roe, contrib_margin, contrib_turnover, contrib_leverage, end_roe],
        text=[f"{y:.2%}" for y in [start_roe, contrib_margin, contrib_turnover, contrib_leverage, end_roe]],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#28a745"}},
        decreasing={"marker": {"color": "#dc3545"}},
    ))

    fig.update_layout(
        title=f"Profitability Bridge (ROE): {df_tendencia['periodo'].iloc[0]} vs {df_tendencia['periodo'].iloc[-1]}",
        showlegend=False,
        yaxis_title="Return on Equity (ROE)",
        yaxis_tickformat='.2%'
    )
    return fig

def run_what_if_scenario(df_er: pd.DataFrame, value_col: str, changes: Dict[str, float]):
    """
    Runs a simulation based on percentage changes in the main accounts.
    """
    sim_er = df_er.copy()
    
    # Extract base values
    base_ingresos = sim_er[sim_er['Cuenta'].astype(str).str.startswith('4')][value_col].sum()
    base_costos = sim_er[sim_er['Cuenta'].astype(str).str.startswith('6')][value_col].sum()
    base_gastos = sim_er[sim_er['Cuenta'].astype(str).str.startswith('5')][value_col].sum()
    
    # Apply changes. The sign logic is crucial here.
    # Revenues are positive, costs/expenses are negative.
    sim_ingresos = base_ingresos * (1 + changes['ingresos'] / 100)
    sim_costos = base_costos * (1 + changes['costos'] / 100) # An increase on the slider increases the expense (makes it more negative)
    sim_gastos = base_gastos * (1 + changes['gastos'] / 100)

    sim_utilidad_bruta = sim_ingresos + sim_costos # Costs is already negative
    sim_utilidad_operativa = sim_utilidad_bruta + sim_gastos # Expenses is already negative
    
    # Assume other income/expenses do not change for simplicity.
    otros_neto = df_er[~df_er['Cuenta'].astype(str).str.match('^[456]')][value_col].sum()
    sim_utilidad_neta = sim_utilidad_operativa + otros_neto

    # Calculate simulated KPIs
    kpis = {}
    kpis['Utilidad Neta'] = sim_utilidad_neta
    kpis['Margen Neto'] = (sim_utilidad_neta / sim_ingresos) if sim_ingresos != 0 else 0
    kpis['Margen Bruto'] = (sim_utilidad_bruta / sim_ingresos) if sim_ingresos != 0 else 0
    kpis['Margen Operativo'] = (sim_utilidad_operativa / sim_ingresos) if sim_ingresos != 0 else 0
    
    return kpis

# ==============================================================================
#                      PAGE CONFIGURATION AND AUTHENTICATION
# ==============================================================================
st.set_page_config(layout="wide", page_title="Virtual CFO PRO with AI", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .reportview-container {background: #f0f2f6}
    .main .block-container {padding-top: 2rem;}
    .kpi-card {
        padding: 1.5rem; 
        border-radius: 10px; 
        box-shadow: 0 8px 12px rgba(0,0,0,0.1); 
        background-color: white; 
        text-align: center;
        border-left: 7px solid #1967d2;
    }
    .kpi-title {font-size: 1.1em; font-weight: 600; color: #5f6368;}
    .kpi-value {font-size: 2.2em; font-weight: 700; color: #202124;}
    .kpi-delta {font-size: 1em; font-weight: 600;}
    .ai-analysis-text {
        background-color: #e8f0fe;
        border-left: 5px solid #1967d2;
        padding: 20px;
        border-radius: 8px;
        font-size: 1.1em;
        line-height: 1.6;
    }
    .stTabs [data-baseweb="tab-list"] {
		gap: 24px;
	}
	.stTabs [data-baseweb="tab"] {
		height: 50px;
        white-space: pre-wrap;
		background-color: #F0F2F6;
		border-radius: 4px 4px 0px 0px;
		gap: 1px;
		padding-top: 10px;
		padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("👨‍🚀 Virtual CFO PRO: Augmented Financial Analysis")

try:
    real_password = st.secrets["general"]["password"]
except Exception:
    st.error("Password not found in Streamlit secrets."); st.stop()

if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if not st.session_state.authenticated:
    password = st.text_input("Enter the password to access:", type="password")
    if password == real_password:
        st.session_state.authenticated = True
        st.rerun()
    else:
        if password: st.warning("Incorrect password.")
        st.stop()

# ==============================================================================
#                       AUTOMATIC DATA LOADING FROM DROPBOX
# ==============================================================================
@st.cache_data(ttl=3600)
def cargar_y_procesar_datos_full():
    dbx = get_dropbox_client()
    if not dbx: return None
    archivos_financieros = find_financial_files(dbx, base_folder="/data")
    if not archivos_financieros:
        st.warning("No Excel files found in the /data folder of Dropbox.")
        return None

    datos_historicos = {}
    progress_bar = st.progress(0, text="Initiating load...")
    for i, file_info in enumerate(archivos_financieros):
        periodo = file_info["periodo"]
        path = file_info["path"]
        progress_bar.progress((i + 1) / len(archivos_financieros), text=f"Processing {periodo}...")
        excel_bytes = load_excel_from_dropbox(dbx, path)
        if excel_bytes:
            try:
                df_er, df_bg = procesar_archivo_excel(excel_bytes)
                datos_periodo = {'df_er_master': df_er, 'df_bg_master': df_bg, 'kpis': {}}
                # Calculate KPIs for consolidated and cost centers
                er_conf = COL_CONFIG['ESTADO_DE_RESULTADOS']
                cc_cols = ['Todos'] + [name for name in er_conf.get('CENTROS_COSTO_COLS', {}).values() if name in df_er and name not in ['Total_Consolidado_ER', 'Sin centro de coste']]
                for cc in cc_cols:
                    datos_periodo['kpis'][cc] = calcular_kpis_periodo(df_er, df_bg, cc)
                datos_historicos[periodo] = datos_periodo
            except Exception as e:
                st.error(f"Error processing the file for the period {periodo}: {e}")
    progress_bar.empty()
    return datos_historicos

if 'datos_historicos' not in st.session_state: st.session_state.datos_historicos = None
if st.sidebar.button("♻️ Refresh Data from Dropbox", use_container_width=True, type="primary"):
    st.cache_data.clear()
    st.session_state.datos_historicos = None
if st.session_state.datos_historicos is None:
    st.session_state.datos_historicos = cargar_y_procesar_datos_full()
if not st.session_state.datos_historicos:
    st.error("Could not load data. Check the connection and file structure in Dropbox.")
    st.stop()

# ==============================================================================
#                            MAIN USER INTERFACE
# ==============================================================================
st.sidebar.title("🚀 Analysis Options")
sorted_periods = sorted(st.session_state.datos_historicos.keys(), reverse=True)
period_options = ["📈 Strategic Evolution Analysis"] + sorted_periods
selected_view = st.sidebar.selectbox("Select the analysis view:", period_options)

# ==============================================================================
#                  VIEW 1: STRATEGIC EVOLUTION ANALYSIS (TRENDS)
# ==============================================================================
if selected_view == "📈 Strategic Evolution Analysis":
    st.header("📈 Management Evolution Report")
    df_tendencia = preparar_datos_tendencia(st.session_state.datos_historicos)
    
    if df_tendencia.empty or len(df_tendencia) < 2:
        st.info("At least two periods of data are needed to generate an evolution analysis.")
        st.stop()

    with st.expander("🧠 **Strategic Diagnosis by the Virtual CFO (AI)**", expanded=True):
        with st.spinner('The Senior AI Analyst is evaluating the multi-year trajectory...'):
            # We prepare a much richer context for the AI
            contexto_tendencia = {
                "datos_tendencia": df_tendencia.to_dict('records'),
                "kpi_principales": ['margen_neto', 'roe', 'razon_corriente', 'endeudamiento_activo', 'ingresos', 'utilidad_neta'],
                "volatilidad": {
                    k: (df_tendencia[k].std() / df_tendencia[k].mean()) if df_tendencia[k].mean() != 0 else 0
                    for k in ['margen_neto', 'roe', 'ingresos']
                },
                "correlaciones": df_tendencia[['ingresos', 'gastos_operativos', 'utilidad_neta', 'margen_neto']].corr().to_dict(),
                "convencion_contable": "Standard P&L (Revenues+, Expenses-), System BS (Assets+, Liabilities-).",
                "pregunta_clave": (
                    "Act as a strategic CFO. Analyze the company's financial evolution based on trend, volatility, and correlation data. "
                    "1. **Growth Narrative:** Is revenue growth healthy and profitable? Is it accelerating or decelerating? "
                    "2. **Profitability Narrative:** How have ROE and Net Margin evolved? How stable are they? What drives profitability (sales or efficiency)? "
                    "3. **Solvency Narrative:** Is the liquidity and debt position stronger or weaker over time? "
                    "4. **Inflection Point:** Identify the most critical period where a key trend changed (positively or negatively). "
                    "5. **Conclusion and Strategic Focus:** Summarize the overall financial status and recommend the top 2-3 key strategic priorities for management."
                )
            }
            analisis_tendencia_ia = generar_analisis_tendencia_ia(contexto_tendencia) 
            st.markdown(f"<div class='ai-analysis-text'>{analisis_tendencia_ia}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Key Performance Indicators (KPIs) Over Time")

    kpi_cols = st.columns(4)
    kpi_definitions = {
        'margen_neto': {'title': 'Net Margin', 'is_percent': True, 'lower_is_better': False},
        'roe': {'title': 'ROE (Return on Equity)', 'is_percent': True, 'lower_is_better': False},
        'razon_corriente': {'title': 'Current Ratio (Liquidity)', 'is_percent': False, 'lower_is_better': False},
        'endeudamiento_activo': {'title': 'Asset Indebtedness', 'is_percent': True, 'lower_is_better': True}
    }

    for i, (kpi, config) in enumerate(kpi_definitions.items()):
        with kpi_cols[i]:
            last_val = df_tendencia[kpi].iloc[-1]
            prev_val = df_tendencia[kpi].iloc[-2]
            consistency = (df_tendencia[kpi].std() / df_tendencia[kpi].mean()) if df_tendencia[kpi].mean() != 0 else 0
            
            st.markdown(f"<p class='kpi-title'>{config['title']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='kpi-value'>{last_val:.2% if config['is_percent'] else f'{last_val:.2f}'}</p>", unsafe_allow_html=True)
            st.markdown(f"**Consistency:** {'Low' if consistency > 0.2 else 'Medium' if consistency > 0.1 else 'High'}")
            st.plotly_chart(plot_enhanced_sparkline(df_tendencia[kpi], config['title'], config['is_percent'], config['lower_is_better']), use_container_width=True)

    st.markdown("---")
    st.subheader("Strategic Visual Diagnostics")
    
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.plotly_chart(plot_growth_profitability_matrix(df_tendencia), use_container_width=True)
    with v_col2:
        st.plotly_chart(plot_roe_decomposition_waterfall(df_tendencia), use_container_width=True)

    fig_combinada = go.Figure()
    fig_combinada.add_trace(go.Bar(x=df_tendencia['periodo'], y=df_tendencia['ingresos'], name='Revenues', marker_color='#28a745'))
    fig_combinada.add_trace(go.Bar(x=df_tendencia['periodo'], y=df_tendencia['gastos_operativos'].abs(), name='Operating Expenses', marker_color='#ffc107'))
    fig_combinada.add_trace(go.Scatter(x=df_tendencia['periodo'], y=df_tendencia['utilidad_neta'], name='Net Profit', mode='lines+markers', line=dict(color='#0d6efd', width=4), yaxis="y2"))
    
    fig_combinada.update_layout(
        title='Evolution of P&L Components', barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title='Revenues and Expenses (COP)'),
        yaxis2=dict(title='Net Profit (COP)', overlaying='y', side='right', showgrid=False, tickformat="$,.0f"),
        height=500
    )
    st.plotly_chart(fig_combinada, use_container_width=True)

# ==============================================================================
#                 VIEW 2: DEEP DIVE ANALYSIS CENTER (SINGLE PERIOD)
# ==============================================================================
else:
    st.header(f"Analysis Center for Period: {selected_view}")
    
    # --- Data Preparation ---
    data_actual = st.session_state.datos_historicos.get(selected_view)
    if not data_actual:
        st.error(f"No data found for period: {selected_view}"); st.stop()

    periodo_actual_idx = sorted_periods.index(selected_view)
    periodo_previo = sorted_periods[periodo_actual_idx + 1] if periodo_actual_idx + 1 < len(sorted_periods) else None
    data_previa = st.session_state.datos_historicos.get(periodo_previo) if periodo_previo else None

    df_er_actual = data_actual['df_er_master']
    df_bg_actual = data_actual['df_bg_master']
    kpis_por_cc = data_actual['kpis']

    # --- Sidebar Filters ---
    st.sidebar.subheader("Period Filters")
    cc_options_all = sorted(list(kpis_por_cc.keys()))
    cc_filter = st.sidebar.selectbox("Filter by Cost Center:", cc_options_all, key=f"cc_{selected_view}")
    
    er_conf = COL_CONFIG['ESTADO_DE_RESULTADOS']
    nivel_col = er_conf.get('NIVEL_LINEA', 'Grupo')
    if nivel_col in df_er_actual.columns:
        max_nivel = int(df_er_actual[nivel_col].max())
        nivel_seleccionado = st.sidebar.slider("Account Detail Level:", 1, max_nivel, 2, key=f"nivel_er_{selected_view}")
    else:
        nivel_seleccionado = 2

    st.sidebar.subheader("Account Search")
    search_account_input = st.sidebar.text_input("Search by account number:", key=f"search_{selected_view}", placeholder="e.g., 510506")
    
    # --- Variation Calculation ---
    df_variacion_er = None
    if data_previa:
        valor_col_nombre = 'Total_Consolidado_ER' if cc_filter == 'Todos' else cc_filter
        # Important: we reuse the original 'calcular_variaciones_er' function which we now understand well
        from Dashboard_Principal import calcular_variaciones_er
        df_variacion_er = calcular_variaciones_er(df_er_actual, data_previa['df_er_master'], cc_filter)
        if df_variacion_er is not None and not df_variacion_er.empty:
            st.info(f"Comparative analysis against the period **{periodo_previo}**.")
    else:
        st.warning("No previous period for comparative analysis.")

    # --- Detailed Analysis Tabs ---
    tab_list = [
        "📊 AI Executive Summary", "🎯 DuPont & ROE Analysis", "💰 Profitability Analysis",
        "🧾 Expense Analysis", "🔬 Scenario Simulator", "📋 Financial Reports"
    ]
    tab_gen, tab_roe, tab_utilidad, tab_gas, tab_simulador, tab_rep = st.tabs(tab_list)

    with tab_gen:
        st.subheader(f"Executive Summary for: {cc_filter}")
        selected_kpis = kpis_por_cc.get(cc_filter, {})
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        kpi_col1.metric("Net Margin", f"{selected_kpis.get('margen_neto', 0):.2%}")
        kpi_col2.metric("ROE", f"{selected_kpis.get('roe', 0):.2%}")
        kpi_col3.metric("Current Ratio", f"{selected_kpis.get('razon_corriente', 0):.2f}")
        kpi_col4.metric("Indebtedness", f"{selected_kpis.get('endeudamiento_activo', 0):.2%}")

        st.markdown("---")
        with st.expander("🧠 **Diagnosis and Recommendations from the Virtual CFO (AI)**", expanded=True):
            with st.spinner('The Virtual CFO is preparing a deep analysis...'):
                contexto_ia = {
                    "kpis_actuales": selected_kpis,
                    "kpis_previos": data_previa['kpis'].get(cc_filter, {}) if data_previa else {},
                    "periodo_actual": selected_view, "periodo_previo": periodo_previo, "centro_costo": cc_filter,
                    "convencion_contable": "Standard P&L (Revenues+, Expenses-). System BS (Assets+, Liabilities-).",
                }
                if df_variacion_er is not None and not df_variacion_er.empty:
                    top_favorables = df_variacion_er.nlargest(5, 'Variacion_Absoluta')
                    top_desfavorables = df_variacion_er.nsmallest(5, 'Variacion_Absoluta')
                    contexto_ia["variaciones_favorables"] = top_favorables[['Descripción', 'Variacion_Absoluta']].to_dict('records')
                    contexto_ia["variaciones_desfavorables"] = top_desfavorables[['Descripción', 'Variacion_Absoluta']].to_dict('records')

                contexto_ia["pregunta_clave"] = (
                    "Act as an incisive CFO. Analyze the KPIs and variations for this period. "
                    "1. **Financial Health Diagnosis:** What is the overall condition (Excellent, Good, Concerning, Critical)? Justify with the 2-3 most relevant KPIs. "
                    "2. **Root Cause Analysis (ROE):** The ROE changed. Break it down using DuPont logic. Was it due to margin, asset turnover, or leverage? Be explicit. "
                    "3. **Change Drivers:** Which 3 specific accounts explain most of the variation in net income vs. the previous period? "
                    "4. **Questions for Management:** Formulate 3 direct, data-driven questions a manager should answer. (e.g., 'We see travel expenses increased by 30% while revenues only grew 5%. What was the ROI on this investment?'). "
                    "5. **Key Risk and Opportunity:** Identify the biggest financial risk and the clearest opportunity revealed by these numbers."
                )
                analisis_ia = generar_analisis_avanzado_ia(contexto_ia)
                st.markdown(f"<div class='ai-analysis-text'>{analisis_ia}</div>", unsafe_allow_html=True)

    with tab_roe:
        st.subheader("🎯 Profitability Analysis (ROE) with DuPont Model")
        st.markdown("The **DuPont Analysis** breaks down ROE into three levers: operating efficiency (Net Margin), asset use efficiency (Turnover), and financial leverage (Debt). It helps understand *why* profitability changed.")
        
        if data_previa:
            kpis_actuales = kpis_por_cc.get(cc_filter, {})
            kpis_previos = data_previa['kpis'].get(cc_filter, {})
            dupont_data = {
                'Component': ['Net Margin (Op. Efficiency)', 'Asset Turnover (Asset Use)', 'Financial Leverage (Debt)', 'ROE (Final Result)'],
                'Formula': ['Net Profit / Revenues', 'Revenues / Total Assets', 'Total Assets / Equity', 'Margin * Turnover * Leverage'],
                selected_view: [kpis_actuales.get(k, 0) for k in ['margen_neto', 'rotacion_activos', 'apalancamiento', 'roe']],
                periodo_previo: [kpis_previos.get(k, 0) for k in ['margen_neto', 'rotacion_activos', 'apalancamiento', 'roe']]
            }
            df_dupont = pd.DataFrame(dupont_data)
            df_dupont['Variación'] = df_dupont[selected_view] - df_dupont[periodo_previo]
            
            st.dataframe(
                df_dupont.style.format({
                    selected_view: '{:.2%}', periodo_previo: '{:.2%}', 'Variación': '{:+.2%}',
                    'Asset Turnover (Asset Use)': '{:.2f}x', 'Financial Leverage (Debt)': '{:.2f}x'
                }).background_gradient(cmap='RdYlGn', subset=['Variación'], low=0.4, high=0.4),
                use_container_width=True
            )
        else:
            st.info("A previous period is required for the comparative DuPont analysis.")

    with tab_utilidad:
        st.subheader("💰 Net Profit Analysis: What drove the result?")
        if df_variacion_er is not None and not df_variacion_er.empty:
            # We reuse the original function
            from Dashboard_Principal import plot_waterfall_utilidad_neta
            st.plotly_chart(plot_waterfall_utilidad_neta(df_variacion_er, selected_view, periodo_previo), use_container_width=True)
            
            st.markdown("#### Main Drivers of Change vs. Previous Period")
            col1, col2 = st.columns(2)
            
            top_favorables = df_variacion_er[df_variacion_er['Variacion_Absoluta'] > 0].sort_values('Variacion_Absoluta', ascending=False).head(10)
            top_desfavorables = df_variacion_er[df_variacion_er['Variacion_Absoluta'] < 0].sort_values('Variacion_Absoluta').head(10)
            format_dict = {'Valor_previo': '${:,.0f}', 'Valor_actual': '${:,.0f}', 'Variacion_Absoluta': '${:,.0f}'}

            with col1:
                st.markdown("✅ **Positive Impacts (Helped Profitability)**")
                st.dataframe(top_favorables[['Descripción', 'Valor_actual', 'Valor_previo', 'Variacion_Absoluta']].style.format(format_dict).background_gradient(cmap='Greens', subset=['Variacion_Absoluta']), use_container_width=True)
            with col2:
                st.markdown("❌ **Negative Impacts (Hurt Profitability)**")
                st.dataframe(top_desfavorables[['Descripción', 'Valor_actual', 'Valor_previo', 'Variacion_Absoluta']].style.format(format_dict).background_gradient(cmap='Reds_r', subset=['Variacion_Absoluta']), use_container_width=True)
        else:
            st.info("A previous period is required for this analysis.")
    
    with tab_gas:
        st.subheader("🧾 Detailed Expense and Cost Analysis")
        valor_col_nombre = 'Total_Consolidado_ER' if cc_filter == 'Todos' else cc_filter

        if valor_col_nombre in df_er_actual.columns:
            df_gastos_costos = df_er_actual[df_er_actual['Cuenta'].astype(str).str.match('^[56]')]
            st.markdown("#### Expense and Cost Composition of the Period")
            
            fig_treemap = px.treemap(
                df_gastos_costos, 
                path=[px.Constant("Total Expenses and Costs"), 'Tipo', 'Descripción'], 
                values=df_gastos_costos[valor_col_nombre].abs(),
                title='Expense and Cost Distribution (Click to explore)',
                color=df_gastos_costos[valor_col_nombre].abs(),
                color_continuous_scale='Reds'
            )
            fig_treemap.update_layout(margin = dict(t=50, l=25, r=25, b=25))
            st.plotly_chart(fig_treemap, use_container_width=True)

        if df_variacion_er is not None and not df_variacion_er.empty:
            st.markdown("#### Expense and Cost Variation vs. Previous Period")
            df_gc_var = df_variacion_er[df_variacion_er['Cuenta'].astype(str).str.match('^[56]')]
            st.dataframe(df_gc_var[['Descripción', 'Valor_actual', 'Valor_previo', 'Variacion_Absoluta']].style.format(format_dict), use_container_width=True)

    with tab_simulador:
        st.subheader("🔬 Profitability Scenario Simulator")
        st.markdown("Use the sliders to model the impact of changes in key P&L components. See how each change affects your profit and margins in real-time.")

        valor_col_nombre = 'Total_Consolidado_ER' if cc_filter == 'Todos' else cc_filter
        
        sim_col1, sim_col2 = st.columns([1, 2])
        
        with sim_col1:
            st.markdown("#### Simulation Controls")
            ingresos_change = st.slider("Change % in Total Revenues", -25.0, 25.0, 0.0, 0.5)
            costos_change = st.slider("Change % in Cost of Goods Sold (COGS)", -25.0, 25.0, 0.0, 0.5)
            gastos_change = st.slider("Change % in Operating Expenses", -25.0, 25.0, 0.0, 0.5)
            
            changes = {'ingresos': ingresos_change, 'costos': costos_change, 'gastos': gastos_change}

        # Run the simulation
        sim_kpis = run_what_if_scenario(df_er_actual, valor_col_nombre, changes)
        actual_kpis = kpis_por_cc.get(cc_filter, {})

        with sim_col2:
            st.markdown("#### Simulation Results")
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric("Simulated Net Profit", 
                          f"${sim_kpis.get('Utilidad Neta', 0):,.0f}",
                          f"{sim_kpis.get('Utilidad Neta', 0) - actual_kpis.get('utilidad_neta', 0):,.0f} vs. Actual")
            with res_col2:
                 st.metric("Simulated Net Margin", 
                          f"{sim_kpis.get('Margen Neto', 0):.2%}",
                          f"{sim_kpis.get('Margen Neto', 0) - actual_kpis.get('margen_neto', 0):+.2%} vs. Actual")
            with res_col3:
                 st.metric("Simulated Operating Margin", 
                          f"{sim_kpis.get('Margen Operativo', 0):.2%}",
                          f"{sim_kpis.get('Margen Operativo', 0) - actual_kpis.get('margen_operativo', 0):+.2%} vs. Actual")

            st.markdown("---")
            st.write("This analysis allows you to quantify the impact of strategic decisions such as marketing campaigns (increase in revenues and expenses), negotiations with suppliers (reduction of costs), or efficiency plans (reduction of expenses).")

    with tab_rep:
        st.subheader("📊 Detailed Financial Reports")
        
        st.markdown(f"#### Income Statement (Detail Level: {nivel_seleccionado})")
        df_er_display = generate_financial_statement(df_er_actual, 'Estado de Resultados', cc_filter, nivel_seleccionado)
        st.dataframe(df_er_display.style.format({'Valor': "${:,.0f}"}), use_container_width=True, height=600)

        st.markdown("#### Balance Sheet")
        df_bg_display = generate_financial_statement(df_bg_actual, 'Balance General', 99) # Maximum detail
        st.dataframe(df_bg_display.style.format({'Valor': "${:,.0f}"}), use_container_width=True, height=600)
    
        st.markdown("#### Statement of Cash Flows (Indirect Method)")
        if data_previa:
            with st.spinner("Building Cash Flow Statement..."):
                df_flujo = construir_flujo_de_caja(
                    df_er_actual, df_bg_actual, data_previa['df_bg_master'], 
                    'Total_Consolidado_ER' if cc_filter == 'Todos' else cc_filter,
                    er_conf['CUENTA'], COL_CONFIG['BALANCE_GENERAL']['SALDO_FINAL'], COL_CONFIG['BALANCE_GENERAL']['CUENTA']
                )
                st.dataframe(df_flujo.style.format({'Valor': '${:,.0f}'}), use_container_width=True)
        else:
            st.info("A previous period is required to generate the Statement of Cash Flows.")

    if search_account_input:
        st.markdown("---")
        with st.expander(f"Search result for accounts starting with '{search_account_input}'", expanded=True):
            cuenta_col_er = er_conf.get('CUENTA', 'Cuenta')
            cuenta_col_bg = COL_CONFIG['BALANCE_GENERAL'].get('CUENTA', 'Cuenta')
            
            st.write("**Income Statement**")
            df_search_er = df_er_actual[df_er_actual[cuenta_col_er].astype(str).str.startswith(search_account_input)]
            if not df_search_er.empty:
                st.dataframe(df_search_er)
            else:
                st.info(f"No accounts found in the P&L for '{search_account_input}'.")
            
            st.write("**Balance Sheet**")
            df_search_bg = df_bg_actual[df_bg_actual[cuenta_col_bg].astype(str).str.startswith(search_account_input)]
            if not df_search_bg.empty:
                st.dataframe(df_search_bg)
            else:
                st.info(f"No accounts found in the BS for '{search_account_input}'.")

    st.sidebar.markdown("---")
    er_to_dl = generate_financial_statement(df_er_actual, 'Estado de Resultados', cc_filter, 99)
    bg_to_dl = generate_financial_statement(df_bg_actual, 'Balance General', 99)
    excel_buffer = to_excel_buffer(er_to_dl, bg_to_dl)
    st.sidebar.download_button(
        label=f"📥 Download Reports ({selected_view}, {cc_filter})",
        data=excel_buffer,
        file_name=f"Financial_Report_{selected_view}_{cc_filter}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
