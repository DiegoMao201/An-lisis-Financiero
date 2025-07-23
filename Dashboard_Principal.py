# Dashboard_Principal.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List

# --- Importamos nuestros módulos (el archivo corregido será kpis_y_analisis.py) ---
from mi_logica_original import procesar_archivo_excel, generate_financial_statement, to_excel_buffer, COL_CONFIG
from dropbox_connector import get_dropbox_client, find_financial_files, load_excel_from_dropbox
from kpis_y_analisis import calcular_kpis_periodo, preparar_datos_tendencia, generar_analisis_avanzado_ia, generar_analisis_tendencia_ia
from analisis_adicional import calcular_analisis_vertical, calcular_analisis_horizontal, construir_flujo_de_caja

# ==============================================================================
# ⭐️ NOTACIÓN CONTABLE CORREGIDA Y FINAL ⭐️
# ==============================================================================
# En todo el análisis se asume la siguiente LÓGICA MIXTA:
# 1. ESTADO DE RESULTADOS (P&L): Lógica financiera estándar.
#    - INGRESOS son POSITIVOS (+).
#    - GASTOS y COSTOS son NEGATIVOS (-).
#    - Una Utilidad Neta > 0 es una GANANCIA.
# 2. BALANCE GENERAL (BS): Lógica del sistema contable.
#    - ACTIVOS son POSITIVOS (+).
#    - PASIVOS y PATRIMONIO son NEGATIVOS (-).
# Las funciones de análisis y visualización están diseñadas para interpretar esta lógica.
# ==============================================================================


# ==============================================================================
#                      NUEVAS FUNCIONES DE ANÁLISIS AVANZADO
# ==============================================================================
# (Estas funciones ya están en tu archivo principal y son correctas)
def plot_enhanced_sparkline(data: pd.Series, title: str, is_percent: bool = False, lower_is_better: bool = False):
    """
    Crea un minigráfico de línea mejorado que incluye el cambio y la consistencia.
    """
    if data.empty or len(data.dropna()) < 2:
        fig = go.Figure().update_layout(width=200, height=70, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig.add_annotation(text="N/A", showarrow=False)
        return fig

    last_val = data.iloc[-1]
    prev_val = data.iloc[-2] if len(data) > 1 else data.iloc[0]
    
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
    Visualiza la relación entre el crecimiento de ingresos y el de utilidad.
    """
    df = df_tendencia.copy()
    df['crecimiento_ingresos_pct'] = df['ingresos'].pct_change() * 100
    df['crecimiento_utilidad_pct'] = df['utilidad_neta'].pct_change() * 100
    df.dropna(inplace=True)

    if df.empty:
        return go.Figure().add_annotation(text="Datos insuficientes para la matriz de crecimiento", showarrow=False)

    fig = px.scatter(
        df,
        x='crecimiento_ingresos_pct',
        y='crecimiento_utilidad_pct',
        text='periodo',
        title="Matriz de Crecimiento vs. Rentabilidad (Periodo a Periodo)",
        labels={'crecimiento_ingresos_pct': 'Crecimiento de Ingresos (%)', 'crecimiento_utilidad_pct': 'Crecimiento de Utilidad Neta (%)'},
        size_max=60
    )
    
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")

    fig.add_annotation(x=5, y=5, text="Crecimiento Ideal", showarrow=False, xanchor="left", yanchor="bottom", font=dict(color="green"))
    fig.add_annotation(x=-5, y=5, text="Enfoque en Eficiencia", showarrow=False, xanchor="right", yanchor="bottom", font=dict(color="orange"))
    fig.add_annotation(x=5, y=-5, text="Crecimiento a un Costo", showarrow=False, xanchor="left", yanchor="top", font=dict(color="orange"))
    fig.add_annotation(x=-5, y=-5, text="Zona de Peligro", showarrow=False, xanchor="right", yanchor="top", font=dict(color="red"))
    
    fig.update_traces(textposition='top center')
    fig.update_layout(height=500)
    return fig

def plot_roe_decomposition_waterfall(df_tendencia: pd.DataFrame):
    """
    Crea un gráfico de cascada que descompone el cambio en ROE a lo largo del tiempo.
    """
    if len(df_tendencia) < 2:
        return go.Figure().add_annotation(text="Se necesitan al menos dos periodos para descomponer el ROE.", showarrow=False)
        
    start_roe = df_tendencia['roe'].iloc[0]
    end_roe = df_tendencia['roe'].iloc[-1]

    start_margin = df_tendencia['margen_neto'].iloc[0]
    end_margin = df_tendencia['margen_neto'].iloc[-1]
    
    start_turnover = df_tendencia['rotacion_activos'].iloc[0]
    end_turnover = df_tendencia['rotacion_activos'].iloc[-1]

    start_leverage = df_tendencia['apalancamiento'].iloc[0]
    end_leverage = df_tendencia['apalancamiento'].iloc[-1]

    contrib_margin = (end_margin - start_margin) * start_turnover * start_leverage
    contrib_turnover = end_margin * (end_turnover - start_turnover) * start_leverage
    contrib_leverage = end_margin * end_turnover * (end_leverage - start_leverage)

    fig = go.Figure(go.Waterfall(
        name="ROE Decomposition",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["ROE Inicial", "Impacto Margen", "Impacto Rotación Activos", "Impacto Apalancamiento", "ROE Final"],
        y=[start_roe, contrib_margin, contrib_turnover, contrib_leverage, end_roe],
        text=[f"{y:.2%}" for y in [start_roe, contrib_margin, contrib_turnover, contrib_leverage, end_roe]],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#28a745"}},
        decreasing={"marker": {"color": "#dc3545"}},
    ))

    fig.update_layout(
        title=f"Puente de Rentabilidad (ROE): {df_tendencia['periodo'].iloc[0]} vs {df_tendencia['periodo'].iloc[-1]}",
        showlegend=False,
        yaxis_title="Retorno sobre Patrimonio (ROE)",
        yaxis_tickformat='.2%'
    )
    return fig

def run_what_if_scenario(df_er: pd.DataFrame, value_col: str, changes: Dict[str, float]):
    """
    Ejecuta una simulación basada en cambios porcentuales en las cuentas principales.
    """
    sim_er = df_er.copy()
    
    base_ingresos = sim_er[sim_er['Cuenta'].astype(str).str.startswith('4')][value_col].sum()
    base_costos = sim_er[sim_er['Cuenta'].astype(str).str.startswith('6')][value_col].sum()
    base_gastos = sim_er[sim_er['Cuenta'].astype(str).str.startswith('5')][value_col].sum()
    
    sim_ingresos = base_ingresos * (1 + changes['ingresos'] / 100)
    sim_costos = base_costos * (1 + changes['costos'] / 100)
    sim_gastos = base_gastos * (1 + changes['gastos'] / 100)

    sim_utilidad_bruta = sim_ingresos + sim_costos
    sim_utilidad_operativa = sim_utilidad_bruta + sim_gastos
    
    otros_neto = df_er[~df_er['Cuenta'].astype(str).str.match('^[456]')][value_col].sum()
    sim_utilidad_neta = sim_utilidad_operativa + otros_neto

    kpis = {}
    kpis['Utilidad Neta'] = sim_utilidad_neta
    kpis['Margen Neto'] = (sim_utilidad_neta / sim_ingresos) if sim_ingresos != 0 else 0
    kpis['Margen Bruto'] = (sim_utilidad_bruta / sim_ingresos) if sim_ingresos != 0 else 0
    kpis['Margen Operativo'] = (sim_utilidad_operativa / sim_ingresos) if sim_ingresos != 0 else 0
    
    return kpis

def calcular_variaciones_er(df_actual: pd.DataFrame, df_previo: pd.DataFrame, cc_filter: str) -> pd.DataFrame:
    """Calcula las variaciones absolutas y porcentuales para el Estado de Resultados."""
    er_conf = COL_CONFIG['ESTADO_DE_RESULTADOS']
    cuenta_col = er_conf.get('CUENTA', 'Cuenta')
    desc_col = er_conf.get('DESCRIPCION_CUENTA', 'Título')
    valor_col_nombre = 'Total_Consolidado_ER' if cc_filter == 'Todos' else cc_filter

    required_cols_actual = [cuenta_col, desc_col, valor_col_nombre]
    if any(col not in df_actual.columns for col in required_cols_actual):
        st.error(f"Error Crítico (Periodo Actual): Faltan columnas en `df_actual`. Requeridas: {required_cols_actual}")
        return pd.DataFrame()

    df1 = df_actual[required_cols_actual].copy()
    df1.rename(columns={valor_col_nombre: 'Valor_actual'}, inplace=True)

    if all(col in df_previo.columns for col in required_cols_actual):
        df2 = df_previo[required_cols_actual].copy()
        df2.rename(columns={valor_col_nombre: 'Valor_previo'}, inplace=True)
    else:
        st.warning(f"ADVERTENCIA: Columnas no encontradas en periodo anterior para CC '{cc_filter}'. Se usarán ceros.")
        df2 = df_actual[[cuenta_col, desc_col]].copy()
        df2['Valor_previo'] = 0

    df_variacion = pd.merge(df1, df2, on=[cuenta_col, desc_col], how='outer').fillna(0)
    df_variacion['Variacion_Absoluta'] = df_variacion['Valor_actual'] - df_variacion['Valor_previo']
    
    if desc_col != 'Descripción':
        df_variacion.rename(columns={desc_col: 'Descripción'}, inplace=True)
    
    return df_variacion

def plot_waterfall_utilidad_neta(df_variacion: pd.DataFrame, periodo_actual: str, periodo_previo: str):
    """Crea un gráfico de cascada para explicar la variación de la Utilidad Neta."""
    cuenta_col = 'Cuenta' 
    if cuenta_col not in df_variacion.columns:
        st.error(f"Columna '{cuenta_col}' no existe en datos de variación para cascada.")
        return go.Figure()

    utilidad_neta_actual = df_variacion['Valor_actual'].sum()
    utilidad_neta_previa = df_variacion['Valor_previo'].sum()

    variacion_ingresos = df_variacion[df_variacion[cuenta_col].astype(str).str.startswith('4')]['Variacion_Absoluta'].sum()
    variacion_costos = df_variacion[df_variacion[cuenta_col].astype(str).str.startswith('6')]['Variacion_Absoluta'].sum()
    variacion_gastos = df_variacion[df_variacion[cuenta_col].astype(str).str.startswith('5')]['Variacion_Absoluta'].sum()
    otras_variaciones = df_variacion['Variacion_Absoluta'].sum() - (variacion_ingresos + variacion_costos + variacion_gastos)

    medidas = ["relative"] * 4
    textos = [f"${v:,.0f}" for v in [variacion_ingresos, variacion_costos, variacion_gastos, otras_variaciones]]

    fig = go.Figure(go.Waterfall(
        name="Variación", orientation="v",
        measure=["absolute"] + medidas + ["total"],
        x=["Utilidad Neta " + periodo_previo, "Ingresos", "Costos", "Gastos Op.", "Otros", "Utilidad Neta " + periodo_actual],
        text=[""] + textos + [""],
        y=[utilidad_neta_previa, variacion_ingresos, variacion_costos, variacion_gastos, otras_variaciones, utilidad_neta_actual],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#28a745"}},
        decreasing={"marker": {"color": "#dc3545"}},
    ))
    fig.update_layout(title=f"Puente de Utilidad Neta: {periodo_previo} vs {periodo_actual}", showlegend=False, yaxis_title="Monto (COP)", height=500)
    fig.update_yaxes(tickformat="$,.0f")
    return fig

# ==============================================================================
#                      CONFIGURACIÓN DE PÁGINA Y AUTENTICACIÓN
# ==============================================================================
st.set_page_config(layout="wide", page_title="CFO Virtual PRO con IA", initial_sidebar_state="expanded")

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
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
	.stTabs [data-baseweb="tab"] {
		height: 50px; white-space: pre-wrap; background-color: #F0F2F6;
		border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("👨‍🚀 CFO Virtual PRO: Análisis Financiero Aumentado")

try:
    real_password = st.secrets["general"]["password"]
except Exception:
    st.error("No se encontró la contraseña en los secretos de Streamlit."); st.stop()

if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if not st.session_state.authenticated:
    password = st.text_input("Introduce la contraseña para acceder:", type="password")
    if password == real_password:
        st.session_state.authenticated = True
        st.rerun()
    else:
        if password: st.warning("Contraseña incorrecta.")
        st.stop()

# ==============================================================================
#                       CARGA DE DATOS AUTOMÁTICA DESDE DROPBOX
# ==============================================================================
@st.cache_data(ttl=3600)
def cargar_y_procesar_datos_full():
    dbx = get_dropbox_client()
    if not dbx: return None
    archivos_financieros = find_financial_files(dbx, base_folder="/data")
    if not archivos_financieros:
        st.warning("No se encontraron archivos de Excel en la carpeta /data de Dropbox.")
        return None

    datos_historicos = {}
    progress_bar = st.progress(0, text="Iniciando carga...")
    for i, file_info in enumerate(archivos_financieros):
        periodo = file_info["periodo"]
        path = file_info["path"]
        progress_bar.progress((i + 1) / len(archivos_financieros), text=f"Procesando {periodo}...")
        excel_bytes = load_excel_from_dropbox(dbx, path)
        if excel_bytes:
            try:
                df_er, df_bg = procesar_archivo_excel(excel_bytes)
                datos_periodo = {'df_er_master': df_er, 'df_bg_master': df_bg, 'kpis': {}}
                er_conf = COL_CONFIG['ESTADO_DE_RESULTADOS']
                cc_cols = ['Todos'] + [name for name in er_conf.get('CENTROS_COSTO_COLS', {}).values() if name in df_er and name not in ['Total_Consolidado_ER', 'Sin centro de coste']]
                for cc in cc_cols:
                    datos_periodo['kpis'][cc] = calcular_kpis_periodo(df_er, df_bg, cc)
                datos_historicos[periodo] = datos_periodo
            except Exception as e:
                st.error(f"Error al procesar el archivo del periodo {periodo}: {e}")
    progress_bar.empty()
    return datos_historicos

if 'datos_historicos' not in st.session_state: st.session_state.datos_historicos = None
if st.sidebar.button("♻️ Refrescar Datos de Dropbox", use_container_width=True, type="primary"):
    st.cache_data.clear()
    st.session_state.datos_historicos = None
if st.session_state.datos_historicos is None:
    st.session_state.datos_historicos = cargar_y_procesar_datos_full()
if not st.session_state.datos_historicos:
    st.error("No se pudieron cargar datos. Verifica la conexión y la estructura de archivos en Dropbox.")
    st.stop()

# ==============================================================================
#                            INTERFAZ DE USUARIO PRINCIPAL
# ==============================================================================
st.sidebar.title("🚀 Opciones de Análisis")
sorted_periods = sorted(st.session_state.datos_historicos.keys(), reverse=True)
period_options = ["📈 Análisis Estratégico de Evolución"] + sorted_periods
selected_view = st.sidebar.selectbox("Selecciona la vista de análisis:", period_options)

# ==============================================================================
#                  VISTA 1: ANÁLISIS ESTRATÉGICO DE EVOLUCIÓN (TENDENCIAS)
# ==============================================================================
if selected_view == "📈 Análisis Estratégico de Evolución":
    st.header("📈 Informe de Evolución Gerencial")
    df_tendencia = preparar_datos_tendencia(st.session_state.datos_historicos)
    
    if df_tendencia.empty or len(df_tendencia) < 2:
        st.info("Se necesitan al menos dos periodos de datos para generar un análisis de evolución.")
        st.stop()

    with st.expander("🧠 **Diagnóstico Estratégico del CFO Virtual (IA)**", expanded=True):
        with st.spinner('El Analista Senior IA está evaluando la trayectoria plurianual...'):
            contexto_tendencia = {
                "datos_tendencia": df_tendencia.to_dict('records'),
                "kpi_principales": ['margen_neto', 'roe', 'razon_corriente', 'endeudamiento_activo', 'ingresos', 'utilidad_neta'],
                "volatilidad": {
                    k: (df_tendencia[k].std() / df_tendencia[k].mean()) if df_tendencia[k].mean() != 0 else 0
                    for k in ['margen_neto', 'roe', 'ingresos']
                },
                "correlaciones": df_tendencia[['ingresos', 'gastos_operativos', 'utilidad_neta', 'margen_neto']].corr().to_dict(),
                "convencion_contable": "Estándar P&L (Ingresos+, Gastos-), Sistema BS (Activos+, Pasivos-).",
                "pregunta_clave": (
                    "Actúa como un CFO estratégico. Analiza la evolución financiera de la empresa basándote en los datos de tendencia, volatilidad y correlaciones. "
                    "1. **Narrativa de Crecimiento:** ¿El crecimiento de ingresos es saludable y rentable? ¿Se está acelerando o desacelerando? "
                    "2. **Narrativa de Rentabilidad:** ¿Cómo ha evolucionado el ROE y el Margen Neto? ¿Qué tan estable es? ¿Qué impulsa la rentabilidad (ventas o eficiencia)? "
                    "3. **Narrativa de Solvencia:** ¿La posición de liquidez y endeudamiento es más fuerte o más débil con el tiempo? "
                    "4. **Punto de Inflexión:** Identifica el periodo más crítico donde una tendencia clave cambió (positiva o negativamente). "
                    "5. **Conclusión y Foco Estratégico:** Resume el estado financiero general y recomienda las 2-3 prioridades estratégicas clave para la gerencia."
                )
            }
            analisis_tendencia_ia = generar_analisis_tendencia_ia(contexto_tendencia) 
            st.markdown(f"<div class='ai-analysis-text'>{analisis_tendencia_ia}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Indicadores Clave de Desempeño (KPIs) a través del Tiempo")

    kpi_cols = st.columns(4)
    kpi_definitions = {
        'margen_neto': {'title': 'Margen Neto', 'is_percent': True, 'lower_is_better': False},
        'roe': {'title': 'ROE (Retorno sobre Patrimonio)', 'is_percent': True, 'lower_is_better': False},
        'razon_corriente': {'title': 'Razón Corriente (Liquidez)', 'is_percent': False, 'lower_is_better': False},
        'endeudamiento_activo': {'title': 'Endeudamiento del Activo', 'is_percent': True, 'lower_is_better': True}
    }

    for i, (kpi, config) in enumerate(kpi_definitions.items()):
        with kpi_cols[i]:
            last_val = df_tendencia[kpi].iloc[-1]
            consistency = (df_tendencia[kpi].std() / df_tendencia[kpi].mean()) if df_tendencia[kpi].mean() != 0 else 0
            
            st.markdown(f"<p class='kpi-title'>{config['title']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='kpi-value'>{last_val:.2% if config['is_percent'] else f'{last_val:.2f}'}</p>", unsafe_allow_html=True)
            st.markdown(f"**Consistencia:** {'Baja' if consistency > 0.2 else 'Media' if consistency > 0.1 else 'Alta'}")
            st.plotly_chart(plot_enhanced_sparkline(df_tendencia[kpi], config['title'], config['is_percent'], config['lower_is_better']), use_container_width=True)

    st.markdown("---")
    st.subheader("Diagnósticos Visuales Estratégicos")
    
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.plotly_chart(plot_growth_profitability_matrix(df_tendencia), use_container_width=True)
    with v_col2:
        st.plotly_chart(plot_roe_decomposition_waterfall(df_tendencia), use_container_width=True)

    fig_combinada = go.Figure()
    fig_combinada.add_trace(go.Bar(x=df_tendencia['periodo'], y=df_tendencia['ingresos'], name='Ingresos', marker_color='#28a745'))
    fig_combinada.add_trace(go.Bar(x=df_tendencia['periodo'], y=df_tendencia['gastos_operativos'].abs(), name='Gastos Operativos', marker_color='#ffc107'))
    fig_combinada.add_trace(go.Scatter(x=df_tendencia['periodo'], y=df_tendencia['utilidad_neta'], name='Utilidad Neta', mode='lines+markers', line=dict(color='#0d6efd', width=4), yaxis="y2"))
    
    fig_combinada.update_layout(
        title='Evolución de Componentes del Resultado', barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title='Ingresos y Gastos (COP)'),
        yaxis2=dict(title='Utilidad Neta (COP)', overlaying='y', side='right', showgrid=False, tickformat="$,.0f"),
        height=500
    )
    st.plotly_chart(fig_combinada, use_container_width=True)

# ==============================================================================
#                 VISTA 2: CENTRO DE ANÁLISIS PROFUNDO (PERIODO ÚNICO)
# ==============================================================================
else:
    st.header(f"Centro de Análisis para el Periodo: {selected_view}")
    
    data_actual = st.session_state.datos_historicos.get(selected_view)
    if not data_actual:
        st.error(f"No se encontraron datos para el periodo: {selected_view}"); st.stop()

    periodo_actual_idx = sorted_periods.index(selected_view)
    periodo_previo = sorted_periods[periodo_actual_idx + 1] if periodo_actual_idx + 1 < len(sorted_periods) else None
    data_previa = st.session_state.datos_historicos.get(periodo_previo) if periodo_previo else None

    df_er_actual = data_actual['df_er_master']
    df_bg_actual = data_actual['df_bg_master']
    kpis_por_cc = data_actual['kpis']

    st.sidebar.subheader("Filtros del Periodo")
    cc_options_all = sorted(list(kpis_por_cc.keys()))
    cc_filter = st.sidebar.selectbox("Filtrar por Centro de Costo:", cc_options_all, key=f"cc_{selected_view}")
    
    er_conf = COL_CONFIG['ESTADO_DE_RESULTADOS']
    nivel_col = er_conf.get('NIVEL_LINEA', 'Grupo')
    if nivel_col in df_er_actual.columns:
        max_nivel = int(df_er_actual[nivel_col].max())
        nivel_seleccionado = st.sidebar.slider("Nivel de Detalle de Cuentas:", 1, max_nivel, 2, key=f"nivel_er_{selected_view}")
    else:
        nivel_seleccionado = 2

    st.sidebar.subheader("Buscador de Cuentas")
    search_account_input = st.sidebar.text_input("Buscar por número de cuenta:", key=f"search_{selected_view}", placeholder="Ej: 510506")
    
    df_variacion_er = None
    if data_previa:
        df_variacion_er = calcular_variaciones_er(df_er_actual, data_previa['df_er_master'], cc_filter)
        if df_variacion_er is not None and not df_variacion_er.empty:
            st.info(f"Análisis comparativo contra el periodo **{periodo_previo}**.")
    else:
        st.warning("No hay un periodo anterior para realizar análisis comparativo.")

    tab_list = [
        "📊 Resumen Ejecutivo IA", "🎯 Análisis DuPont y ROE", "💰 Análisis de Utilidad",
        "🧾 Análisis de Gastos", "🔬 Simulador de Escenarios", "📋 Reportes Financieros"
    ]
    tab_gen, tab_roe, tab_utilidad, tab_gas, tab_simulador, tab_rep = st.tabs(tab_list)

    with tab_gen:
        st.subheader(f"Resumen Ejecutivo para: {cc_filter}")
        selected_kpis = kpis_por_cc.get(cc_filter, {})
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        kpi_col1.metric("Margen Neto", f"{selected_kpis.get('margen_neto', 0):.2%}")
        kpi_col2.metric("ROE", f"{selected_kpis.get('roe', 0):.2%}")
        kpi_col3.metric("Razón Corriente", f"{selected_kpis.get('razon_corriente', 0):.2f}")
        kpi_col4.metric("Endeudamiento", f"{selected_kpis.get('endeudamiento_activo', 0):.2%}")

        st.markdown("---")
        with st.expander("🧠 **Diagnóstico y Recomendaciones del CFO Virtual (IA)**", expanded=True):
            with st.spinner('El CFO Virtual está preparando un análisis profundo...'):
                contexto_ia = {
                    "kpis_actuales": selected_kpis,
                    "kpis_previos": data_previa['kpis'].get(cc_filter, {}) if data_previa else {},
                    "periodo_actual": selected_view, "periodo_previo": periodo_previo, "centro_costo": cc_filter,
                    "convencion_contable": "P&L estándar (Ingresos+, Gastos-). BS de sistema (Activos+, Pasivos-).",
                }
                if df_variacion_er is not None and not df_variacion_er.empty:
                    top_favorables = df_variacion_er.nlargest(5, 'Variacion_Absoluta')
                    top_desfavorables = df_variacion_er.nsmallest(5, 'Variacion_Absoluta')
                    contexto_ia["variaciones_favorables"] = top_favorables[['Descripción', 'Variacion_Absoluta']].to_dict('records')
                    contexto_ia["variaciones_desfavorables"] = top_desfavorables[['Descripción', 'Variacion_Absoluta']].to_dict('records')

                contexto_ia["pregunta_clave"] = (
                    "Actúa como un CFO incisivo. Analiza los KPIs y las variaciones de este periodo. "
                    "1. **Diagnóstico de Salud Financiera:** ¿Cuál es la condición general (Excelente, Buena, Preocupante, Crítica)? Justifica con los 2-3 KPIs más relevantes. "
                    "2. **Análisis Causa-Raíz (ROE):** El ROE cambió. Descompónlo usando la lógica DuPont. ¿Fue por margen, rotación de activos o apalancamiento? Sé explícito. "
                    "3. **Motores del Cambio:** ¿Qué 3 cuentas específicas explican la mayor parte de la variación en la utilidad neta vs el periodo anterior? "
                    "4. **Preguntas para la Gerencia:** Formula 3 preguntas directas y basadas en datos que un gerente debería responder. (Ej: 'Vemos que los gastos de viaje aumentaron un 30% mientras los ingresos solo un 5%. ¿Cuál fue el ROI de esta inversión?'). "
                    "5. **Riesgo y Oportunidad Clave:** Identifica el riesgo financiero más grande y la oportunidad más clara que revelan estos números."
                )
                analisis_ia = generar_analisis_avanzado_ia(contexto_ia)
                st.markdown(f"<div class='ai-analysis-text'>{analisis_ia}</div>", unsafe_allow_html=True)

    with tab_roe:
        st.subheader("🎯 Análisis de Rentabilidad (ROE) con Modelo DuPont")
        st.markdown("El **Análisis DuPont** descompone el ROE en tres palancas: eficiencia operativa (Margen Neto), eficiencia en el uso de activos (Rotación) y apalancamiento financiero. Permite entender *por qué* cambió la rentabilidad.")
        
        if data_previa:
            kpis_actuales = kpis_por_cc.get(cc_filter, {})
            kpis_previos = data_previa['kpis'].get(cc_filter, {})
            dupont_data = {
                'Componente': ['Margen Neto (Eficiencia Op.)', 'Rotación de Activos (Uso de Activos)', 'Apalancamiento Financiero (Deuda)', 'ROE (Resultado Final)'],
                'Formula': ['Utilidad Neta / Ingresos', 'Ingresos / Activos Totales', 'Activos Totales / Patrimonio', 'Margen * Rotación * Apalanc.'],
                selected_view: [kpis_actuales.get(k, 0) for k in ['margen_neto', 'rotacion_activos', 'apalancamiento', 'roe']],
                periodo_previo: [kpis_previos.get(k, 0) for k in ['margen_neto', 'rotacion_activos', 'apalancamiento', 'roe']]
            }
            df_dupont = pd.DataFrame(dupont_data)
            df_dupont['Variación'] = df_dupont[selected_view] - df_dupont[periodo_previo]
            
            st.dataframe(
                df_dupont.style.format({
                    selected_view: '{:.2%}', periodo_previo: '{:.2%}', 'Variación': '{:+.2%}',
                    'Rotación de Activos (Uso de Activos)': '{:.2f}x', 'Apalancamiento Financiero (Deuda)': '{:.2f}x'
                }).background_gradient(cmap='RdYlGn', subset=['Variación'], low=0.4, high=0.4),
                use_container_width=True
            )
        else:
            st.info("Se requiere un periodo anterior para el análisis DuPont comparativo.")

    with tab_utilidad:
        st.subheader("💰 Análisis de la Utilidad Neta: ¿Qué movió el resultado?")
        if df_variacion_er is not None and not df_variacion_er.empty:
            st.plotly_chart(plot_waterfall_utilidad_neta(df_variacion_er, selected_view, periodo_previo), use_container_width=True)
            
            st.markdown("#### Principales Motores del Cambio vs. Periodo Anterior")
            col1, col2 = st.columns(2)
            
            top_favorables = df_variacion_er[df_variacion_er['Variacion_Absoluta'] > 0].sort_values('Variacion_Absoluta', ascending=False).head(10)
            top_desfavorables = df_variacion_er[df_variacion_er['Variacion_Absoluta'] < 0].sort_values('Variacion_Absoluta').head(10)
            format_dict = {'Valor_previo': '${:,.0f}', 'Valor_actual': '${:,.0f}', 'Variacion_Absoluta': '${:,.0f}'}

            with col1:
                st.markdown("✅ **Impactos Positivos (Ayudaron a la Utilidad)**")
                st.dataframe(top_favorables[['Descripción', 'Valor_actual', 'Valor_previo', 'Variacion_Absoluta']].style.format(format_dict).background_gradient(cmap='Greens', subset=['Variacion_Absoluta']), use_container_width=True)
            with col2:
                st.markdown("❌ **Impactos Negativos (Perjudicaron la Utilidad)**")
                st.dataframe(top_desfavorables[['Descripción', 'Valor_actual', 'Valor_previo', 'Variacion_Absoluta']].style.format(format_dict).background_gradient(cmap='Reds_r', subset=['Variacion_Absoluta']), use_container_width=True)
        else:
            st.info("Se requiere un periodo anterior para este análisis.")
    
    with tab_gas:
        st.subheader("🧾 Análisis Detallado de Gastos y Costos")
        valor_col_nombre = 'Total_Consolidado_ER' if cc_filter == 'Todos' else cc_filter

        if valor_col_nombre in df_er_actual.columns:
            # Asumiendo que 'Tipo' existe en el df_er_actual (ej: Gasto, Costo)
            if 'Tipo' not in df_er_actual.columns:
                df_er_actual['Tipo'] = np.where(df_er_actual['Cuenta'].astype(str).str.startswith('5'), 'Gasto', 'Costo')
            
            df_gastos_costos = df_er_actual[df_er_actual['Cuenta'].astype(str).str.match('^[56]')]
            st.markdown("#### Composición de Gastos y Costos del Periodo")
            
            fig_treemap = px.treemap(
                df_gastos_costos, 
                path=[px.Constant("Total Gastos y Costos"), 'Tipo', 'Descripción'], 
                values=df_gastos_costos[valor_col_nombre].abs(),
                title='Distribución de Gastos y Costos (Haga clic para explorar)',
                color=df_gastos_costos[valor_col_nombre].abs(),
                color_continuous_scale='Reds'
            )
            fig_treemap.update_layout(margin = dict(t=50, l=25, r=25, b=25))
            st.plotly_chart(fig_treemap, use_container_width=True)

        if df_variacion_er is not None and not df_variacion_er.empty:
            st.markdown("#### Variación de Gastos y Costos vs. Periodo Anterior")
            df_gc_var = df_variacion_er[df_variacion_er['Cuenta'].astype(str).str.match('^[56]')]
            st.dataframe(df_gc_var[['Descripción', 'Valor_actual', 'Valor_previo', 'Variacion_Absoluta']].style.format(format_dict), use_container_width=True)

    with tab_simulador:
        st.subheader("🔬 Simulador de Escenarios de Rentabilidad")
        st.markdown("Use los deslizadores para modelar el impacto de cambios en los componentes clave del resultado. Vea cómo cada cambio afecta su utilidad y márgenes en tiempo real.")

        valor_col_nombre = 'Total_Consolidado_ER' if cc_filter == 'Todos' else cc_filter
        
        sim_col1, sim_col2 = st.columns([1, 2])
        
        with sim_col1:
            st.markdown("#### Controles de Simulación")
            ingresos_change = st.slider("Cambio % en Ingresos Totales", -25.0, 25.0, 0.0, 0.5)
            costos_change = st.slider("Cambio % en Costo de Ventas (COGS)", -25.0, 25.0, 0.0, 0.5)
            gastos_change = st.slider("Cambio % en Gastos Operativos", -25.0, 25.0, 0.0, 0.5)
            
            changes = {'ingresos': ingresos_change, 'costos': costos_change, 'gastos': gastos_change}

        sim_kpis = run_what_if_scenario(df_er_actual, valor_col_nombre, changes)
        actual_kpis = kpis_por_cc.get(cc_filter, {})

        with sim_col2:
            st.markdown("#### Resultados de la Simulación")
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric("Utilidad Neta Simulada", 
                          f"${sim_kpis.get('Utilidad Neta', 0):,.0f}",
                          f"{sim_kpis.get('Utilidad Neta', 0) - actual_kpis.get('utilidad_neta', 0):,.0f} vs. Actual")
            with res_col2:
                 st.metric("Margen Neto Simulado", 
                          f"{sim_kpis.get('Margen Neto', 0):.2%}",
                          f"{sim_kpis.get('Margen Neto', 0) - actual_kpis.get('margen_neto', 0):+.2%} vs. Actual")
            with res_col3:
                 st.metric("Margen Operativo Simulado", 
                          f"{sim_kpis.get('Margen Operativo', 0):.2%}",
                          f"{sim_kpis.get('Margen Operativo', 0) - actual_kpis.get('margen_operativo', 0):+.2%} vs. Actual")

            st.markdown("---")
            st.write("Este análisis le permite cuantificar el impacto de decisiones estratégicas como campañas de marketing, negociaciones con proveedores o planes de eficiencia.")

    with tab_rep:
        st.subheader("📊 Reportes Financieros Detallados")
        
        st.markdown(f"#### Estado de Resultados (Nivel de Detalle: {nivel_seleccionado})")
        df_er_display = generate_financial_statement(df_er_actual, 'Estado de Resultados', cc_filter, nivel_seleccionado)
        st.dataframe(df_er_display.style.format({'Valor': "${:,.0f}"}), use_container_width=True, height=600)

        st.markdown("#### Balance General")
        df_bg_display = generate_financial_statement(df_bg_actual, 'Balance General', 99)
        st.dataframe(df_bg_display.style.format({'Valor': "${:,.0f}"}), use_container_width=True, height=600)
    
        st.markdown("#### Estado de Flujo de Caja (Método Indirecto)")
        if data_previa:
            with st.spinner("Construyendo Flujo de Caja..."):
                df_flujo = construir_flujo_de_caja(
                    df_er_actual, df_bg_actual, data_previa['df_bg_master'], 
                    'Total_Consolidado_ER' if cc_filter == 'Todos' else cc_filter,
                    er_conf['CUENTA'], COL_CONFIG['BALANCE_GENERAL']['SALDO_FINAL'], COL_CONFIG['BALANCE_GENERAL']['CUENTA']
                )
                st.dataframe(df_flujo.style.format({'Valor': '${:,.0f}'}), use_container_width=True)
        else:
            st.info("Se requiere un periodo anterior para generar el Estado de Flujo de Caja.")

    if search_account_input:
        st.markdown("---")
        with st.expander(f"Resultado de la búsqueda para cuentas que inician con '{search_account_input}'", expanded=True):
            cuenta_col_er = er_conf.get('CUENTA', 'Cuenta')
            cuenta_col_bg = COL_CONFIG['BALANCE_GENERAL'].get('CUENTA', 'Cuenta')
            
            st.write("**Estado de Resultados**")
            df_search_er = df_er_actual[df_er_actual[cuenta_col_er].astype(str).str.startswith(search_account_input)]
            st.dataframe(df_search_er) if not df_search_er.empty else st.info(f"No se encontraron cuentas en el ER para '{search_account_input}'.")
            
            st.write("**Balance General**")
            df_search_bg = df_bg_actual[df_bg_actual[cuenta_col_bg].astype(str).str.startswith(search_account_input)]
            st.dataframe(df_search_bg) if not df_search_bg.empty else st.info(f"No se encontraron cuentas en el BG para '{search_account_input}'.")

    st.sidebar.markdown("---")
    er_to_dl = generate_financial_statement(df_er_actual, 'Estado de Resultados', cc_filter, 99)
    bg_to_dl = generate_financial_statement(df_bg_actual, 'Balance General', 99)
    excel_buffer = to_excel_buffer(er_to_dl, bg_to_dl)
    st.sidebar.download_button(
        label=f"📥 Descargar Reportes ({selected_view}, {cc_filter})",
        data=excel_buffer,
        file_name=f"Reporte_Financiero_{selected_view}_{cc_filter}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
