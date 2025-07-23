# Dashboard_Principal.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List

# --- Importamos nuestros módulos ---
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
def plot_enhanced_sparkline(data: pd.Series, title: str, is_percent: bool = False, lower_is_better: bool = False):
    if data.empty or len(data.dropna()) < 2:
        fig = go.Figure().update_layout(width=200, height=70, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig.add_annotation(text="N/A", showarrow=False)
        return fig

    last_val = data.iloc[-1]
    color = '#28a745' if (not lower_is_better and last_val > data.iloc[0]) or (lower_is_better and last_val < data.iloc[0]) else '#dc3545'

    fig = go.Figure(go.Scatter(
        x=list(range(len(data))), y=data, mode='lines',
        line=dict(color=color, width=3),
        fill='tozeroy',
        fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)"
    ))
    fig.update_layout(
        width=200, height=70, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=5, r=5, t=5, b=5), xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False
    )
    return fig

def plot_growth_profitability_matrix(df_tendencia: pd.DataFrame):
    df = df_tendencia.copy()
    df['crecimiento_ingresos_pct'] = df['ingresos'].pct_change() * 100
    df['crecimiento_utilidad_pct'] = df['utilidad_neta'].pct_change() * 100
    df.dropna(inplace=True)

    if df.empty:
        return go.Figure().add_annotation(text="Datos insuficientes para la matriz de crecimiento", showarrow=False)

    fig = px.scatter(
        df, x='crecimiento_ingresos_pct', y='crecimiento_utilidad_pct', text='periodo',
        title="Matriz de Crecimiento vs. Rentabilidad",
        labels={'crecimiento_ingresos_pct': 'Crecimiento de Ingresos (%)', 'crecimiento_utilidad_pct': 'Crecimiento de Utilidad Neta (%)'}
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_traces(textposition='top center')
    fig.update_layout(height=500)
    return fig

def plot_roe_decomposition_waterfall(df_tendencia: pd.DataFrame):
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
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["ROE Inicial", "Impacto Margen", "Impacto Rotación Activos", "Impacto Apalancamiento", "ROE Final"],
        y=[start_roe, contrib_margin, contrib_turnover, contrib_leverage, end_roe],
        text=[f"{y:.2%}" for y in [start_roe, contrib_margin, contrib_turnover, contrib_leverage, end_roe]],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#28a745"}},
        decreasing={"marker": {"color": "#dc3545"}},
    ))
    fig.update_layout(
        title=f"Puente de Rentabilidad (ROE): {df_tendencia['periodo'].iloc[0].strftime('%Y-%m')} vs {df_tendencia['periodo'].iloc[-1].strftime('%Y-%m')}",
        yaxis_title="Retorno sobre Patrimonio (ROE)", yaxis_tickformat='.2%'
    )
    return fig

def run_what_if_scenario(df_er: pd.DataFrame, value_col: str, changes: Dict[str, float]):
    sim_er = df_er.copy()
    cuenta_col = COL_CONFIG['ESTADO_DE_RESULTADOS']['CUENTA']
    
    base_ingresos = sim_er[sim_er[cuenta_col].astype(str).str.startswith('4')][value_col].sum()
    base_costos = sim_er[sim_er[cuenta_col].astype(str).str.startswith('6')][value_col].sum()
    base_gastos = sim_er[sim_er[cuenta_col].astype(str).str.startswith('5')][value_col].sum()
    
    sim_ingresos = base_ingresos * (1 + changes['ingresos'] / 100)
    sim_costos = base_costos * (1 + changes['costos'] / 100)
    sim_gastos = base_gastos * (1 + changes['gastos'] / 100)

    sim_utilidad_bruta = sim_ingresos + sim_costos
    sim_utilidad_operativa = sim_utilidad_bruta + sim_gastos
    
    otros_neto = sim_er[~sim_er[cuenta_col].astype(str).str.match('^[456]')][value_col].sum()
    sim_utilidad_neta = sim_utilidad_operativa + otros_neto

    kpis = {
        'Utilidad Neta': sim_utilidad_neta,
        'Margen Neto': (sim_utilidad_neta / sim_ingresos) if sim_ingresos != 0 else 0,
        'Margen Operativo': (sim_utilidad_operativa / sim_ingresos) if sim_ingresos != 0 else 0
    }
    return kpis

def calcular_variaciones_er(df_actual: pd.DataFrame, df_previo: pd.DataFrame, cc_filter: str) -> pd.DataFrame:
    er_conf = COL_CONFIG['ESTADO_DE_RESULTADOS']
    cuenta_col = er_conf.get('CUENTA', 'Cuenta')
    desc_col = er_conf.get('DESCRIPCION_CUENTA', 'Título')
    valor_col_nombre = 'Total_Consolidado_ER' if cc_filter == 'Todos' else cc_filter

    required_cols = [cuenta_col, desc_col, valor_col_nombre]
    if any(col not in df_actual.columns for col in required_cols):
        return pd.DataFrame()

    df1 = df_actual[required_cols].copy()
    df1.rename(columns={valor_col_nombre: 'Valor_actual'}, inplace=True)
    
    df2 = df_previo[[cuenta_col, desc_col, valor_col_nombre]].copy() if all(c in df_previo for c in required_cols) else df_actual[[cuenta_col, desc_col]].assign(Valor_previo=0)
    if valor_col_nombre in df2.columns:
        df2.rename(columns={valor_col_nombre: 'Valor_previo'}, inplace=True)

    df_variacion = pd.merge(df1, df2, on=[cuenta_col, desc_col], how='outer').fillna(0)
    df_variacion['Variacion_Absoluta'] = df_variacion['Valor_actual'] - df_variacion['Valor_previo']
    
    if desc_col != 'Descripción':
        df_variacion.rename(columns={desc_col: 'Descripción'}, inplace=True)
    
    return df_variacion

def plot_waterfall_utilidad_neta(df_variacion: pd.DataFrame, periodo_actual: str, periodo_previo: str):
    cuenta_col = 'Cuenta' 
    if cuenta_col not in df_variacion.columns:
        return go.Figure()

    utilidad_neta_actual = df_variacion['Valor_actual'].sum()
    utilidad_neta_previa = df_variacion['Valor_previo'].sum()

    variacion_ingresos = df_variacion[df_variacion[cuenta_col].astype(str).str.startswith('4')]['Variacion_Absoluta'].sum()
    variacion_costos = df_variacion[df_variacion[cuenta_col].astype(str).str.startswith('6')]['Variacion_Absoluta'].sum()
    variacion_gastos = df_variacion[df_variacion[cuenta_col].astype(str).str.startswith('5')]['Variacion_Absoluta'].sum()
    otras_variaciones = df_variacion['Variacion_Absoluta'].sum() - (variacion_ingresos + variacion_costos + variacion_gastos)

    fig = go.Figure(go.Waterfall(
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=[f"Utilidad Neta {periodo_previo}", "Ingresos", "Costos", "Gastos Op.", "Otros", f"Utilidad Neta {periodo_actual}"],
        y=[utilidad_neta_previa, variacion_ingresos, variacion_costos, variacion_gastos, otras_variaciones, utilidad_neta_actual],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#28a745"}},
        decreasing={"marker": {"color": "#dc3545"}},
    ))
    fig.update_layout(title=f"Puente de Utilidad Neta: {periodo_previo} vs {periodo_actual}", yaxis_title="Monto (COP)")
    return fig

# ==============================================================================
#                      CONFIGURACIÓN DE PÁGINA Y AUTENTICACIÓN
# ==============================================================================
st.set_page_config(layout="wide", page_title="CFO Virtual PRO con IA", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main .block-container {padding-top: 2rem;}
    .kpi-title {font-size: 1.1em; font-weight: 600; color: #5f6368;}
    .kpi-value {font-size: 2.2em; font-weight: 700; color: #202124;}
    .ai-analysis-text {
        background-color: #e8f0fe; border-left: 5px solid #1967d2; padding: 20px;
        border-radius: 8px; font-size: 1.1em; line-height: 1.6;
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
    password = st.text_input("Introduce la contraseña:", type="password")
    if password == real_password:
        st.session_state.authenticated = True
        st.rerun()
    elif password:
        st.warning("Contraseña incorrecta.")
        st.stop()
    else:
        st.stop()

# ==============================================================================
#                       CARGA DE DATOS AUTOMÁTICA
# ==============================================================================
@st.cache_data(ttl=3600)
def cargar_y_procesar_datos_full():
    dbx = get_dropbox_client()
    if not dbx: return None
    archivos_financieros = find_financial_files(dbx, base_folder="/data")
    if not archivos_financieros:
        st.warning("No se encontraron archivos de Excel en Dropbox.")
        return None

    datos_historicos = {}
    progress_bar = st.progress(0, text="Cargando datos...")
    for i, file_info in enumerate(archivos_financieros):
        periodo = file_info["periodo"]
        progress_bar.progress((i + 1) / len(archivos_financieros), text=f"Procesando {periodo}...")
        excel_bytes = load_excel_from_dropbox(dbx, file_info["path"])
        if excel_bytes:
            try:
                df_er, df_bg = procesar_archivo_excel(excel_bytes)
                datos_periodo = {'df_er_master': df_er, 'df_bg_master': df_bg, 'kpis': {}}
                cc_cols = ['Todos'] + [v for k, v in COL_CONFIG['ESTADO_DE_RESULTADOS'].get('CENTROS_COSTO_COLS', {}).items() if str(k).lower() != 'total' and v in df_er]
                for cc in cc_cols:
                    datos_periodo['kpis'][cc] = calcular_kpis_periodo(df_er, df_bg, cc)
                datos_historicos[periodo] = datos_periodo
            except Exception as e:
                st.error(f"Error al procesar el archivo del periodo {periodo}: {e}")
    progress_bar.empty()
    return datos_historicos

if 'datos_historicos' not in st.session_state: st.session_state.datos_historicos = None
if st.sidebar.button("♻️ Refrescar Datos", use_container_width=True, type="primary"):
    st.cache_data.clear()
    st.session_state.datos_historicos = None
if st.session_state.datos_historicos is None:
    st.session_state.datos_historicos = cargar_y_procesar_datos_full()
if not st.session_state.datos_historicos:
    st.error("No se pudieron cargar datos. Verifique la conexión y los archivos en Dropbox.")
    st.stop()

# ==============================================================================
#                            INTERFAZ DE USUARIO PRINCIPAL
# ==============================================================================
st.sidebar.title("🚀 Opciones de Análisis")
sorted_periods = sorted(st.session_state.datos_historicos.keys(), reverse=True)
period_options = ["📈 Análisis Estratégico de Evolución"] + sorted_periods
selected_view = st.sidebar.selectbox("Selecciona la vista de análisis:", period_options)

# ==============================================================================
#                  VISTA 1: ANÁLISIS ESTRATÉGICO DE EVOLUCIÓN
# ==============================================================================
if selected_view == "📈 Análisis Estratégico de Evolución":
    st.header("📈 Informe de Evolución Gerencial")
    df_tendencia = preparar_datos_tendencia(st.session_state.datos_historicos)
    
    if df_tendencia.empty or len(df_tendencia) < 2:
        st.info("Se necesitan al menos dos periodos de datos para generar un análisis de evolución.")
        st.stop()

    with st.expander("🧠 **Diagnóstico Estratégico del CFO Virtual (IA)**", expanded=True):
        analisis_tendencia_ia = generar_analisis_tendencia_ia(df_tendencia) 
        st.markdown(f"<div class='ai-analysis-text'>{analisis_tendencia_ia}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Indicadores Clave de Desempeño (KPIs) a través del Tiempo")
    
    kpi_cols = st.columns(4)
    kpi_definitions = {
        'margen_neto': {'title': 'Margen Neto', 'is_percent': True, 'lower_is_better': False},
        'roe': {'title': 'ROE', 'is_percent': True, 'lower_is_better': False},
        'razon_corriente': {'title': 'Razón Corriente', 'is_percent': False, 'lower_is_better': False},
        'endeudamiento_activo': {'title': 'Endeudamiento', 'is_percent': True, 'lower_is_better': True}
    }
    for i, (kpi, config) in enumerate(kpi_definitions.items()):
        with kpi_cols[i]:
            last_val = df_tendencia[kpi].iloc[-1]
            st.markdown(f"<p class='kpi-title'>{config['title']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='kpi-value'>{last_val:.2% if config['is_percent'] else f'{last_val:.2f}'}</p>", unsafe_allow_html=True)
            st.plotly_chart(plot_enhanced_sparkline(df_tendencia[kpi], **config), use_container_width=True)

    st.markdown("---")
    st.subheader("Diagnósticos Visuales Estratégicos")
    
    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.plotly_chart(plot_growth_profitability_matrix(df_tendencia), use_container_width=True)
    with v_col2:
        st.plotly_chart(plot_roe_decomposition_waterfall(df_tendencia), use_container_width=True)
    
# ==============================================================================
#                 VISTA 2: CENTRO DE ANÁLISIS PROFUNDO (PERIODO ÚNICO)
# ==============================================================================
else:
    st.header(f"Centro de Análisis para el Periodo: {selected_view}")
    
    data_actual = st.session_state.datos_historicos.get(selected_view, {})
    periodo_actual_idx = sorted_periods.index(selected_view)
    periodo_previo = sorted_periods[periodo_actual_idx + 1] if periodo_actual_idx + 1 < len(sorted_periods) else None
    data_previa = st.session_state.datos_historicos.get(periodo_previo) if periodo_previo else None

    df_er_actual = data_actual.get('df_er_master', pd.DataFrame())
    df_bg_actual = data_actual.get('df_bg_master', pd.DataFrame())
    kpis_por_cc = data_actual.get('kpis', {})

    st.sidebar.subheader("Filtros del Periodo")
    cc_options_all = sorted(list(kpis_por_cc.keys()))
    cc_filter = st.sidebar.selectbox("Filtrar por Centro de Costo:", cc_options_all, key=f"cc_{selected_view}")
    
    er_conf = COL_CONFIG.get('ESTADO_DE_RESULTADOS', {})
    nivel_col = er_conf.get('NIVEL_LINEA', 'Grupo')
    if nivel_col in df_er_actual.columns:
        max_nivel = int(df_er_actual[nivel_col].max())
        nivel_seleccionado = st.sidebar.slider("Nivel de Detalle:", 1, max_nivel, 2, key=f"nivel_er_{selected_view}")
    else:
        nivel_seleccionado = 2

    df_variacion_er = None
    if data_previa:
        df_variacion_er = calcular_variaciones_er(df_er_actual, data_previa.get('df_er_master', pd.DataFrame()), cc_filter)
        if df_variacion_er is not None and not df_variacion_er.empty:
            st.info(f"Análisis comparativo contra el periodo **{periodo_previo}**.")

    tab_list = [
        "📊 Resumen Ejecutivo IA", "💰 Análisis de Utilidad",
        "🧾 Análisis de Gastos", "🔬 Simulador de Escenarios", "📋 Reportes Financieros"
    ]
    tab_gen, tab_utilidad, tab_gas, tab_simulador, tab_rep = st.tabs(tab_list)

    with tab_gen:
        st.subheader(f"Resumen Ejecutivo para: {cc_filter}")
        selected_kpis = kpis_por_cc.get(cc_filter, {})
        
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        kpi_col1.metric("Margen Neto", f"{selected_kpis.get('margen_neto', 0):.2%}")
        kpi_col2.metric("ROE", f"{selected_kpis.get('roe', 0):.2%}")
        kpi_col3.metric("Razón Corriente", f"{selected_kpis.get('razon_corriente', 0):.2f}")
        kpi_col4.metric("Endeudamiento", f"{selected_kpis.get('endeudamiento_activo', 0):.2%}")

        with st.expander("🧠 **Diagnóstico y Recomendaciones del CFO Virtual (IA)**", expanded=True):
            contexto_ia = {
                "kpis": selected_kpis, "periodo": selected_view, "centro_costo": cc_filter,
            }
            analisis_ia = generar_analisis_avanzado_ia(contexto_ia)
            st.markdown(f"<div class='ai-analysis-text'>{analisis_ia}</div>", unsafe_allow_html=True)

    with tab_utilidad:
        st.subheader("💰 Análisis de la Utilidad Neta")
        if df_variacion_er is not None and not df_variacion_er.empty:
            st.plotly_chart(plot_waterfall_utilidad_neta(df_variacion_er, selected_view, periodo_previo), use_container_width=True)
        else:
            st.info("Se requiere un periodo anterior para este análisis.")
    
    with tab_gas:
        st.subheader("🧾 Análisis Detallado de Gastos y Costos")
        valor_col_nombre = 'Total_Consolidado_ER' if cc_filter == 'Todos' else cc_filter

        if valor_col_nombre in df_er_actual.columns:
            # --- INICIO DEL BLOQUE CORREGIDO ---
            er_conf = COL_CONFIG.get('ESTADO_DE_RESULTADOS', {})
            cuenta_col = er_conf.get('CUENTA', 'Cuenta')
            desc_col_name = er_conf.get('DESCRIPCION_CUENTA', 'Título') # Obtener nombre dinámico

            df_gastos_costos = df_er_actual[df_er_actual[cuenta_col].astype(str).str.match('^[567]')].copy()
            
            # Asegurarse de que la columna 'Tipo' existe
            df_gastos_costos['Tipo'] = np.select(
                [
                    df_gastos_costos[cuenta_col].astype(str).str.startswith('5'),
                    df_gastos_costos[cuenta_col].astype(str).str.startswith('6'),
                    df_gastos_costos[cuenta_col].astype(str).str.startswith('7')
                ],
                [
                    'Gasto',
                    'Costo de Ventas',
                    'Costo de Producción'
                ],
                default='Otro'
            )
            
            if desc_col_name in df_gastos_costos.columns:
                fig_treemap = px.treemap(
                    df_gastos_costos, 
                    path=[px.Constant("Total Gastos y Costos"), 'Tipo', desc_col_name], # Usar variable dinámica
                    values=df_gastos_costos[valor_col_nombre].abs(),
                    title='Distribución de Gastos y Costos',
                    color=df_gastos_costos[valor_col_nombre].abs(),
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_treemap, use_container_width=True)
            else:
                st.warning(f"La columna de descripción '{desc_col_name}' no se encontró para generar el treemap.")
            # --- FIN DEL BLOQUE CORREGIDO ---

    with tab_simulador:
        st.subheader("🔬 Simulador de Escenarios de Rentabilidad")
        valor_col_nombre = 'Total_Consolidado_ER' if cc_filter == 'Todos' else cc_filter
        
        sim_col1, sim_col2 = st.columns([1, 2])
        with sim_col1:
            ingresos_change = st.slider("Cambio % en Ingresos", -25.0, 25.0, 0.0, 0.5)
            costos_change = st.slider("Cambio % en Costo de Ventas", -25.0, 25.0, 0.0, 0.5)
            gastos_change = st.slider("Cambio % en Gastos Operativos", -25.0, 25.0, 0.0, 0.5)
            changes = {'ingresos': ingresos_change, 'costos': costos_change, 'gastos': gastos_change}

        sim_kpis = run_what_if_scenario(df_er_actual, valor_col_nombre, changes)
        actual_kpis = kpis_por_cc.get(cc_filter, {})

        with sim_col2:
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Utilidad Neta Simulada", f"${sim_kpis.get('Utilidad Neta', 0):,.0f}", f"{sim_kpis.get('Utilidad Neta', 0) - actual_kpis.get('utilidad_neta', 0):,.0f} vs. Actual")
            res_col2.metric("Margen Neto Simulado", f"{sim_kpis.get('Margen Neto', 0):.2%}", f"{sim_kpis.get('Margen Neto', 0) - actual_kpis.get('margen_neto', 0):+.2%} vs. Actual")
    
    with tab_rep:
        st.subheader("📊 Reportes Financieros Detallados")
        
        st.markdown("#### Estado de Resultados")
        df_er_display = generate_financial_statement(df_er_actual, 'Estado de Resultados', cc_filter, nivel_seleccionado)
        st.dataframe(df_er_display.style.format({'Valor': "${:,.0f}"}), use_container_width=True, height=600)

        st.markdown("#### Balance General")
        df_bg_display = generate_financial_statement(df_bg_actual, 'Balance General', 99)
        st.dataframe(df_bg_display.style.format({'Valor': "${:,.0f}"}), use_container_width=True, height=600)
