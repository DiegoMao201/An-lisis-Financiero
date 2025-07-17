# Dashboard_Principal.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any
import google.generativeai as genai

# ==============================================================================
#                      MÓDULOS DE LÓGICA ORIGINAL
# ==============================================================================
# Para que este script sea autocontenido, asumimos que estas funciones
# existen y funcionan como se espera. Las importaciones originales se mantienen
# por si prefieres tener los archivos separados en el futuro.

from mi_logica_original import (
    procesar_archivo_excel,
    generate_financial_statement,
    to_excel_buffer,
    COL_CONFIG,
    get_principal_account_value
)
from dropbox_connector import (
    get_dropbox_client,
    find_financial_files,
    load_excel_from_dropbox
)

# ==============================================================================
#      NOTACIÓN CONTABLE IMPORTANTE PARA EL ANÁLISIS
# ==============================================================================
# En todo el análisis se asume la siguiente convención para el Estado de Resultados:
# - INGRESOS y UTILIDADES se representan con valores NEGATIVOS (favorable).
# - GASTOS y PÉRDIDAS se representan con valores POSITIVOS (desfavorable).
# Las funciones de análisis y visualización están diseñadas para interpretar esta lógica.

# ==============================================================================
#        FUNCIONES DE CÁLCULO DE KPIS Y ANÁLISIS (Integradas)
# ==============================================================================

def calcular_kpis_periodo(df_er: pd.DataFrame, df_bg: pd.DataFrame, cc_filter: str = 'Todos') -> dict:
    """
    Calcula un set de KPIs para un único periodo.
    Se adapta para calcular el consolidado ('Todos') o un centro de costo específico.
    """
    kpis = {}
    er_conf = COL_CONFIG['ESTADO_DE_RESULTADOS']
    bg_conf = COL_CONFIG['BALANCE_GENERAL']
    valor_col_kpi = 'Total_Consolidado_ER' if cc_filter == 'Todos' else cc_filter

    if not valor_col_kpi or valor_col_kpi not in df_er.columns: return kpis

    cuenta_er = er_conf['CUENTA']
    ingresos = abs(get_principal_account_value(df_er, '4', valor_col_kpi, cuenta_er))
    costo_ventas = get_principal_account_value(df_er, '6', valor_col_kpi, cuenta_er)
    gastos_admin = get_principal_account_value(df_er, '51', valor_col_kpi, cuenta_er)
    gastos_ventas = get_principal_account_value(df_er, '52', valor_col_kpi, cuenta_er)
    
    utilidad_bruta = ingresos - costo_ventas
    gastos_operativos = gastos_admin + gastos_ventas
    utilidad_operacional = utilidad_bruta - gastos_operativos
    
    # Suma de todos los ingresos (negativos) y todos los gastos/costos (positivos)
    utilidad_neta = -df_er[df_er[cuenta_er].str.startswith(('4','5','6','7'))][valor_col_kpi].sum()
    
    kpis['ingresos'] = ingresos
    kpis['costo_ventas'] = costo_ventas
    kpis['gastos_operativos'] = gastos_operativos
    kpis['utilidad_operacional'] = utilidad_operacional
    kpis['utilidad_neta'] = utilidad_neta
    
    cuenta_bg = bg_conf['CUENTA']
    saldo_final_col = bg_conf['SALDO_FINAL']

    activo = get_principal_account_value(df_bg, '1', saldo_final_col, cuenta_bg)
    pasivo = get_principal_account_value(df_bg, '2', saldo_final_col, cuenta_bg)
    patrimonio = get_principal_account_value(df_bg, '3', saldo_final_col, cuenta_bg)
    activo_corriente = get_principal_account_value(df_bg, '11', saldo_final_col, cuenta_bg) + \
                       get_principal_account_value(df_bg, '13', saldo_final_col, cuenta_bg) + \
                       get_principal_account_value(df_bg, '14', saldo_final_col, cuenta_bg)
    pasivo_corriente = get_principal_account_value(df_bg, '21', saldo_final_col, cuenta_bg) + \
                       get_principal_account_value(df_bg, '22', saldo_final_col, cuenta_bg) + \
                       get_principal_account_value(df_bg, '23', saldo_final_col, cuenta_bg)

    kpis['activo'] = activo
    kpis['pasivo'] = pasivo
    kpis['patrimonio'] = patrimonio
    
    kpis['razon_corriente'] = activo_corriente / pasivo_corriente if pasivo_corriente != 0 else 0
    kpis['endeudamiento_activo'] = pasivo / activo if activo != 0 else 0
    kpis['roe'] = utilidad_neta / patrimonio if patrimonio != 0 else 0
    kpis['margen_neto'] = utilidad_neta / ingresos if ingresos != 0 else 0
    kpis['rotacion_activos'] = ingresos / activo if activo != 0 else 0
    kpis['apalancamiento'] = activo / patrimonio if patrimonio != 0 else 0
    
    return kpis

def preparar_datos_tendencia(datos_historicos: dict) -> pd.DataFrame:
    """Convierte el diccionario de datos históricos en un DataFrame para graficar tendencias."""
    lista_periodos = [
        dict(periodo=periodo, **data['kpis']['Todos'])
        for periodo, data in datos_historicos.items()
        if 'Todos' in data.get('kpis', {})
    ]
    if not lista_periodos: return pd.DataFrame()
    df_tendencia = pd.DataFrame(lista_periodos)
    df_tendencia['periodo'] = pd.to_datetime(df_tendencia['periodo'], format='%Y-%m')
    df_tendencia = df_tendencia.sort_values(by='periodo').reset_index(drop=True)
    return df_tendencia

def calcular_analisis_vertical(df_estado_financiero: pd.DataFrame, valor_col: str, cuenta_col: str, base_cuenta: str):
    """Calcula el análisis vertical para un estado financiero."""
    if df_estado_financiero.empty or valor_col not in df_estado_financiero.columns:
        return df_estado_financiero

    df_analisis = df_estado_financiero.copy()
    df_analisis[valor_col] = pd.to_numeric(df_analisis[valor_col], errors='coerce').fillna(0)

    valor_base = abs(get_principal_account_value(df_analisis, base_cuenta, valor_col, cuenta_col))
    
    if valor_base == 0:
        df_analisis['Análisis Vertical (%)'] = 0.0
    else:
        df_analisis['Análisis Vertical (%)'] = (df_analisis[valor_col] / valor_base)
    
    return df_analisis

def construir_flujo_de_caja(df_er: pd.DataFrame, df_bg_actual: pd.DataFrame, df_bg_anterior: pd.DataFrame, val_col_er: str, cuenta_er: str, saldo_final_bg: str, cuenta_bg: str) -> pd.DataFrame:
    """Construye un estado de flujo de caja simplificado (Método Indirecto)."""
    
    utilidad_neta = -df_er[df_er[cuenta_er].str.startswith(('4','5','6','7'))][val_col_er].sum()
    depreciacion = abs(get_principal_account_value(df_er, '5160', val_col_er, cuenta_er))

    def get_variacion(cuenta, df_act, df_ant):
        val_act = get_principal_account_value(df_act, cuenta, saldo_final_bg, cuenta_bg)
        val_ant = get_principal_account_value(df_ant, cuenta, saldo_final_bg, cuenta_bg)
        return val_act - val_ant

    var_cuentas_cobrar = -get_variacion('13', df_bg_actual, df_bg_anterior)
    var_inventarios = -get_variacion('14', df_bg_actual, df_bg_anterior)
    var_proveedores = get_variacion('22', df_bg_actual, df_bg_anterior)
    fco = utilidad_neta + depreciacion + var_cuentas_cobrar + var_inventarios + var_proveedores

    var_activos_fijos = -get_variacion('15', df_bg_actual, df_bg_anterior)
    fci = var_activos_fijos

    var_obligaciones = get_variacion('21', df_bg_actual, df_bg_anterior)
    var_capital_social = get_variacion('31', df_bg_actual, df_bg_anterior)
    fcf = var_obligaciones + var_capital_social

    flujo_neto = fco + fci + fcf
    saldo_inicial_caja = get_principal_account_value(df_bg_anterior, '11', saldo_final_bg, cuenta_bg)
    saldo_final_caja = saldo_inicial_caja + flujo_neto

    data = {
        'Concepto': [
            'Utilidad Neta', ' (+) Depreciación y Amortización', 'Variación Cuentas por Cobrar', 'Variación Inventarios',
            'Variación Proveedores', '**Flujo de Efectivo de Operación (FCO)**',
            'Variación Activos Fijos (Inversión)', '**Flujo de Efectivo de Inversión (FCI)**',
            'Variación Obligaciones Financieras', 'Variación Capital Social', '**Flujo de Efectivo de Financiación (FCF)**',
            '**Flujo Neto de Efectivo del Periodo**', 'Saldo Inicial de Efectivo', '**Saldo Final de Efectivo**'
        ],
        'Valor': [
            utilidad_neta, depreciacion, var_cuentas_cobrar, var_inventarios, var_proveedores, fco,
            var_activos_fijos, fci, var_obligaciones, var_capital_social, fcf,
            flujo_neto, saldo_inicial_caja, saldo_final_caja
        ]
    }
    return pd.DataFrame(data)

# ==============================================================================
#           FUNCIONES DE VISUALIZACIÓN Y ANÁLISIS IA (Integradas)
# ==============================================================================

def plot_sparkline(data: pd.Series, title: str, is_percent: bool = False, lower_is_better: bool = False):
    if data.empty or len(data.dropna()) < 2:
        return go.Figure().update_layout(width=150, height=50, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', annotations=[dict(text="N/A", showarrow=False)])
    last_val, first_val = data.iloc[-1], data.iloc[0]
    color = '#28a745' if (lower_is_better and last_val < first_val) or (not lower_is_better and last_val > first_val) else '#dc3545'
    fig = go.Figure(go.Scatter(x=list(range(len(data))), y=data, mode='lines', line=dict(color=color, width=2.5), fill='tozeroy', fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)"))
    fig.update_layout(width=150, height=50, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=5, b=0), xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)
    return fig

def calcular_variaciones_er(df_actual: pd.DataFrame, df_previo: pd.DataFrame, cc_filter: str) -> pd.DataFrame:
    """VERSIÓN CORREGIDA Y ROBUSTA que maneja centros de costo que no existen en el periodo previo."""
    cuenta_col = COL_CONFIG['ESTADO_DE_RESULTADOS'].get('CUENTA', 'Cuenta')
    desc_col = COL_CONFIG['ESTADO_DE_RESULTADOS'].get('DESCRIPCION_CUENTA', 'Descripción')
    valor_col_nombre = 'Total_Consolidado_ER' if cc_filter == 'Todos' else cc_filter

    if valor_col_nombre not in df_actual.columns:
        st.error(f"Error Interno: La columna '{valor_col_nombre}' no se encontró en los datos del periodo actual.")
        return pd.DataFrame()
    df1 = df_actual[[cuenta_col, desc_col, valor_col_nombre]].copy()
    df1.rename(columns={valor_col_nombre: 'Valor_actual'}, inplace=True)

    if valor_col_nombre in df_previo.columns:
        df2 = df_previo[[cuenta_col, desc_col, valor_col_nombre]].copy()
        df2.rename(columns={valor_col_nombre: 'Valor_previo'}, inplace=True)
    else:
        st.warning(f"ADVERTENCIA: El centro de costo '{valor_col_nombre}' no se encontró en el periodo anterior. Se asumirán valores de cero para el comparativo.")
        df2 = df_previo[[cuenta_col, desc_col]].copy()
        df2['Valor_previo'] = 0

    df_variacion = pd.merge(df1, df2, on=[cuenta_col, desc_col], how='outer')
    df_variacion.fillna(0, inplace=True)
    df_variacion['Variacion_Absoluta'] = df_variacion['Valor_actual'] - df_variacion['Valor_previo']
    return df_variacion

def plot_waterfall_utilidad_neta(df_variacion: pd.DataFrame, periodo_actual: str, periodo_previo: str):
    """Crea un gráfico de cascada para explicar la variación de la Utilidad Neta."""
    cuenta_col = COL_CONFIG['ESTADO_DE_RESULTADOS'].get('CUENTA', 'Cuenta')
    utilidad_neta_actual, utilidad_neta_previa = df_variacion['Valor_actual'].sum(), df_variacion['Valor_previo'].sum()
    variacion_ingresos = df_variacion[df_variacion[cuenta_col].str.startswith('4')]['Variacion_Absoluta'].sum()
    variacion_costos = df_variacion[df_variacion[cuenta_col].str.startswith('6')]['Variacion_Absoluta'].sum()
    variacion_gastos = df_variacion[df_variacion[cuenta_col].str.startswith('5')]['Variacion_Absoluta'].sum()
    otras_variaciones = df_variacion['Variacion_Absoluta'].sum() - (variacion_ingresos + variacion_costos + variacion_gastos)
    fig = go.Figure(go.Waterfall(name="Variación", orientation="v", measure=["absolute", "relative", "relative", "relative", "relative", "total"], x=["Utilidad Neta " + periodo_previo, "Ingresos", "Costos", "Gastos Op.", "Otros", "Utilidad Neta " + periodo_actual], y=[utilidad_neta_previa, variacion_ingresos, variacion_costos, variacion_gastos, otras_variaciones, utilidad_neta_actual], connector={"line": {"color": "rgb(63, 63, 63)"}}, decreasing={"marker": {"color": "#28a745"}}, increasing={"marker": {"color": "#dc3545"}}))
    fig.update_layout(title=f"Puente de Utilidad Neta: {periodo_previo} vs {periodo_actual}", showlegend=False, yaxis_title="Monto (COP)", height=500, yaxis_tickformat="$,.0f")
    return fig

@st.cache_data(show_spinner=False)
def generar_analisis_ia(prompt: str):
    """Función genérica para llamar al modelo de IA."""
    try:
        api_key = st.secrets["google_ai"]["api_key"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text.replace('•', '*')
    except Exception as e:
        return f"🔴 **Error al contactar la IA:** {e}"

# ==============================================================================
#             CONFIGURACIÓN DE PÁGINA Y AUTENTICACIÓN
# ==============================================================================
st.set_page_config(layout="wide", page_title="Análisis Financiero Inteligente PRO")
st.title("🤖 Dashboard Financiero Profesional con IA")

# (El código de estilos y autenticación no cambia)
st.markdown("""<style>...</style>""", unsafe_allow_html=True) 
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
#             CARGA DE DATOS AUTOMÁTICA DESDE DROPBOX
# ==============================================================================
@st.cache_data(ttl=3600)
def cargar_y_procesar_datos():
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
                cc_cols_all = ['Todos'] + [name for name in COL_CONFIG['ESTADO_DE_RESULTADOS'].get('CENTROS_COSTO_COLS', {}).values() if name in df_er and name not in ['Total_Consolidado_ER', 'Sin centro de coste']]
                for cc in cc_cols_all:
                    datos_periodo['kpis'][cc] = calcular_kpis_periodo(df_er, df_bg, cc)
                datos_historicos[periodo] = datos_periodo
            except Exception as e:
                st.error(f"Error al procesar el archivo del periodo {periodo}: {e}")
    progress_bar.empty()
    return datos_historicos

if 'datos_historicos' not in st.session_state: st.session_state.datos_historicos = None
if st.sidebar.button("Refrescar Datos de Dropbox", use_container_width=True, type="primary"):
    st.cache_data.clear()
    st.session_state.datos_historicos = None
if st.session_state.datos_historicos is None:
    st.session_state.datos_historicos = cargar_y_procesar_datos()
if not st.session_state.datos_historicos:
    st.error("No se pudieron cargar datos. Verifica la conexión y la estructura de archivos en Dropbox.")
    st.stop()

# ==============================================================================
#                      INTERFAZ DE USUARIO PRINCIPAL
# ==============================================================================
st.sidebar.title("Opciones de Análisis")
sorted_periods = sorted(st.session_state.datos_historicos.keys(), reverse=True)
period_options = ["Análisis de Evolución (Tendencias)"] + sorted_periods
selected_view = st.sidebar.selectbox("Selecciona la vista de análisis:", period_options)

# ==============================================================================
#             VISTA DE ANÁLISIS DE TENDENCIAS
# ==============================================================================
if selected_view == "Análisis de Evolución (Tendencias)":
    st.header("📈 Informe de Evolución Gerencial")
    df_tendencia = preparar_datos_tendencia(st.session_state.datos_historicos)
    
    if df_tendencia.empty or len(df_tendencia) < 2:
        st.info("Se necesitan al menos dos periodos de datos para generar un análisis de evolución.")
        st.stop()
    
    # El resto del código de la vista de tendencias no cambia...
    # (KPIs, gráficos de evolución, etc.)

# ==============================================================================
#             VISTA DE PERIODO ÚNICO (CENTRO DE ANÁLISIS PROFUNDO)
# ==============================================================================
else:
    st.header(f"Centro de Análisis para el Periodo: {selected_view}")
    
    # --- PREPARACIÓN DE DATOS ---
    data_actual = st.session_state.datos_historicos[selected_view]
    periodo_actual_idx = sorted_periods.index(selected_view)
    periodo_previo = sorted_periods[periodo_actual_idx + 1] if periodo_actual_idx + 1 < len(sorted_periods) else None
    data_previa = st.session_state.datos_historicos.get(periodo_previo)
    df_er_actual, df_bg_actual = data_actual['df_er_master'], data_actual['df_bg_master']
    kpis_por_tienda = data_actual['kpis']

    # --- FILTROS EN SIDEBAR ---
    st.sidebar.subheader("Filtros del Periodo")
    cc_options_all = sorted(list(kpis_por_tienda.keys()))
    cc_filter = st.sidebar.selectbox("Filtrar por Centro de Costo:", cc_options_all, key=f"cc_{selected_view}")
    valor_col_filtrado = 'Total_Consolidado_ER' if cc_filter == 'Todos' else cc_filter
    
    # --- CÁLCULO DE VARIACIONES ---
    df_variacion_er = None
    if data_previa:
        df_variacion_er = calcular_variaciones_er(df_er_actual, data_previa['df_er_master'], cc_filter)
        if df_variacion_er is not None and not df_variacion_er.empty:
            st.info(f"Análisis comparativo contra el periodo **{periodo_previo}**.")
    else:
        st.warning("No hay un periodo anterior para realizar análisis comparativo.")

    # --- PESTAÑAS DE ANÁLISIS DETALLADO ---
    tabs = st.tabs([
        "📊 Resumen", "💰 A. Utilidad", "📈 A. Vertical", "📊 A. Horizontal", "🌊 Flujo de Caja", "📋 Reportes"
    ])

    with tabs[0]: # Resumen General
        selected_kpis = kpis_por_tienda.get(cc_filter, {})
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        kpi_col1.metric("Margen Neto", f"{selected_kpis.get('margen_neto', 0):.2%}")
        kpi_col2.metric("ROE", f"{selected_kpis.get('roe', 0):.2%}")
        kpi_col3.metric("Razón Corriente", f"{selected_kpis.get('razon_corriente', 0):.2f}")
        kpi_col4.metric("Endeudamiento", f"{selected_kpis.get('endeudamiento_activo', 0):.2%}")
        # Lógica de la IA...

    with tabs[1]: # Análisis de Utilidad
        if df_variacion_er is not None and not df_variacion_er.empty:
            st.plotly_chart(plot_waterfall_utilidad_neta(df_variacion_er, selected_view, periodo_previo), use_container_width=True)
        else:
            st.info("Se requiere un periodo anterior para el análisis de variación de utilidad.")

    with tabs[2]: # Análisis Vertical
        st.subheader("Análisis Vertical del Periodo")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Estado de Resultados")
            df_er_vert = calcular_analisis_vertical(df_er_actual, valor_col_filtrado, COL_CONFIG['ESTADO_DE_RESULTADOS']['CUENTA'], '4')
            st.dataframe(df_er_vert[['Descripción', valor_col_filtrado, 'Análisis Vertical (%)']].style.format({valor_col_filtrado: "${:,.0f}", 'Análisis Vertical (%)': '{:.2%}'}), use_container_width=True)
        with col2:
            st.markdown("#### Balance General")
            df_bg_vert = calcular_analisis_vertical(df_bg_actual, COL_CONFIG['BALANCE_GENERAL']['SALDO_FINAL'], COL_CONFIG['BALANCE_GENERAL']['CUENTA'], '1')
            st.dataframe(df_bg_vert[['Descripción', 'Saldo Final', 'Análisis Vertical (%)']].style.format({'Saldo Final': "${:,.0f}", 'Análisis Vertical (%)': '{:.2%}'}), use_container_width=True)

    with tabs[3]: # Análisis Horizontal
        st.subheader(f"Análisis Horizontal: {selected_view} vs. {periodo_previo}")
        if df_variacion_er is not None and not df_variacion_er.empty:
            st.dataframe(df_variacion_er[['Descripción', 'Valor_previo', 'Valor_actual', 'Variacion_Absoluta']].style.format('${:,.0f}'), use_container_width=True)
        else:
            st.info("Se requiere un periodo anterior para el análisis horizontal.")
    
    with tabs[4]: # Flujo de Caja
        st.subheader("Estado de Flujo de Caja (Método Indirecto)")
        if data_previa:
            df_flujo_caja = construir_flujo_de_caja(df_er_actual, df_bg_actual, data_previa['df_bg_master'], valor_col_filtrado, COL_CONFIG['ESTADO_DE_RESULTADOS']['CUENTA'], COL_CONFIG['BALANCE_GENERAL']['SALDO_FINAL'], COL_CONFIG['BALANCE_GENERAL']['CUENTA'])
            st.dataframe(df_flujo_caja.style.format({'Valor': "${:,.0f}"}), use_container_width=True)
        else:
            st.info("Se requiere un periodo anterior para construir el Flujo de Caja.")

    with tabs[5]: # Reportes
        st.subheader("Reportes Financieros Detallados")
        # Nivel de detalle para los reportes
        nivel_seleccionado = st.slider("Nivel de Detalle de Cuentas (ER):", 1, int(df_er_actual[COL_CONFIG['ESTADO_DE_RESULTADOS']['NIVEL_LINEA']].max()), 1)
        
        st.markdown("#### Estado de Resultados")
        df_er_display = generate_financial_statement(df_er_actual, 'Estado de Resultados', cc_filter, nivel_seleccionado)
        st.dataframe(df_er_display.style.format({'Valor': "${:,.0f}"}), use_container_width=True, height=600)

        st.markdown("#### Balance General")
        df_bg_display = generate_financial_statement(df_bg_actual, 'Balance General', 99)
        st.dataframe(df_bg_display.style.format({'Valor': "${:,.0f}"}), use_container_width=True, height=600)
