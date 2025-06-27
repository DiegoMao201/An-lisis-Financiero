# Dashboard_Principal.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Importamos nuestros módulos ---
from mi_logica_original import procesar_archivo_excel, generate_financial_statement, to_excel_buffer, COL_CONFIG
from dropbox_connector import get_dropbox_client, find_financial_files, load_excel_from_dropbox
from kpis_y_analisis import calcular_kpis_periodo, preparar_datos_tendencia, generar_analisis_avanzado_ia

# ==============================================================================
#                 CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ==============================================================================
st.set_page_config(layout="wide", page_title="Análisis Financiero Inteligente")
st.title("🤖 Análisis Financiero Inteligente por IA")

# --- ESTILOS VISUALES CON LA CORRECCIÓN PARA TEXTO LARGO ---
st.markdown("""
<style>
.ai-analysis-text {
    /* ESTA ES LA REGLA CLAVE: Fuerza al texto a romperse si no cabe */
    overflow-wrap: break-word;
    word-wrap: break-word;
}
.main .block-container {
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
#                 AUTENTICACIÓN
# ==============================================================================
try:
    real_password = st.secrets["general"]["password"]
except Exception:
    st.error("No se encontró la contraseña en los secretos. Contacta al administrador.")
    st.stop()

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    password = st.text_input("Introduce la contraseña para acceder:", type="password")
    if password == real_password:
        st.session_state.authenticated = True
        st.rerun()
    else:
        if password:
            st.warning("Contraseña incorrecta.")
        st.stop()

# ==============================================================================
#               CARGA DE DATOS AUTOMÁTICA DESDE DROPBOX
# ==============================================================================
@st.cache_data(ttl=3600)
def cargar_y_procesar_datos():
    """Función maestra que conecta, descarga y procesa todos los archivos de Dropbox."""
    dbx = get_dropbox_client()
    if not dbx: return None
    
    archivos_financieros = find_financial_files(dbx, base_folder="/data")
    if not archivos_financieros:
        st.warning("No se encontraron archivos de Excel en la carpeta /data de Dropbox.")
        return None

    datos_historicos = {}
    progress_bar = st.progress(0, text="Iniciando carga desde Dropbox...")
    
    for i, file_info in enumerate(archivos_financieros):
        periodo = file_info["periodo"]
        path = file_info["path"]
        progress_bar.progress((i + 1) / len(archivos_financieros), text=f"Procesando {periodo}...")
        
        excel_bytes = load_excel_from_dropbox(dbx, path)
        if excel_bytes:
            try:
                df_er, df_bg = procesar_archivo_excel(excel_bytes)
                datos_periodo = {'df_er_master': df_er, 'df_bg_master': df_bg, 'kpis': {}}
                
                # Calcular KPIs para 'Todos' (consolidado)
                kpis_totales = calcular_kpis_periodo(df_er, df_bg, 'Todos')
                datos_periodo['kpis']['Todos'] = kpis_totales
                
                # Calcular KPIs para cada centro de costo individual
                er_conf = COL_CONFIG['ESTADO_DE_RESULTADOS']
                cc_cols = [name for name in er_conf.get('CENTROS_COSTO_COLS', {}).values() if name in df_er and name not in ['Total_Consolidado_ER', 'Sin centro de coste']]
                for cc in cc_cols:
                    datos_periodo['kpis'][cc] = calcular_kpis_periodo(df_er, df_bg, cc)
                
                datos_historicos[periodo] = datos_periodo
            except Exception as e:
                st.error(f"Error al procesar el archivo del periodo {periodo}: {e}")
    
    progress_bar.empty()
    return datos_historicos

# --- Lógica principal de ejecución de carga de datos ---
if 'datos_historicos' not in st.session_state:
    st.session_state.datos_historicos = None

if st.sidebar.button("Refrescar Datos de Dropbox", use_container_width=True):
    st.cache_data.clear()
    st.session_state.datos_historicos = None

if st.session_state.datos_historicos is None:
    st.session_state.datos_historicos = cargar_y_procesar_datos()

if not st.session_state.datos_historicos:
    st.error("No se pudieron cargar datos. Verifica la conexión y la estructura de archivos en Dropbox.")
    st.stop()

# ==============================================================================
#                     INTERFAZ DE USUARIO PRINCIPAL
# ==============================================================================
st.sidebar.title("Opciones de Análisis")
sorted_periods = sorted(st.session_state.datos_historicos.keys(), reverse=True)
period_options = ["Análisis General (Tendencias)"] + sorted_periods
selected_view = st.sidebar.selectbox("Selecciona el periodo de análisis:", period_options)

# ==============================================================================
#                  VISTA DE ANÁLISIS DE TENDENCIAS
# ==============================================================================
if selected_view == "Análisis General (Tendencias)":
    st.header("📈 Análisis de Tendencias Financieras")
    df_tendencia = preparar_datos_tendencia(st.session_state.datos_historicos)
    
    if df_tendencia.empty or len(df_tendencia) < 2:
        st.info("Se necesitan al menos dos periodos para mostrar tendencias.")
        st.stop()
    
    latest_kpis = df_tendencia.iloc[-1]
    previous_kpis = df_tendencia.iloc[-2]
    
    st.subheader("Indicadores Clave del Último Periodo (Consolidado)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Utilidad Neta", f"${latest_kpis['utilidad_neta']:,.0f}", f"{latest_kpis['utilidad_neta'] - previous_kpis['utilidad_neta']:,.0f}")
    col2.metric("Margen Neto", f"{latest_kpis['margen_neto']:.2%}", f"{latest_kpis['margen_neto'] - previous_kpis['margen_neto']:.2%}")
    col3.metric("Razón Corriente", f"{latest_kpis['razon_corriente']:.2f}", f"{latest_kpis['razon_corriente'] - previous_kpis['razon_corriente']:.2f}")
    col4.metric("ROE", f"{latest_kpis['roe']:.2%}", f"{latest_kpis['roe'] - previous_kpis['roe']:.2%}")

    st.markdown("---")
    st.subheader("Evolución Financiera")
    
    fig_utilidades = go.Figure()
    fig_utilidades.add_trace(go.Bar(x=df_tendencia['periodo'], y=df_tendencia['ingresos'], name='Ingresos', marker_color='#1f77b4'))
    fig_utilidades.add_trace(go.Scatter(x=df_tendencia['periodo'], y=df_tendencia['utilidad_operacional'], name='Utilidad Operacional', mode='lines+markers', line=dict(color='#ff7f0e', width=3)))
    fig_utilidades.add_trace(go.Scatter(x=df_tendencia['periodo'], y=df_tendencia['utilidad_neta'], name='Utilidad Neta', mode='lines+markers', line=dict(color='#2ca02c', width=3)))
    fig_utilidades.update_layout(title='Evolución de Ingresos y Utilidades', legend_title_text='')
    st.plotly_chart(fig_utilidades, use_container_width=True)

# ==============================================================================
#       VISTA DE PERIODO ÚNICO (CON LAYOUT LIMPIO Y TODAS LAS FUNCIONALIDADES)
# ==============================================================================
else:
    st.header(f"Análisis Financiero para el Periodo: {selected_view}")
    
    data_actual = st.session_state.datos_historicos.get(selected_view)
    if not data_actual:
        st.error(f"No se encontraron datos para el periodo: {selected_view}"); st.stop()

    df_er_actual = data_actual['df_er_master']
    df_bg_actual = data_actual['df_bg_master']
    kpis_por_tienda = data_actual['kpis']

    # --- FILTROS COMPLETOS EN LA BARRA LATERAL ---
    st.sidebar.subheader("Filtros del Periodo")
    cc_options_all = sorted(list(kpis_por_tienda.keys()))
    cc_filter = st.sidebar.selectbox("Filtrar por Centro de Costo:", cc_options_all, key=f"cc_{selected_view}")
    
    er_conf = COL_CONFIG['ESTADO_DE_RESULTADOS']
    nivel_col = er_conf.get('NIVEL_LINEA', 'Grupo')
    if nivel_col in df_er_actual.columns:
        max_nivel = int(df_er_actual[nivel_col].max())
        nivel_seleccionado = st.sidebar.slider("Nivel de Detalle de Cuentas:", 1, max_nivel, 1, key=f"nivel_er_{selected_view}")
    else:
        nivel_seleccionado = 1

    st.sidebar.subheader("Buscador de Cuentas")
    search_account_input = st.sidebar.text_input("Buscar por número de cuenta:", key=f"search_{selected_view}", placeholder="Ej: 510506")
    
    # --- OBTENCIÓN DE DATOS PARA LA VISTA ACTUAL ---
    selected_kpis = kpis_por_tienda.get(cc_filter, {})

    # --- KPIs PRINCIPALES ---
    st.subheader(f"🔍 KPIs para: {cc_filter}")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    kpi_col1.metric("Margen Neto", f"{selected_kpis.get('margen_neto', 0):.2%}")
    kpi_col2.metric("ROE", f"{selected_kpis.get('roe', 0):.2%}")
    kpi_col3.metric("Razón Corriente", f"{selected_kpis.get('razon_corriente', 0):.2f}")
    kpi_col4.metric("Endeudamiento", f"{selected_kpis.get('endeudamiento_activo', 0):.2%}")

    st.markdown("---")

    # --- ANÁLISIS DE IA DENTRO DE UN EXPANDER ---
    with st.expander("🧠 Ver Análisis y Consejos del CFO Virtual (IA)", expanded=True):
        with st.spinner('El CFO Virtual está analizando los datos...'):
            analisis_ia = generar_analisis_avanzado_ia(selected_kpis, df_er_actual, cc_filter, selected_view)
        
        st.markdown(f"<div class='ai-analysis-text'>{analisis_ia}</div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # --- REPORTES DETALLADOS EN PESTAÑAS ---
    st.subheader("📊 Reportes Financieros Detallados")
    tab1, tab2 = st.tabs(["Estado de Resultados", "Balance General"])

    with tab1:
        df_er_display = generate_financial_statement(df_er_actual, 'Estado de Resultados', cc_filter, nivel_seleccionado)
        st.dataframe(df_er_display.style.format({'Valor': "${:,.0f}"}), use_container_width=True, hide_index=True)

    with tab2:
        df_bg_display = generate_financial_statement(df_bg_actual, 'Balance General', 99)
        st.dataframe(df_bg_display.style.format({'Valor': "${:,.0f}"}), use_container_width=True, hide_index=True)

    # --- RESULTADOS DEL BUSCADOR DE CUENTAS ---
    if search_account_input:
        st.markdown("---")
        with st.expander(f"Resultado de la búsqueda para cuentas que inician con '{search_account_input}'", expanded=True):
            cuenta_col_er = er_conf.get('CUENTA', 'Cuenta')
            cuenta_col_bg = COL_CONFIG['BALANCE_GENERAL'].get('CUENTA', 'Cuenta')

            st.write("**Estado de Resultados**")
            df_search_er = df_er_actual[df_er_actual[cuenta_col_er].astype(str).str.startswith(search_account_input)]
            if not df_search_er.empty:
                st.dataframe(df_search_er)
            else:
                st.info(f"No se encontraron cuentas en el Estado de Resultados para '{search_account_input}'.")

            st.write("**Balance General**")
            df_search_bg = df_bg_actual[df_bg_actual[cuenta_col_bg].astype(str).str.startswith(search_account_input)]
            if not df_search_bg.empty:
                st.dataframe(df_search_bg)
            else:
                st.info(f"No se encontraron cuentas en el Balance General para '{search_account_input}'.")

    # --- BOTÓN DE DESCARGA ---
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
