# Dashboard_Principal.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Importamos nuestros módulos ---
from mi_logica_original import procesar_archivo_excel, generate_financial_statement, to_excel_buffer
from dropbox_connector import get_dropbox_client, find_financial_files, load_excel_from_dropbox
from kpis_y_analisis import calcular_kpis_periodo, preparar_datos_tendencia

# ==============================================================================
#                 CONFIGURACIÓN Y AUTENTICACIÓN
# ==============================================================================
st.set_page_config(layout="wide", page_title="Análisis Financiero Inteligente")
st.title("🤖 Análisis Financiero Inteligente y Tablero Gerencial")

try:
    real_password = st.secrets["general"]["password"]
except Exception:
    st.error("No se encontró la contraseña en los secretos. Contacta al administrador.")
    st.stop()

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

password = st.text_input("Introduce la contraseña para acceder:", type="password")
if password == real_password:
    st.session_state.authenticated = True

if not st.session_state.authenticated:
    st.warning("Debes ingresar la contraseña correcta para acceder al tablero.")
    st.stop()

# ==============================================================================
#               CARGA DE DATOS AUTOMÁTICA DESDE DROPBOX
# ==============================================================================
@st.cache_data(ttl=3600) # Cache de 1 hora
def cargar_y_procesar_datos():
    """Función maestra que conecta, descarga y procesa todos los archivos de Dropbox."""
    dbx = get_dropbox_client()
    if not dbx:
        return None
    
    archivos_financieros = find_financial_files(dbx, base_folder="/data")
    if not archivos_financieros:
        st.warning("No se encontraron archivos en la carpeta /data de Dropbox.")
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
                kpis = calcular_kpis_periodo(df_er, df_bg)
                datos_historicos[periodo] = {
                    'df_er_master': df_er,
                    'df_bg_master': df_bg,
                    'kpis': kpis
                }
            except Exception as e:
                st.error(f"Error al procesar el archivo del periodo {periodo}: {e}")
    
    progress_bar.empty()
    return datos_historicos

# --- Lógica principal de ejecución ---
if 'datos_historicos' not in st.session_state:
    st.session_state.datos_historicos = None

if st.sidebar.button("Refrescar Datos de Dropbox"):
    st.cache_data.clear()
    st.session_state.datos_historicos = None

if st.session_state.datos_historicos is None:
    st.session_state.datos_historicos = cargar_y_procesar_datos()

if not st.session_state.datos_historicos:
    st.error("No se pudieron cargar datos. Revisa la conexión a Dropbox y la estructura de carpetas.")
    st.stop()

# ==============================================================================
#                     INTERFAZ DE USUARIO PRINCIPAL
# ==============================================================================
st.sidebar.title("Opciones de Análisis")
sorted_periods = sorted(st.session_state.datos_historicos.keys(), reverse=True)
period_options = ["Análisis General (Tendencias)"] + sorted_periods
selected_view = st.sidebar.selectbox("Selecciona la vista de análisis:", period_options)

# ==============================================================================
#                  VISTA DE ANÁLISIS DE TENDENCIAS
# ==============================================================================
if selected_view == "Análisis General (Tendencias)":
    st.header("🚀 Análisis de Tendencias Financieras")
    df_tendencia = preparar_datos_tendencia(st.session_state.datos_historicos)

    if df_tendencia.empty:
        st.warning("No hay suficientes datos para mostrar tendencias.")
        st.stop()

    latest_kpis = df_tendencia.iloc[-1]
    previous_kpis = df_tendencia.iloc[-2] if len(df_tendencia) > 1 else latest_kpis
    
    # --- KPIs Principales con Deltas ---
    st.subheader("Indicadores Clave del Último Periodo")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Utilidad Neta", f"${latest_kpis['utilidad_neta']:,.0f}", f"{latest_kpis['utilidad_neta'] - previous_kpis['utilidad_neta']:,.0f}")
    col2.metric("Margen Neto", f"{latest_kpis['margen_neto']:.2%}", f"{latest_kpis['margen_neto'] - previous_kpis['margen_neto']:.2%}")
    col3.metric("Razón Corriente", f"{latest_kpis['razon_corriente']:.2f}", f"{latest_kpis['razon_corriente'] - previous_kpis['razon_corriente']:.2f}")
    col4.metric("ROE", f"{latest_kpis['roe']:.2%}", f"{latest_kpis['roe'] - previous_kpis['roe']:.2%}")

    # --- Gráficos de Tendencia ---
    st.markdown("---")
    st.subheader("Evolución Financiera")
    
    # Gráfico 1: Rentabilidad
    fig_rentabilidad = px.line(df_tendencia, x='periodo', y=['margen_neto', 'margen_operacional', 'roe'],
                               title='Tendencia de Márgenes de Rentabilidad y ROE', labels={'value': 'Porcentaje', 'variable': 'Indicador'})
    fig_rentabilidad.update_layout(yaxis_tickformat='.2%')
    st.plotly_chart(fig_rentabilidad, use_container_width=True)

    # Gráfico 2: Ingresos vs Utilidades
    fig_utilidades = go.Figure()
    fig_utilidades.add_trace(go.Bar(x=df_tendencia['periodo'], y=df_tendencia['ingresos'], name='Ingresos'))
    fig_utilidades.add_trace(go.Scatter(x=df_tendencia['periodo'], y=df_tendencia['utilidad_operacional'], name='Utilidad Operacional', mode='lines+markers'))
    fig_utilidades.add_trace(go.Scatter(x=df_tendencia['periodo'], y=df_tendencia['utilidad_neta'], name='Utilidad Neta', mode='lines+markers'))
    fig_utilidades.update_layout(title='Evolución de Ingresos y Utilidades')
    st.plotly_chart(fig_utilidades, use_container_width=True)

    # Gráfico 3: Liquidez y Endeudamiento
    fig_solvencia = px.line(df_tendencia, x='periodo', y=['razon_corriente', 'endeudamiento_activo', 'apalancamiento'],
                            title='Tendencia de Liquidez y Endeudamiento', labels={'value': 'Ratio', 'variable': 'Indicador'})
    st.plotly_chart(fig_solvencia, use_container_width=True)


# ==============================================================================
#                  VISTA DE ANÁLISIS DE PERIODO ÚNICO
# ==============================================================================
else:
    st.header(f"📄 Análisis Detallado para el Periodo: {selected_view}")
    
    # Obtenemos los dataframes para el mes seleccionado
    data_actual = st.session_state.datos_historicos[selected_view]
    df_er_actual = data_actual['df_er_master']
    df_bg_actual = data_actual['df_bg_master']
    kpis_actuales = data_actual['kpis']
    
    # --- Interfaz de la Sidebar para este periodo ---
    report_type = st.sidebar.radio("Selecciona el reporte:", ["Estado de Resultados", "Balance General"], key=f"radio_{selected_view}")
    
    # --- REUTILIZAMOS TU LÓGICA DE UI ORIGINAL ---
    if report_type == "Estado de Resultados":
        st.subheader("📈 Estado de Resultados")
        
        # Filtro de Centro de Costo (adaptado de tu código)
        from mi_logica_original import COL_CONFIG
        er_conf_sidebar = COL_CONFIG['ESTADO_DE_RESULTADOS']
        cc_options_ui_list = [name for name in er_conf_sidebar.get('CENTROS_COSTO_COLS', {}).values() if name in df_er_actual.columns and name not in [er_conf_sidebar.get('CENTROS_COSTO_COLS', {}).get('Total'), er_conf_sidebar.get('CENTROS_COSTO_COLS', {}).get('Sin centro de coste')]]
        cc_options_ui_list = sorted(list(set(cc_options_ui_list)))
        active_cc = "Todos"
        if cc_options_ui_list:
            active_cc = st.sidebar.selectbox("Filtrar por Centro de Costo:", ['Todos'] + cc_options_ui_list, key=f"cc_{selected_view}")
        
        # Slider de Nivel de Detalle (adaptado de tu código)
        max_lvl_er_val = 5
        er_niv_col_slider_disp = er_conf_sidebar.get('NIVEL_LINEA', 'Grupo')
        if er_niv_col_slider_disp in df_er_actual.columns:
            min_l_er_val = int(df_er_actual[er_niv_col_slider_disp].min())
            max_l_er_val = int(df_er_actual[er_niv_col_slider_disp].max())
            max_lvl_er_val = st.sidebar.slider("Nivel Detalle (ER):", min_l_er_val, max_l_er_val, min_l_er_val, key=f"lvl_er_{selected_view}")

        # Mostrar KPIs del periodo
        col1, col2, col3 = st.columns(3)
        col1.metric("Utilidad Operacional", f"${kpis_actuales.get('utilidad_operacional', 0):,.0f}", f"{kpis_actuales.get('margen_operacional', 0):.1%} Margen")
        col2.metric("Utilidad Neta", f"${kpis_actuales.get('utilidad_neta', 0):,.0f}", f"{kpis_actuales.get('margen_neto', 0):.1%} Margen")
        col3.metric("Ingresos Totales", f"${kpis_actuales.get('ingresos', 0):,.0f}")
        
        # Generar y mostrar la tabla usando TU función
        df_er_display = generate_financial_statement(df_er_actual, 'Estado de Resultados', active_cc, max_lvl_er_val)
        if not df_er_display.empty:
            df_er_display_fmt = df_er_display.copy()
            if 'Valor' in df_er_display_fmt.columns:
                df_er_display_fmt['Valor'] = pd.to_numeric(df_er_display_fmt['Valor'], errors='coerce').apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
            st.dataframe(df_er_display_fmt, use_container_width=True, hide_index=True)

    if report_type == "Balance General":
        st.subheader("⚖️ Balance General")
        
        # Slider de Nivel
        max_lvl_bg_val = 5
        # (Aquí podrías agregar lógica para el slider de BG si lo deseas)

        # Mostrar KPIs del periodo
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Razón Corriente", f"{kpis_actuales.get('razon_corriente', 0):.2f}")
        col2.metric("Prueba Ácida", f"{kpis_actuales.get('prueba_acida', 0):.2f}")
        col3.metric("Endeudamiento Activo", f"{kpis_actuales.get('endeudamiento_activo', 0):.2%}")
        col4.metric("ROE (Rent. Patrimonio)", f"{kpis_actuales.get('roe', 0):.2%}")
        
        # Generar y mostrar la tabla usando TU función
        df_bg_display = generate_financial_statement(df_bg_actual, 'Balance General', max_level=max_lvl_bg_val)
        if not df_bg_display.empty:
            st.dataframe(df_bg_display.style.format({'Valor': "${:,.0f}"}), use_container_width=True)
            
    # --- Botón de Descarga ---
    st.sidebar.markdown("---")
    er_to_dl = generate_financial_statement(df_er_actual, 'Estado de Resultados', 'Todos', 99)
    bg_to_dl = generate_financial_statement(df_bg_actual, 'Balance General', 'Todos', 99)
    excel_buffer = to_excel_buffer(er_to_dl, bg_to_dl)
    st.sidebar.download_button(
        label=f"Descargar Reportes ({selected_view})",
        data=excel_buffer,
        file_name=f"Reporte_Financiero_{selected_view}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
