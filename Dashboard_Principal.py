# Dashboard_Principal.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Importamos nuestros módulos ---
from mi_logica_original import procesar_archivo_excel, generate_financial_statement, to_excel_buffer, COL_CONFIG
from dropbox_connector import get_dropbox_client, find_financial_files, load_excel_from_dropbox
from kpis_y_analisis import calcular_kpis_periodo, preparar_datos_tendencia, generar_comentario_kpi

# ==============================================================================
#                 CONFIGURACIÓN Y AUTENTICACIÓN
# ==============================================================================
st.set_page_config(layout="wide", page_title="Análisis Financiero Inteligente")
st.title("📊 Análisis Financiero Inteligente y Tablero Gerencial")

# --- ESTILOS VISUALES MEJORADOS ---
st.markdown("""
    <style>
    /* Clases de métricas de Streamlit */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 600;
    }
    [data-testid="stMetricLabel"] {
        font-size: 16px;
    }
    [data-testid="stMetricDelta"] {
        font-size: 14px;
    }
    /* Contenedor principal para un layout más limpio */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    /* Estilo para los dataframes */
    .stDataFrame {
        border: 1px solid #e1e4e8;
        border-radius: 8px;
    }
    /* Títulos de las secciones */
    h2, h3 {
        border-bottom: 2px solid #f0f2f6;
        padding-bottom: 8px;
    }
    </style>
""", unsafe_allow_html=True)


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
        st.rerun() # Recarga la página para mostrar el contenido
    else:
        if password: # Si el usuario ha escrito algo y es incorrecto
            st.warning("Contraseña incorrecta. Inténtalo de nuevo.")
        st.stop()

# Si ya está autenticado, la app continúa.

# ==============================================================================
#               CARGA DE DATOS AUTOMÁTICA DESDE DROPBOX
# ==============================================================================
@st.cache_data(ttl=3600) # Cache de 1 hora
def cargar_y_procesar_datos():
    """Función maestra que conecta, descarga y procesa todos los archivos de Dropbox."""
    dbx = get_dropbox_client()
    if not dbx: return None
    
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
                
                # Preparamos el diccionario para el periodo actual
                datos_historicos_periodo = {
                    'df_er_master': df_er,
                    'df_bg_master': df_bg,
                    'kpis': {} 
                }
                
                # Calcular KPIs consolidados ('Total' o 'Todos')
                kpis_totales = calcular_kpis_periodo(df_er, df_bg, 'Todos')
                datos_historicos_periodo['kpis']['Todos'] = kpis_totales
                
                # Calcular KPIs por cada centro de costo
                er_conf = COL_CONFIG['ESTADO_DE_RESULTADOS']
                cc_cols = [name for name in er_conf.get('CENTROS_COSTO_COLS', {}).values() if name in df_er and name not in ['Total_Consolidado_ER', 'Sin centro de coste']]
                for cc in cc_cols:
                    datos_historicos_periodo['kpis'][cc] = calcular_kpis_periodo(df_er, df_bg, cc)
                
                # Guardamos los datos procesados para el periodo
                datos_historicos[periodo] = datos_historicos_periodo

            except Exception as e:
                st.error(f"Error al procesar el archivo del periodo {periodo}: {e}")
    
    progress_bar.empty()
    return datos_historicos

# --- Lógica principal de ejecución ---
if 'datos_historicos' not in st.session_state:
    st.session_state.datos_historicos = None

if st.sidebar.button("Refrescar Datos de Dropbox", use_container_width=True):
    st.cache_data.clear()
    st.session_state.datos_historicos = None

if st.session_state.datos_historicos is None:
    st.session_state.datos_historicos = cargar_y_procesar_datos()

if not st.session_state.datos_historicos:
    st.error("No se pudieron cargar datos. Revisa la conexión a Dropbox y la estructura de archivos.")
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
    st.header("📈 Análisis de Tendencias Financieras")
    
    # Preparamos un diccionario donde la clave 'Total' contiene los KPIs consolidados de cada mes
    datos_tendencia_totales = {
        periodo: data['kpis']['Todos'] for periodo, data in st.session_state.datos_historicos.items()
    }
    df_tendencia = preparar_datos_tendencia(datos_tendencia_totales)

    if df_tendencia.empty or len(df_tendencia) < 1:
        st.warning("No hay suficientes datos para mostrar tendencias. Se necesita al menos un periodo.")
        st.stop()

    latest_kpis = df_tendencia.iloc[-1]
    previous_kpis = df_tendencia.iloc[-2] if len(df_tendencia) > 1 else latest_kpis
    
    st.subheader("Indicadores Clave del Último Periodo (Consolidado)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Utilidad Neta", f"${latest_kpis['utilidad_neta']:,.0f}", f"{latest_kpis['utilidad_neta'] - previous_kpis['utilidad_neta']:,.0f}", help="Ganancia final después de todos los gastos e impuestos.")
    col2.metric("Margen Neto", f"{latest_kpis['margen_neto']:.2%}", f"{latest_kpis['margen_neto'] - previous_kpis['margen_neto']:.2%}", help="Porcentaje de los ingresos que se convierte en ganancia neta.")
    col3.metric("Razón Corriente", f"{latest_kpis['razon_corriente']:.2f}", f"{latest_kpis['razon_corriente'] - previous_kpis['razon_corriente']:.2f}", help="Capacidad de cubrir deudas a corto plazo (Activo Corriente / Pasivo Corriente).")
    col4.metric("ROE", f"{latest_kpis['roe']:.2%}", f"{latest_kpis['roe'] - previous_kpis['roe']:.2%}", help="Rentabilidad generada sobre el capital de los accionistas.")

    st.markdown("---")
    st.subheader("Evolución Financiera")
    
    col_graf1, col_graf2 = st.columns(2)
    with col_graf1:
        fig_rentabilidad = px.line(df_tendencia, x='periodo', y=['margen_neto', 'margen_operacional', 'roe'],
                                   title='Tendencia de Rentabilidad', labels={'value': 'Porcentaje', 'variable': 'Indicador'})
        fig_rentabilidad.update_layout(yaxis_tickformat='.2%', legend_title_text='')
        st.plotly_chart(fig_rentabilidad, use_container_width=True)

    with col_graf2:
        fig_solvencia = px.line(df_tendencia, x='periodo', y=['razon_corriente', 'endeudamiento_activo'],
                                title='Tendencia de Liquidez y Endeudamiento', labels={'value': 'Ratio', 'variable': 'Indicador'})
        fig_solvencia.update_layout(legend_title_text='')
        st.plotly_chart(fig_solvencia, use_container_width=True)
    
    fig_utilidades = go.Figure()
    fig_utilidades.add_trace(go.Bar(x=df_tendencia['periodo'], y=df_tendencia['ingresos'], name='Ingresos', marker_color='#1f77b4'))
    fig_utilidades.add_trace(go.Scatter(x=df_tendencia['periodo'], y=df_tendencia['utilidad_operacional'], name='Utilidad Operacional', mode='lines+markers', line=dict(color='#ff7f0e', width=3)))
    fig_utilidades.add_trace(go.Scatter(x=df_tendencia['periodo'], y=df_tendencia['utilidad_neta'], name='Utilidad Neta', mode='lines+markers', line=dict(color='#2ca02c', width=3)))
    fig_utilidades.update_layout(title='Evolución de Ingresos y Utilidades', legend_title_text='')
    st.plotly_chart(fig_utilidades, use_container_width=True)


# ==============================================================================
#                  VISTA DE ANÁLISIS DE PERIODO ÚNICO
# ==============================================================================
else:
    st.header(f"Detalle Financiero para el Periodo: {selected_view}")
    
    data_actual = st.session_state.datos_historicos.get(selected_view)
    if not data_actual:
        st.error(f"No se encontraron datos para el periodo: {selected_view}")
        st.stop()
        
    df_er_actual = data_actual['df_er_master']
    df_bg_actual = data_actual['df_bg_master']
    kpis_por_tienda = data_actual['kpis']
    
    st.sidebar.subheader("Filtros del Periodo")
    
    er_conf_sidebar = COL_CONFIG['ESTADO_DE_RESULTADOS']
    cc_options_all = sorted(list(kpis_por_tienda.keys()))
    
    cc_filter = st.sidebar.selectbox("Filtrar por Centro de Costo:", cc_options_all, key=f"cc_{selected_view}")
    
    selected_kpis = kpis_por_tienda.get(cc_filter, {})

    st.subheader(f"🔍 KPIs para: {cc_filter}")
    
    # --- KPIs DINÁMICOS CON COMENTARIOS ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Margen Neto", f"{selected_kpis.get('margen_neto', 0):.2%}", help=generar_comentario_kpi('margen_neto', selected_kpis.get('margen_neto')))
    col2.metric("ROE", f"{selected_kpis.get('roe', 0):.2%}", help=generar_comentario_kpi('roe', selected_kpis.get('roe')))
    col3.metric("Razón Corriente", f"{selected_kpis.get('razon_corriente', 0):.2f}", help=generar_comentario_kpi('razon_corriente', selected_kpis.get('razon_corriente')))
    col4.metric("Endeudamiento", f"{selected_kpis.get('endeudamiento_activo', 0):.2%}", help=generar_comentario_kpi('endeudamiento_activo', selected_kpis.get('endeudamiento_activo')))

    st.markdown("---")

    report_type = st.radio("Selecciona el reporte a visualizar:", ["Estado de Resultados", "Balance General"], key=f"radio_{selected_view}", horizontal=True)

    if report_type == "Estado de Resultados":
        st.subheader(f"Estado de Resultados ({cc_filter})")
        
        max_lvl_er_val = 5
        er_niv_col = er_conf_sidebar.get('NIVEL_LINEA', 'Grupo')
        if er_niv_col in df_er_actual.columns:
            min_l, max_l = int(df_er_actual[er_niv_col].min()), int(df_er_actual[er_niv_col].max())
            max_lvl_er_val = st.sidebar.slider("Nivel de Detalle (ER):", min_l, max_l, min_l, key=f"lvl_er_{selected_view}")

        df_er_display = generate_financial_statement(df_er_actual, 'Estado de Resultados', cc_filter, max_lvl_er_val)
        if not df_er_display.empty:
            df_er_display_fmt = df_er_display.copy()
            if 'Valor' in df_er_display_fmt.columns:
                df_er_display_fmt['Valor'] = pd.to_numeric(df_er_display_fmt['Valor'], errors='coerce').apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
            st.dataframe(df_er_display_fmt, use_container_width=True, hide_index=True)

    if report_type == "Balance General":
        st.subheader("Balance General")
        
        # El balance no se filtra por tienda, siempre se muestra el consolidado.
        df_bg_display = generate_financial_statement(df_bg_actual, 'Balance General', max_level=99)
        if not df_bg_display.empty:
            df_bg_display_fmt = df_bg_display.copy()
            if 'Valor' in df_bg_display_fmt.columns:
                 df_bg_display_fmt['Valor'] = pd.to_numeric(df_bg_display_fmt['Valor'], errors='coerce').apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "")
            st.dataframe(df_bg_display_fmt, use_container_width=True, hide_index=True)
            
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
