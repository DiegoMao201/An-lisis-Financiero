# Dashboard_Principal.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Importamos nuestros módulos ---
from mi_logica_original import procesar_archivo_excel, generate_financial_statement, to_excel_buffer, COL_CONFIG
from dropbox_connector import get_dropbox_client, find_financial_files, load_excel_from_dropbox
# --- LÍNEA CORREGIDA ---
from kpis_y_analisis import calcular_kpis_periodo, preparar_datos_tendencia, generar_analisis_avanzado_ia, generar_lista_cuentas # <-- CORREGIDO A PLURAL

# ==============================================================================
#                 CONFIGURACIÓN Y AUTENTICACIÓN
# ==============================================================================
st.set_page_config(layout="wide", page_title="Análisis Financiero Inteligente")
st.title("🤖 Análisis Financiero Inteligente por IA")

st.markdown("""<style> /* ... (Estilos CSS omitidos por brevedad) ... */ </style>""", unsafe_allow_html=True) # Mantén tus estilos

try:
    real_password = st.secrets["general"]["password"]
except Exception:
    st.error("No se encontró la contraseña."); st.stop()

if 'authenticated' not in st.session_state: st.session_state.authenticated = False

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
    dbx = get_dropbox_client()
    if not dbx: return None
    archivos_financieros = find_financial_files(dbx, base_folder="/data")
    if not archivos_financieros:
        st.warning("No se encontraron archivos en la carpeta /data de Dropbox.")
        return None

    datos_historicos = {}
    progress_bar = st.progress(0, text="Cargando y procesando datos...")
    
    for i, file_info in enumerate(archivos_financieros):
        periodo = file_info["periodo"]
        path = file_info["path"]
        progress_bar.progress((i + 1) / len(archivos_financieros), text=f"Procesando {periodo}...")
        excel_bytes = load_excel_from_dropbox(dbx, path)
        if excel_bytes:
            try:
                df_er, df_bg = procesar_archivo_excel(excel_bytes)
                datos_historicos_periodo = {'df_er_master': df_er, 'df_bg_master': df_bg, 'kpis': {}}
                
                kpis_totales = calcular_kpis_periodo(df_er, df_bg, 'Todos')
                datos_historicos_periodo['kpis']['Todos'] = kpis_totales
                
                er_conf = COL_CONFIG['ESTADO_DE_RESULTADOS']
                cc_cols = [name for name in er_conf.get('CENTROS_COSTO_COLS', {}).values() if name in df_er and name not in ['Total_Consolidado_ER', 'Sin centro de coste']]
                for cc in cc_cols:
                    datos_historicos_periodo['kpis'][cc] = calcular_kpis_periodo(df_er, df_bg, cc)
                
                # Pre-generar el estado de resultados para la vista "Todos" para eficiencia
                datos_historicos_periodo['df_er_display_todos'] = generate_financial_statement(df_er, 'Estado de Resultados', 'Todos', 99)
                
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
    # ... Aquí iría el resto del código para los gráficos de tendencias ...

# ==============================================================================
#       VISTA DE ANÁLISIS DE PERIODO ÚNICO (CON IA Y FILTROS RESTAURADOS)
# ==============================================================================
else:
    st.header(f"Detalle Financiero para el Periodo: {selected_view}")
    data_actual = st.session_state.datos_historicos.get(selected_view)
    if not data_actual:
        st.error(f"No se encontraron datos para el periodo: {selected_view}"); st.stop()

    df_er_actual = data_actual['df_er_master']
    df_bg_actual = data_actual['df_bg_master']
    kpis_por_tienda = data_actual['kpis']
    df_er_display_todos = data_actual.get('df_er_display_todos', pd.DataFrame())

    st.sidebar.subheader("Filtros del Periodo")
    cc_options_all = sorted(list(kpis_por_tienda.keys()))
    cc_filter = st.sidebar.selectbox("Filtrar por Centro de Costo:", cc_options_all, key=f"cc_{selected_view}")
    
    selected_kpis = kpis_por_tienda.get(cc_filter, {})

    st.subheader(f"🔍 KPIs para: {cc_filter}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Margen Neto", f"{selected_kpis.get('margen_neto', 0):.2%}")
    col2.metric("ROE", f"{selected_kpis.get('roe', 0):.2%}")
    col3.metric("Razón Corriente", f"{selected_kpis.get('razon_corriente', 0):.2f}")
    col4.metric("Endeudamiento", f"{selected_kpis.get('endeudamiento_activo', 0):.2%}")

    st.markdown("---")

    with st.spinner('El CFO Virtual está analizando los datos...'):
        analisis_ia = generar_analisis_avanzado_ia(selected_kpis, df_er_actual, cc_filter, selected_view)
    
    st.subheader("🧠 Análisis del CFO Virtual (IA)")
    st.markdown(analisis_ia)
    
    st.markdown("---")
    
    st.subheader("📊 Reportes Detallados")
    
    # Restauramos la sección de filtros de cuentas y nivel
    st.sidebar.subheader("Filtros de Reporte")
    er_conf_sidebar = COL_CONFIG['ESTADO_DE_RESULTADOS']
    nivel_col = er_conf_sidebar.get('NIVEL_LINEA', 'Grupo')
    if nivel_col in df_er_actual.columns:
        max_nivel = int(df_er_actual[nivel_col].max())
        nivel_seleccionado = st.sidebar.slider("Nivel de Detalle de Cuentas:", 1, max_nivel, 1, key=f"nivel_er_{selected_view}")
    else:
        nivel_seleccionado = 1

    # Desplegar los reportes en pestañas para un look más limpio
    tab1, tab2 = st.tabs(["Estado de Resultados", "Balance General"])

    with tab1:
        st.subheader(f"Estado de Resultados ({cc_filter})")
        df_er_display = generate_financial_statement(df_er_actual, 'Estado de Resultados', cc_filter, nivel_seleccionado)
        st.dataframe(df_er_display.style.format({'Valor': "${:,.0f}"}), use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Balance General (Consolidado)")
        df_bg_display = generate_financial_statement(df_bg_actual, 'Balance General', 99)
        st.dataframe(df_bg_display.style.format({'Valor': "${:,.0f}"}), use_container_width=True, hide_index=True)

    # Botón de descarga en la barra lateral
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
