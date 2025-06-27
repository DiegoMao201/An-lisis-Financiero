# kpis_y_analisis.py
import pandas as pd
import streamlit as st
import google.generativeai as genai
from mi_logica_original import get_principal_account_value, COL_CONFIG

def calcular_kpis_periodo(df_er: pd.DataFrame, df_bg: pd.DataFrame, cc_filter: str = 'Todos') -> dict:
    """
    Calcula un set de KPIs para un único periodo.
    Esta función es la ÚNICA fuente de verdad para los cálculos numéricos.
    Se adapta para calcular el consolidado ('Todos') o un centro de costo específico.
    """
    kpis = {}
    er_conf = COL_CONFIG['ESTADO_DE_RESULTADOS']
    bg_conf = COL_CONFIG['BALANCE_GENERAL']

    val_col_kpi = ''
    if cc_filter and cc_filter != 'Todos':
        if cc_filter in df_er.columns:
            val_col_kpi = cc_filter
        else:
            return {"error": f"Centro de costo '{cc_filter}' no encontrado."}
    else:
        total_col_name = er_conf.get('CENTROS_COSTO_COLS', {}).get('Total')
        if total_col_name and total_col_name in df_er.columns:
            val_col_kpi = total_col_name
        else:
            ind_cc_cols = [v for k, v in er_conf.get('CENTROS_COSTO_COLS', {}).items() if str(k).lower() not in ['total', 'sin centro de coste'] and v in df_er.columns]
            if ind_cc_cols:
                df_er['__temp_sum_kpi'] = df_er.loc[:, ind_cc_cols].sum(axis=1)
                val_col_kpi = '__temp_sum_kpi'
            else:
                scc_name = er_conf.get('CENTROS_COSTO_COLS', {}).get('Sin centro de coste')
                if scc_name and scc_name in df_er.columns:
                    val_col_kpi = scc_name
                elif 'Total_Consolidado_ER' in df_er.columns:
                    val_col_kpi = 'Total_Consolidado_ER'

    if not val_col_kpi or val_col_kpi not in df_er.columns: return kpis

    cuenta_er = er_conf['CUENTA']
    ingresos = get_principal_account_value(df_er, '4', val_col_kpi, cuenta_er)
    costo_ventas = get_principal_account_value(df_er, '6', val_col_kpi, cuenta_er)
    gastos_admin = get_principal_account_value(df_er, '51', val_col_kpi, cuenta_er)
    gastos_ventas = get_principal_account_value(df_er, '52', val_col_kpi, cuenta_er)
    costos_prod = get_principal_account_value(df_er, '7', val_col_kpi, cuenta_er)
    gastos_no_op = get_principal_account_value(df_er, '53', val_col_kpi, cuenta_er)
    impuestos = get_principal_account_value(df_er, '54', val_col_kpi, cuenta_er)
    
    utilidad_bruta = ingresos + costo_ventas
    gastos_operativos = gastos_admin + gastos_ventas + costos_prod
    utilidad_operacional = utilidad_bruta + gastos_operativos
    utilidad_neta = utilidad_operacional + gastos_no_op + impuestos

    kpis['ingresos'] = ingresos
    kpis['costo_ventas'] = costo_ventas
    kpis['gastos_operativos'] = gastos_operativos
    kpis['utilidad_bruta'] = utilidad_bruta
    kpis['utilidad_operacional'] = utilidad_operacional
    kpis['utilidad_neta'] = utilidad_neta
    
    cuenta_bg = bg_conf['CUENTA']
    saldo_final_col = bg_conf['SALDO_FINAL']

    activo = get_principal_account_value(df_bg, '1', saldo_final_col, cuenta_bg)
    pasivo = get_principal_account_value(df_bg, '2', saldo_final_col, cuenta_bg)
    patrimonio = get_principal_account_value(df_bg, '3', saldo_final_col, cuenta_bg)
    activo_corriente = sum([get_principal_account_value(df_bg, c, saldo_final_col, cuenta_bg) for c in ['11','12','13','14']])
    inventarios = get_principal_account_value(df_bg, '14', saldo_final_col, cuenta_bg)
    pasivo_corriente = sum([get_principal_account_value(df_bg, c, saldo_final_col, cuenta_bg) for c in ['21','22','23']])

    kpis['razon_corriente'] = activo_corriente / pasivo_corriente if pasivo_corriente != 0 else 0
    kpis['endeudamiento_activo'] = pasivo / activo if activo != 0 else 0
    kpis['roe'] = utilidad_neta / patrimonio if patrimonio != 0 else 0
    kpis['margen_neto'] = utilidad_neta / ingresos if ingresos > 0 else 0
    kpis['margen_operacional'] = utilidad_operacional / ingresos if ingresos > 0 else 0
    
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

@st.cache_data(show_spinner=False)
def generar_analisis_avanzado_ia(_kpis_actuales: dict, _df_er_actual: pd.DataFrame, nombre_cc: str, periodo_actual: str):
    """
    Genera un análisis financiero profundo y visualmente atractivo utilizando el modelo Gemini de Google.
    """
    try:
        api_key = st.secrets["google_ai"]["api_key"]
        genai.configure(api_key=api_key)
    except Exception:
        return "🔴 **Error:** No se encontró la clave de API de Google AI en los secretos."

    er_conf = COL_CONFIG['ESTADO_DE_RESULTADOS']
    cuenta_col_er = er_conf.get('CUENTA', 'Cuenta')
    nombre_col_er = er_conf.get('NOMBRE_CUENTA', 'Título')
    
    gastos_df = _df_er_actual[_df_er_actual.get(cuenta_col_er, pd.Series(dtype=str)).str.startswith(('5', '7'), na=False)].copy()

    val_col = ''
    if nombre_cc != 'Todos' and nombre_cc in gastos_df.columns:
        val_col = nombre_cc
    elif 'Total_Consolidado_ER' in gastos_df.columns:
        val_col = 'Total_Consolidado_ER'
    else:
        ind_cc_cols = [v for k, v in er_conf.get('CENTROS_COSTO_COLS',{}).items() if str(k).lower() not in ['total', 'sin centro de coste'] and v in gastos_df.columns]
        if ind_cc_cols:
            gastos_df['__temp_sum_gastos'] = gastos_df[ind_cc_cols].sum(axis=1)
            val_col = '__temp_sum_gastos'

    top_5_gastos_str = "No se pudieron determinar los gastos principales."
    if val_col and nombre_col_er in gastos_df.columns and val_col in gastos_df.columns:
        gastos_df_filtered = gastos_df[[nombre_col_er, val_col]].dropna()
        gastos_df_filtered[val_col] = pd.to_numeric(gastos_df_filtered[val_col], errors='coerce').abs()
        top_5_gastos = gastos_df_filtered.nlargest(5, val_col)
        top_5_gastos_str = "\n".join([f"- {row.iloc[0]}: ${row.iloc[1]:,.0f}" for _, row in top_5_gastos.iterrows()])

    prompt = f"""
    **Rol:** Actúa como un Asesor Financiero Estratégico y un experto en comunicación para la alta gerencia. Tu objetivo es transformar datos crudos en un informe gerencial claro, conciso y visualmente atractivo que impulse la toma de decisiones.

    **Contexto:** Estás analizando los resultados del centro de costo: "{nombre_cc}" para el periodo: "{periodo_actual}".

    **Datos Financieros Clave:**
    - **Ingresos:** ${_kpis_actuales.get('ingresos', 0):,.0f}
    - **Utilidad Neta:** ${_kpis_actuales.get('utilidad_neta', 0):,.0f}
    - **Margen Neto:** {_kpis_actuales.get('margen_neto', 0):.2%}
    - **Rentabilidad sobre Patrimonio (ROE):** {_kpis_actuales.get('roe', 0):.2%}
    - **Razón Corriente (Liquidez):** {_kpis_actuales.get('razon_corriente', 0):.2f}
    - **Nivel de Endeudamiento (sobre Activo):** {_kpis_actuales.get('endeudamiento_activo', 0):.2%}
    - **Gastos Operativos Totales:** ${_kpis_actuales.get('gastos_operativos', 0):,.0f}

    **Top 5 Gastos Operativos del Periodo:**
    {top_5_gastos_str}

    **Instrucciones de Formato y Contenido:**
    Tu respuesta debe ser un informe gerencial profesional, fácil de leer y visualmente organizado. Usa emojis de forma inteligente (ej: 📈, 📉, ⚠️, ✅, 💡) para guiar la vista y enfatizar los puntos más importantes. La estructura debe ser exactamente la siguiente:

    ### Diagnóstico General 🎯
    (Ofrece un veredicto claro y directo en un párrafo sobre la salud financiera. ¿La situación es excelente, buena, preocupante o crítica? Sé directo y justifica tu veredicto inicial con 1 o 2 datos clave.)

    ### Puntos Clave del Periodo 🔑
    (Presenta un análisis en formato de lista (bullet points). Para cada punto, no solo menciones el dato, sino su **implicación de negocio**. Por ejemplo: en lugar de 'La razón corriente es 0.8', di '⚠️ **Alerta de Liquidez (Razón Corriente: 0.8):** Existe un riesgo de no poder cubrir las deudas a corto plazo, lo que requiere atención inmediata al flujo de caja.')
    - **Rentabilidad:** Analiza el Margen Neto y el ROE. ¿Se está generando valor de forma eficiente?
    - **Estructura de Costos:** Analiza los gastos operativos en relación con los ingresos. ¿Son sostenibles? Comenta sobre los gastos más significativos.
    - **Solvencia y Riesgo:** Analiza la liquidez (Razón Corriente) y el nivel de endeudamiento. ¿Qué tan riesgosa es la estructura de capital?

    ### Plan de Acción Recomendado 💡
    (Proporciona una lista de 2 a 3 recomendaciones **específicas, priorizadas y accionables** basadas en el diagnóstico. No des consejos genéricos. Si los gastos de personal son altos, sugiere '1. Realizar un análisis de la estructura de personal vs. ingresos para identificar optimizaciones.' en lugar de solo 'reducir gastos'.)
    """

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        cleaned_response = response.text.replace('•', '*')
        return cleaned_response
    except Exception as e:
        return f"🔴 **Error al contactar la IA:** {e}. Revisa la API Key y la configuración."
```

---

### **2. Archivo Final y Completo: `Dashboard_Principal.py`**

Este archivo se mantiene igual que la última versión que te envié, que ya es la completa y funcional. Lo incluyo de nuevo aquí para que tengas la certeza absoluta de que posees el par de archivos correcto y completo.

````python
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

st.markdown("""
<style>
.ai-analysis-text {
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
                
                kpis_totales = calcular_kpis_periodo(df_er, df_bg, 'Todos')
                datos_periodo['kpis']['Todos'] = kpis_totales
                
                er_conf = COL_CONFIG['ESTADO_DE_RESULTADOS']
                cc_cols = [name for name in er_conf.get('CENTROS_COSTO_COLS', {}).values() if name in df_er and name not in ['Total_Consolidado_ER', 'Sin centro de coste']]
                for cc in cc_cols:
                    datos_periodo['kpis'][cc] = calcular_kpis_periodo(df_er, df_bg, cc)
                
                datos_historicos[periodo] = datos_periodo
            except Exception as e:
                st.error(f"Error al procesar el archivo del periodo {periodo}: {e}")
    
    progress_bar.empty()
    return datos_historicos

if 'datos_historicos' not in st.session_state: st.session_state.datos_historicos = None
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
