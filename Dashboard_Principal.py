# Dashboard_Principal.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import io # Necesario para el nuevo buffer de Excel
from datetime import datetime

# --- Importamos nuestros módulos ---
# Asegúrate de que estos archivos .py estén en la misma carpeta
from mi_logica_original import procesar_archivo_excel, generate_financial_statement, COL_CONFIG
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
#         NUEVA FUNCIÓN PARA CONSOLIDACIÓN ANUAL (CORREGIDA)
# ==============================================================================
def consolidar_datos_anuales(datos_historicos: Dict[str, Any], anio: int) -> Optional[Dict[str, Any]]:
    """
    Consolida los datos de todos los meses de un año específico.

    Args:
        datos_historicos: El diccionario con todos los datos cargados.
        anio: El año a consolidar (ej. 2023).

    Returns:
        Un diccionario con los DataFrames y KPIs consolidados para el año,
        o None si no hay datos para ese año.
    """
    periodos_del_anio = [p for p in datos_historicos.keys() if str(p).startswith(str(anio))]
    if not periodos_del_anio:
        st.warning(f"No se encontraron datos para el año {anio}.")
        return None

    st.info(f"Consolidando {len(periodos_del_anio)} periodos para el año {anio}...")

    # --- Consolidación del Estado de Resultados (P&L) ---
    # Se suman los valores de todos los meses.
    df_er_anual_list = []
    er_conf = COL_CONFIG['ESTADO_DE_RESULTADOS']
    cuenta_col_er = er_conf.get('CUENTA', 'Cuenta')
    desc_col_er = er_conf.get('NOMBRE_CUENTA', 'Título')
    nivel_col_er = er_conf.get('NIVEL_LINEA', 'Grupo')
    cc_cols = er_conf.get('CENTROS_COSTO_COLS', {}).values()
    
    # Identificar todas las columnas numéricas que deben sumarse
    columnas_a_sumar_er = []
    primer_df_er = datos_historicos[periodos_del_anio[0]]['df_er_master']
    for col in primer_df_er.columns:
        if col in cc_cols or col == 'Total_Consolidado_ER':
             columnas_a_sumar_er.append(col)

    for periodo in periodos_del_anio:
        # ▼▼▼ CORRECCIÓN APLICADA ▼▼▼
        # Se corrigió el nombre de la variable de 'df_er_list' a 'df_er_anual_list'
        df_er_anual_list.append(datos_historicos[periodo]['df_er_master'])

    df_er_anual = pd.concat(df_er_anual_list)
    
    columnas_agrupacion_er = [cuenta_col_er, desc_col_er, nivel_col_er]
    # Asegurarnos de que las columnas de agrupación existan antes de agrupar
    columnas_agrupacion_er_validas = [col for col in columnas_agrupacion_er if col in df_er_anual.columns]
    
    df_er_anual = df_er_anual.groupby(columnas_agrupacion_er_validas, as_index=False)[columnas_a_sumar_er].sum()

    # --- Consolidación del Balance General (BS) ---
    # Se toma el balance del último mes del año.
    ultimo_periodo = sorted(periodos_del_anio)[-1]
    df_bg_anual = datos_historicos[ultimo_periodo]['df_bg_master'].copy()

    # --- Recalcular KPIs para el consolidado anual ---
    # Usamos los DFs anuales que acabamos de crear.
    kpis_anuales = {'Todos': calcular_kpis_periodo(df_er_anual, df_bg_anual, 'Todos')}
    
    cc_cols_kpis = [name for name in er_conf.get('CENTROS_COSTO_COLS', {}).values() if name in df_er_anual and name not in ['Total_Consolidado_ER', 'Sin centro de coste']]
    for cc in cc_cols_kpis:
        kpis_anuales[cc] = calcular_kpis_periodo(df_er_anual, df_bg_anual, cc)
        
    datos_anuales_consolidados = {
        'df_er_master': df_er_anual,
        'df_bg_master': df_bg_anual,
        'kpis': kpis_anuales
    }

    return datos_anuales_consolidados

# ==============================================================================
#           NUEVA FUNCIÓN PARA EXCEL PROFESIONAL (CORREGIDA Y MODULAR)
# ==============================================================================

def _escribir_hoja_resumen(writer, datos_periodo, periodo_actual_str, formats):
    """Función auxiliar para escribir la hoja de Resumen Gerencial en el Excel."""
    ws = writer.book.add_worksheet('Resumen Gerencial')
    kpis_por_cc = datos_periodo['kpis']
    cc_list = sorted(kpis_por_cc.keys())

    kpi_data = {
        "Indicador": [
            "Utilidad Neta", "Ingresos Totales", "Gastos Operativos", "Margen Neto",
            "ROE", "Razón Corriente", "Endeudamiento del Activo"
        ]
    }
    for cc in cc_list:
        kpis = kpis_por_cc.get(cc, {})
        kpi_data[cc] = [
            kpis.get('utilidad_neta', 0), kpis.get('ingresos', 0), kpis.get('gastos_operativos', 0),
            kpis.get('margen_neto', 0), kpis.get('roe', 0), kpis.get('razon_corriente', 0),
            kpis.get('endeudamiento_activo', 0)
        ]

    df_kpis = pd.DataFrame(kpi_data)

    ws.merge_range(0, 0, 0, len(cc_list), f"Resumen Gerencial - Periodo: {periodo_actual_str}", formats['header'])
    for col_num, value in enumerate(df_kpis.columns.values):
        ws.write(2, col_num, value, formats['header'])

    for row_num, row_data in enumerate(df_kpis.itertuples(index=False), start=3):
        ws.write(row_num, 0, row_data[0])
        for col_num, cell_data in enumerate(row_data[1:], start=1):
            if any(kpi in row_data[0] for kpi in ["Margen", "ROE", "Endeudamiento"]):
                ws.write(row_num, col_num, cell_data, formats['percent'])
            elif "Razón" in row_data[0]:
                ws.write(row_num, col_num, cell_data, formats['decimal'])
            else:
                ws.write(row_num, col_num, cell_data, formats['currency'])

    ws.set_column('A:A', 25)
    ws.set_column('B:Z', 18)

def _escribir_hoja_er(writer, df_er_master, formats, nivel_detalle: int):
    """Función auxiliar para escribir la hoja de Estado de Resultados."""
    ws = writer.book.add_worksheet('Estado de Resultados')
    er_conf = COL_CONFIG['ESTADO_DE_RESULTADOS']
    cuenta_col = er_conf.get('CUENTA', 'Cuenta')
    desc_col = er_conf.get('NOMBRE_CUENTA', 'Título')
    nivel_col = er_conf.get('NIVEL_LINEA', 'Grupo')

    df_er_filtrado = df_er_master[df_er_master[nivel_col] <= nivel_detalle].copy()

    df_reporte_er = df_er_filtrado[[cuenta_col, desc_col, nivel_col]].drop_duplicates().sort_values(cuenta_col)

    cc_cols_er = [name for name in er_conf.get('CENTROS_COSTO_COLS', {}).values() if name in df_er_master]
    if 'Total_Consolidado_ER' in df_er_master.columns and 'Total_Consolidado_ER' not in cc_cols_er:
        cc_cols_er.append('Total_Consolidado_ER')

    for cc in sorted(cc_cols_er):
        if cc in df_er_master:
            df_cc_data = df_er_master[[cuenta_col, cc]].copy()
            df_reporte_er = pd.merge(df_reporte_er, df_cc_data, on=cuenta_col, how='left')

    cc_cols_sorted = sorted(cc_cols_er)
    final_cols = [cuenta_col, desc_col, nivel_col] + cc_cols_sorted
    df_reporte_er = df_reporte_er.reindex(columns=final_cols).fillna(0)

    er_headers = [cuenta_col, desc_col] + cc_cols_sorted
    ws.write_row(0, 0, er_headers, formats['header'])

    for row_num, record in enumerate(df_reporte_er.to_dict('records'), start=1):
        nivel = record.get(nivel_col, 99)
        is_total_row = nivel <= 1

        row_format = formats['nivel4'] if is_total_row else None
        cell_format_currency = formats['total'] if is_total_row else formats['currency']

        ws.write(row_num, 0, record[cuenta_col], row_format)
        ws.write(row_num, 1, record[desc_col], row_format)

        for col_num, cc_name in enumerate(cc_cols_sorted, start=2):
            ws.write(row_num, col_num, record[cc_name], cell_format_currency)

    ws.set_column('A:A', 12)
    ws.set_column('B:B', 40)
    ws.set_column('C:Z', 18)


def _escribir_hoja_bg(writer, df_bg_master, formats):
    """Función auxiliar para escribir la hoja de Balance General."""
    ws = writer.book.add_worksheet('Balance General')
    df_bg_display = generate_financial_statement(df_bg_master, 'Balance General', 99)

    bg_headers = ['Cuenta', 'Descripción', 'Valor']
    ws.write_row(0, 0, bg_headers, formats['header'])

    for row_num, record in enumerate(df_bg_display.to_dict('records'), start=1):
        cuenta_val = record.get('Cuenta', '')
        is_total = not str(cuenta_val).isdigit()
        cell_format = formats['total'] if is_total else formats['currency']

        ws.write(row_num, 0, cuenta_val)
        ws.write(row_num, 1, record.get('Descripción', ''))

        valor_crudo = record.get('Valor')

        if pd.isna(valor_crudo):
            valor_celda = 0
        else:
            valor_celda = valor_crudo

        ws.write(row_num, 2, valor_celda, cell_format)

    ws.set_column('A:A', 15)
    ws.set_column('B:B', 45)
    ws.set_column('C:C', 20)

def generar_excel_gerencial_profesional(
    df_er_master: pd.DataFrame,
    df_bg_master: pd.DataFrame,
    datos_periodo: Dict[str, Any],
    periodo_actual_str: str,
    nivel_detalle_er: int
) -> bytes:
    """
    Crea un archivo Excel profesional y gerencial con múltiples pestañas y formato avanzado.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        workbook = writer.book

        formats = {
            'header': workbook.add_format({
                'bold': True, 'text_wrap': True, 'valign': 'top', 'align': 'center',
                'fg_color': '#1F497D', 'font_color': 'white', 'border': 1
            }),
            'currency': workbook.add_format({'num_format': '$#,##0;[Red]($#,##0)'}),
            'percent': workbook.add_format({'num_format': '0.00%'}),
            'decimal': workbook.add_format({'num_format': '0.00'}),
            'total': workbook.add_format({'bold': True, 'num_format': '$#,##0;[Red]($#,##0)', 'top': 1, 'bottom': 1}),
            'nivel4': workbook.add_format({'bold': True})
        }

        _escribir_hoja_resumen(writer, datos_periodo, periodo_actual_str, formats)
        _escribir_hoja_er(writer, df_er_master, formats, nivel_detalle_er)
        _escribir_hoja_bg(writer, df_bg_master, formats)

    buffer.seek(0)
    return buffer.getvalue()


# ==============================================================================
#           FUNCIONES AUXILIARES DE ANÁLISIS Y VISUALIZACIÓN (Originales)
# ==============================================================================

def plot_sparkline(data: pd.Series, title: str, is_percent: bool = False, lower_is_better: bool = False):
    """Crea un minigráfico de línea mejorado para KPIs."""
    if data.empty or len(data.dropna()) < 2:
        return go.Figure().update_layout(width=150, height=50, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', annotations=[dict(text="N/A", showarrow=False)])

    last_val = data.iloc[-1]
    first_val = data.iloc[0]

    if (lower_is_better and last_val < first_val) or (not lower_is_better and last_val > first_val):
        color = '#28a745'
    else:
        color = '#dc3545'

    fig = go.Figure(go.Scatter(
        x=list(range(len(data))), y=data, mode='lines',
        line=dict(color=color, width=2.5),
        fill='tozeroy',
        fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)"
    ))
    fig.update_layout(
        width=150, height=50, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=5, b=0), xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False
    )
    return fig

def calcular_variaciones_er(df_actual: pd.DataFrame, df_previo: pd.DataFrame, cc_filter: str) -> pd.DataFrame:
    """Calcula las variaciones absolutas y porcentuales para el Estado de Resultados."""
    er_conf = COL_CONFIG['ESTADO_DE_RESULTADOS']
    cuenta_col = er_conf.get('CUENTA', 'Cuenta')
    desc_col = er_conf.get('NOMBRE_CUENTA', 'Título')
    valor_col_nombre = 'Total_Consolidado_ER' if cc_filter == 'Todos' else cc_filter

    required_cols_actual = [cuenta_col, desc_col, valor_col_nombre]
    missing_cols_actual = [col for col in required_cols_actual if col not in df_actual.columns]

    if missing_cols_actual:
        st.error(
            f"Error Crítico (Periodo Actual): Faltan columnas: **{', '.join(missing_cols_actual)}**."
            f" Revisa tu archivo Excel y la configuración 'COL_CONFIG'."
        )
        st.info(f"Columnas disponibles: {list(df_actual.columns)}")
        return pd.DataFrame()

    df1 = df_actual[required_cols_actual].copy()
    df1.rename(columns={valor_col_nombre: 'Valor_actual'}, inplace=True)

    required_cols_previo = [cuenta_col, desc_col, valor_col_nombre]

    if all(col in df_previo.columns for col in required_cols_previo):
        df2 = df_previo[required_cols_previo].copy()
        df2.rename(columns={valor_col_nombre: 'Valor_previo'}, inplace=True)
    else:
        st.warning(f"ADVERTENCIA: '{valor_col_nombre}' no encontrado en periodo anterior. Se usarán ceros.")
        base_cols_previo = [cuenta_col, desc_col]
        missing_base_cols = [col for col in base_cols_previo if col not in df_previo.columns]
        if missing_base_cols:
            st.error(f"Error Crítico (Periodo Previo): Faltan columnas base: {', '.join(missing_base_cols)}")
            return pd.DataFrame()
        df2 = df_previo[base_cols_previo].copy()
        df2['Valor_previo'] = 0

    df_variacion = pd.merge(df1, df2, on=[cuenta_col, desc_col], how='outer').fillna(0)
    df_variacion['Variacion_Absoluta'] = df_variacion['Valor_actual'] - df_variacion['Valor_previo']

    if desc_col != 'Descripción':
        df_variacion.rename(columns={desc_col: 'Descripción'}, inplace=True)

    return df_variacion

def plot_waterfall_utilidad_neta(df_variacion: pd.DataFrame, periodo_actual: str, periodo_previo: str):
    """Crea un gráfico de cascada para explicar la variación de la Utilidad Neta."""
    cuenta_col = COL_CONFIG['ESTADO_DE_RESULTADOS'].get('CUENTA', 'Cuenta')
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
        x=["Utilidad Neta " + str(periodo_previo), "Ingresos", "Costos", "Gastos Op.", "Otros", "Utilidad Neta " + str(periodo_actual)],
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
#                 CONFIGURACIÓN DE PÁGINA Y AUTENTICACIÓN
# ==============================================================================
st.set_page_config(layout="wide", page_title="Análisis Financiero Inteligente PRO")
st.title("🤖 Dashboard Financiero Profesional con IA")

st.markdown("""<style>.reportview-container{background:#f0f2f6}.kpi-card{padding:1rem;border-radius:0.5rem;box-shadow:0 4px 6px rgba(0,0,0,0.1);background-color:white;text-align:center}.ai-analysis-text{background-color:#e8f0fe;border-left:5px solid #1967d2;padding:15px;border-radius:5px;font-size:1.05em}</style>""", unsafe_allow_html=True)

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
if st.sidebar.button("Refrescar Datos de Dropbox", use_container_width=True, type="primary"):
    st.cache_data.clear()
    st.session_state.datos_historicos = None
if st.session_state.datos_historicos is None:
    st.session_state.datos_historicos = cargar_y_procesar_datos()
if not st.session_state.datos_historicos:
    st.error("No se pudieron cargar datos. Verifica la conexión y la estructura de archivos en Dropbox.")
    st.stop()

# ==============================================================================
#                   INTERFAZ DE USUARIO PRINCIPAL
# ==============================================================================
st.sidebar.title("Opciones de Análisis")
sorted_periods = sorted(st.session_state.datos_historicos.keys(), reverse=True)

# ▼▼▼ BLOQUE CORREGIDO PARA EVITAR EL ERROR 'ValueError' ▼▼▼
# Este bloque ahora maneja de forma segura los periodos que no tengan el formato 'YYYYMM'.
valid_years = []
for period_str in sorted_periods:
    try:
        # Intenta convertir la cadena a fecha para extraer el año
        year = pd.to_datetime(period_str, format='%Y%m').year
        valid_years.append(year)
    except ValueError:
        # Si una cadena no coincide con el formato (ej. archivo mal nombrado),
        # se ignora para no detener la aplicación.
        st.warning(f"Se ignoró un periodo con formato inválido encontrado: '{period_str}'")
        continue

# Obtenemos la lista única y ordenada de años válidos.
available_years = sorted(list(set(valid_years)), reverse=True)

period_options = ["Análisis de Evolución (Tendencias)", "Análisis Anual"] + sorted_periods
selected_view = st.sidebar.selectbox("Selecciona la vista de análisis:", period_options)

# Variable para controlar si la vista es anual
is_annual_view = False
selected_year = None
# Si se elige la vista anual, mostramos un selector de año
if selected_view == "Análisis Anual":
    is_annual_view = True
    if available_years:
        selected_year = st.sidebar.selectbox("Selecciona el año a consolidar:", available_years)
        selected_view = f"Consolidado Anual {selected_year}"
    else:
        st.sidebar.error("No hay años con datos válidos para analizar.")
        st.stop()
# ==============================================================================
#               VISTA DE ANÁLISIS DE TENDENCIAS
# ==============================================================================
if selected_view == "Análisis de Evolución (Tendencias)":
    st.header("📈 Informe de Evolución Gerencial")
    df_tendencia = preparar_datos_tendencia(st.session_state.datos_historicos)

    if df_tendencia.empty or len(df_tendencia) < 2:
        st.info("Se necesitan al menos dos periodos de datos para generar un análisis de evolución.")
        st.stop()

    with st.spinner('El Analista Senior IA está evaluando la trayectoria plurianual...'):
        analisis_tendencia_ia = generar_analisis_tendencia_ia(df_tendencia)

    st.markdown("### Diagnóstico Estratégico IA")
    st.markdown(analisis_tendencia_ia, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Indicadores Clave de Desempeño (KPIs) a través del Tiempo")

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.metric("Margen Neto", f"{df_tendencia['margen_neto'].iloc[-1]:.2%}")
        st.plotly_chart(plot_sparkline(df_tendencia['margen_neto'], 'Tendencia Margen Neto'), use_container_width=True)
    with kpi_cols[1]:
        st.metric("ROE (Retorno sobre Patrimonio)", f"{df_tendencia['roe'].iloc[-1]:.2%}")
        st.plotly_chart(plot_sparkline(df_tendencia['roe'], 'Tendencia ROE'), use_container_width=True)
    with kpi_cols[2]:
        st.metric("Razón Corriente (Liquidez)", f"{df_tendencia['razon_corriente'].iloc[-1]:.2f}")
        st.plotly_chart(plot_sparkline(df_tendencia['razon_corriente'], 'Tendencia Liquidez'), use_container_width=True)
    with kpi_cols[3]:
        st.metric("Endeudamiento del Activo", f"{df_tendencia['endeudamiento_activo'].iloc[-1]:.2%}")
        st.plotly_chart(plot_sparkline(df_tendencia['endeudamiento_activo'], 'Tendencia Endeudamiento', lower_is_better=True), use_container_width=True)

    st.markdown("---")
    st.subheader("Evolución de Componentes Financieros Principales")

    fig_combinada = go.Figure()
    fig_combinada.add_trace(go.Bar(x=df_tendencia['periodo'], y=df_tendencia['ingresos'], name='Ingresos', marker_color='#28a745'))
    fig_combinada.add_trace(go.Bar(x=df_tendencia['periodo'], y=df_tendencia['gastos_operativos'].abs(), name='Gastos Operativos', marker_color='#ffc107'))
    fig_combinada.add_trace(go.Scatter(x=df_tendencia['periodo'], y=df_tendencia['utilidad_neta'], name='Utilidad Neta', mode='lines+markers', line=dict(color='#0d6efd', width=4)))
    fig_combinada.update_layout(
        title='Evolución de Ingresos, Gastos y Utilidad Neta', barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), yaxis_title="Monto (COP)"
    )
    st.plotly_chart(fig_combinada, use_container_width=True)

# ==============================================================================
#       VISTA DE PERIODO ÚNICO O ANUAL (CENTRO DE ANÁLISIS PROFUNDO)
# ==============================================================================
else:
    st.header(f"Centro de Análisis para: {selected_view}")

    if is_annual_view and selected_year:
        data_actual = consolidar_datos_anuales(st.session_state.datos_historicos, selected_year)
        data_previa = consolidar_datos_anuales(st.session_state.datos_historicos, selected_year - 1)
        periodo_previo = f"Consolidado Anual {selected_year - 1}" if data_previa else None
    else:
        data_actual = st.session_state.datos_historicos.get(selected_view)
        periodo_actual_idx = sorted_periods.index(selected_view)
        periodo_previo = sorted_periods[periodo_actual_idx + 1] if periodo_actual_idx + 1 < len(sorted_periods) else None
        data_previa = st.session_state.datos_historicos.get(periodo_previo) if periodo_previo else None

    if not data_actual:
        st.error(f"No se encontraron datos para la selección: {selected_view}"); st.stop()

    df_er_actual = data_actual['df_er_master']
    df_bg_actual = data_actual['df_bg_master']
    kpis_por_tienda = data_actual['kpis']

    st.sidebar.subheader("Filtros del Periodo")
    cc_options_all = sorted(list(kpis_por_tienda.keys()))
    cc_filter = st.sidebar.selectbox("Filtrar por Centro de Costo:", cc_options_all, key=f"cc_{selected_view}")

    er_conf = COL_CONFIG['ESTADO_DE_RESULTADOS']
    nivel_col = er_conf.get('NIVEL_LINEA', 'Grupo')
    if nivel_col in df_er_actual.columns:
        max_nivel = int(pd.to_numeric(df_er_actual[nivel_col], errors='coerce').max())
        nivel_seleccionado = st.sidebar.slider("Nivel de Detalle de Cuentas:", 1, max_nivel, 4, key=f"nivel_er_{selected_view}")
    else:
        nivel_seleccionado = 4

    st.sidebar.subheader("Buscador de Cuentas")
    search_account_input = st.sidebar.text_input("Buscar por número de cuenta:", key=f"search_{selected_view}", placeholder="Ej: 510506")

    df_variacion_er = None
    if data_previa:
        df_variacion_er = calcular_variaciones_er(df_er_actual, data_previa['df_er_master'], cc_filter)
        if df_variacion_er is not None and not df_variacion_er.empty:
            st.info(f"Análisis comparativo contra: **{periodo_previo}**.")
    else:
        st.warning("No hay un periodo/año anterior para realizar análisis comparativo.")

    tab_gen, tab_utilidad, tab_ing, tab_gas, tab_roe, tab_rep = st.tabs([
        "📊 Resumen General", "💰 Análisis de Utilidad Neta", "📈 Análisis de Ingresos",
        "🧾 Análisis de Gastos", "🎯 Análisis ROE (DuPont)", "📋 Reportes Financieros"
    ])

    with tab_gen:
        st.subheader(f"Resumen Ejecutivo para: {cc_filter}")
        selected_kpis = kpis_por_tienda.get(cc_filter, {})

        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        kpi_col1.metric("Margen Neto", f"{selected_kpis.get('margen_neto', 0):.2%}")
        kpi_col2.metric("ROE", f"{selected_kpis.get('roe', 0):.2%}")
        kpi_col3.metric("Razón Corriente", f"{selected_kpis.get('razon_corriente', 0):.2f}")
        kpi_col4.metric("Endeudamiento", f"{selected_kpis.get('endeudamiento_activo', 0):.2%}")

        st.markdown("---")
        with st.expander("🧠 **Ver Análisis y Consejos del CFO Virtual (IA)**", expanded=True):
            with st.spinner('El CFO Virtual está preparando un análisis profundo...'):
                contexto_ia = {
                    "kpis": selected_kpis, "periodo": selected_view, "centro_costo": cc_filter,
                    "convencion_contable": (
                        "REGLA DE ORO: Analizas datos con LÓGICA MIXTA. "
                        "1. ESTADO DE RESULTADOS: Es estándar (Ingresos +, Gastos -). Una variación positiva es buena. "
                        "2. BALANCE GENERAL: Es de sistema (Activos +, Pasivos y Patrimonio -)."
                    ),
                    "variaciones_favorables": [], "variaciones_desfavorables": []
                }
                if df_variacion_er is not None and not df_variacion_er.empty:
                    top_favorables = df_variacion_er.nlargest(5, 'Variacion_Absoluta')
                    top_desfavorables = df_variacion_er.nsmallest(5, 'Variacion_Absoluta')
                    if 'Descripción' in top_favorables.columns and 'Variacion_Absoluta' in top_favorables.columns:
                        contexto_ia["variaciones_favorables"] = top_favorables[['Descripción', 'Variacion_Absoluta']].to_dict('records')
                    if 'Descripción' in top_desfavorables.columns and 'Variacion_Absoluta' in top_desfavorables.columns:
                        contexto_ia["variaciones_desfavorables"] = top_desfavorables[['Descripción', 'Variacion_Absoluta']].to_dict('records')

                analisis_ia = generar_analisis_avanzado_ia(contexto_ia)
                st.markdown(f"<div class='ai-analysis-text'>{analisis_ia}</div>", unsafe_allow_html=True)

    with tab_utilidad:
        st.subheader(f"💰 Análisis de la Utilidad Neta: ¿Qué movió el resultado?")
        if df_variacion_er is not None and not df_variacion_er.empty and periodo_previo:
            st.plotly_chart(plot_waterfall_utilidad_neta(df_variacion_er, selected_view, periodo_previo), use_container_width=True)

            st.markdown(f"#### Principales Motores del Cambio vs. {periodo_previo}")
            col1, col2 = st.columns(2)

            top_favorables = df_variacion_er[df_variacion_er['Variacion_Absoluta'] > 0].sort_values('Variacion_Absoluta', ascending=False).head(10)
            top_desfavorables = df_variacion_er[df_variacion_er['Variacion_Absoluta'] < 0].sort_values('Variacion_Absoluta').head(10)
            format_dict = {'Valor_previo': '${:,.0f}', 'Valor_actual': '${:,.0f}', 'Variacion_Absoluta': '${:,.0f}'}

            with col1:
                st.markdown("✅ **Impactos Positivos (Ayudaron a la Utilidad)**")
                st.dataframe(top_favorables[['Descripción', 'Valor_previo', 'Valor_actual', 'Variacion_Absoluta']].style.format(format_dict).background_gradient(cmap='Greens', subset=['Variacion_Absoluta']), use_container_width=True)
            with col2:
                st.markdown("❌ **Impactos Negativos (Perjudicaron la Utilidad)**")
                st.dataframe(top_desfavorables[['Descripción', 'Valor_previo', 'Valor_actual', 'Variacion_Absoluta']].style.format(format_dict).background_gradient(cmap='Reds_r', subset=['Variacion_Absoluta']), use_container_width=True)
        else:
            st.info("Se requiere un periodo/año anterior para este análisis.")

    with tab_ing:
        st.subheader("📈 Análisis Detallado de Ingresos")
        cuenta_col = er_conf.get('CUENTA', 'Cuenta')
        valor_col_nombre = 'Total_Consolidado_ER' if cc_filter == 'Todos' else cc_filter

        if df_variacion_er is not None and not df_variacion_er.empty and periodo_previo:
            df_ing_var = df_variacion_er[df_variacion_er[cuenta_col].astype(str).str.startswith('4')]
            st.markdown(f"##### Comparativo de Ingresos vs. {periodo_previo}")
            st.bar_chart(data=df_ing_var.set_index('Descripción')[['Valor_actual', 'Valor_previo']])
            format_dict = {'Valor_previo': '${:,.0f}', 'Valor_actual': '${:,.0f}', 'Variacion_Absoluta': '${:,.0f}'}
            st.dataframe(df_ing_var[['Descripción', 'Valor_previo', 'Valor_actual', 'Variacion_Absoluta']].style.format(format_dict), use_container_width=True)
        else:
            if valor_col_nombre in df_er_actual.columns and cuenta_col in df_er_actual.columns:
                df_ingresos = df_er_actual[df_er_actual[cuenta_col].astype(str).str.startswith('4')]
                desc_col_name = COL_CONFIG['ESTADO_DE_RESULTADOS'].get('NOMBRE_CUENTA', 'Título')
                st.bar_chart(data=df_ingresos.set_index(desc_col_name)[valor_col_nombre])
                st.dataframe(df_ingresos[[desc_col_name, valor_col_nombre]], use_container_width=True)

    with tab_gas:
        st.subheader("🧾 Análisis Detallado de Gastos")
        cuenta_col = er_conf.get('CUENTA', 'Cuenta')
        valor_col_nombre = 'Total_Consolidado_ER' if cc_filter == 'Todos' else cc_filter

        if valor_col_nombre in df_er_actual.columns and cuenta_col in df_er_actual.columns:
            df_gastos = df_er_actual[df_er_actual[cuenta_col].astype(str).str.startswith('5')]
            st.markdown(f"#### Composición de Gastos de {selected_view}")
            desc_col_name = COL_CONFIG['ESTADO_DE_RESULTADOS'].get('NOMBRE_CUENTA', 'Título')
            fig_treemap = px.treemap(df_gastos, path=[px.Constant("Total Gastos"), desc_col_name], values=df_gastos[valor_col_nombre].abs(),
                                     title='Distribución de Gastos Operacionales',
                                     color=df_gastos[valor_col_nombre].abs(),
                                     color_continuous_scale='Reds')
            fig_treemap.update_layout(margin = dict(t=50, l=25, r=25, b=25))
            st.plotly_chart(fig_treemap, use_container_width=True)

        if df_variacion_er is not None and not df_variacion_er.empty and periodo_previo:
            st.markdown(f"#### Comparativo de Gastos vs. {periodo_previo}")
            df_gas_var = df_variacion_er[df_variacion_er[cuenta_col].astype(str).str.startswith('5')]
            st.bar_chart(data=df_gas_var.set_index('Descripción')[['Valor_actual', 'Valor_previo']])
            format_dict = {'Valor_previo': '${:,.0f}', 'Valor_actual': '${:,.0f}', 'Variacion_Absoluta': '${:,.0f}'}
            st.dataframe(df_gas_var[['Descripción', 'Valor_previo', 'Valor_actual', 'Variacion_Absoluta']].style.format(format_dict), use_container_width=True)

    with tab_roe:
        st.subheader("🎯 Análisis de Rentabilidad (ROE) con Modelo DuPont")
        kpis_actuales = kpis_por_tienda.get(cc_filter, {})

        if data_previa and periodo_previo:
            kpis_previos = data_previa['kpis'].get(cc_filter, {})
            dupont_data = {
                'Componente': ['Margen Neto', 'Rotación de Activos', 'Apalancamiento Financiero', 'ROE'],
                selected_view: [kpis_actuales.get(k, 0) for k in ['margen_neto', 'rotacion_activos', 'apalancamiento', 'roe']],
                periodo_previo: [kpis_previos.get(k, 0) for k in ['margen_neto', 'rotacion_activos', 'apalancamiento', 'roe']]
            }
            df_dupont = pd.DataFrame(dupont_data)
            df_dupont['Variación'] = df_dupont[selected_view] - df_dupont[periodo_previo]
            st.markdown("El **Análisis DuPont** descompone el ROE en tres palancas: eficiencia operativa (Margen Neto), eficiencia en el uso de activos (Rotación) y apalancamiento financiero.")
            st.dataframe(
                df_dupont.style.format({selected_view: '{:.2%}', periodo_previo: '{:.2%}', 'Variación': '{:+.2%}'})
                .background_gradient(cmap='RdYlGn', subset=['Variación'], low=0.4, high=0.4),
                use_container_width=True
            )
        else:
            st.info("Se requiere un periodo/año anterior para el análisis DuPont comparativo.")

    with tab_rep:
        st.subheader("📊 Reportes Financieros Detallados")

        st.markdown("#### Estado de Resultados")
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
                    er_conf['CUENTA'],
                    COL_CONFIG['BALANCE_GENERAL']['SALDO_FINAL'],
                    COL_CONFIG['BALANCE_GENERAL']['CUENTA']
                )
                st.dataframe(df_flujo.style.format({'Valor': '${:,.0f}'}), use_container_width=True)
        else:
            st.info("Se requiere un periodo/año anterior para generar el Estado de Flujo de Caja.")

    if search_account_input:
        st.markdown("---")
        with st.expander(f"Resultado de la búsqueda para cuentas que inician con '{search_account_input}'", expanded=True):
            cuenta_col_er = er_conf.get('CUENTA', 'Cuenta')
            cuenta_col_bg = COL_CONFIG['BALANCE_GENERAL'].get('CUENTA', 'Cuenta')

            st.write("**Estado de Resultados**")
            df_search_er = df_er_actual[df_er_actual[cuenta_col_er].astype(str).str.startswith(search_account_input)]
            st.dataframe(df_search_er if not df_search_er.empty else "No se encontraron cuentas en el ER.")

            st.write("**Balance General**")
            df_search_bg = df_bg_actual[df_bg_actual[cuenta_col_bg].astype(str).str.startswith(search_account_input)]
            st.dataframe(df_search_bg if not df_search_bg.empty else "No se encontraron cuentas en el BG.")

    st.sidebar.markdown("---")
    excel_buffer_profesional = generar_excel_gerencial_profesional(
        df_er_master=df_er_actual,
        df_bg_master=df_bg_actual,
        datos_periodo=data_actual,
        periodo_actual_str=str(selected_view),
        nivel_detalle_er=nivel_seleccionado
    )
    
    file_name_prefix = "Reporte_Gerencial_Anual" if is_annual_view else "Reporte_Gerencial_Mensual"
    
    st.sidebar.download_button(
        label=f"📥 Descargar Reporte Gerencial",
        data=excel_buffer_profesional,
        file_name=f"{file_name_prefix}_{selected_view.replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        type="primary"
    )
