# Dashboard_Principal.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

# --- Importamos nuestros módulos ---
# (Asegúrate de que estos módulos estén en el mismo directorio o en el PYTHONPATH)
from mi_logica_original import procesar_archivo_excel, generate_financial_statement, to_excel_buffer, COL_CONFIG
from dropbox_connector import get_dropbox_client, find_financial_files, load_excel_from_dropbox
from kpis_y_analisis import calcular_kpis_periodo, preparar_datos_tendencia, generar_analisis_avanzado_ia, generar_analisis_tendencia_ia

# ==============================================================================
#   NOTACIÓN CONTABLE IMPORTANTE PARA EL ANÁLISIS
# ==============================================================================
# En todo el análisis se asume la siguiente convención para el Estado de Resultados:
# - INGRESOS y UTILIDADES se representan con valores NEGATIVOS (favorable).
# - GASTOS y PÉRDIDAS se representan con valores POSITIVOS (desfavorable).
# Las funciones de análisis y visualización están diseñadas para interpretar esta lógica.

# ==============================================================================
#             FUNCIONES AUXILIARES DE ANÁLISIS Y VISUALIZACIÓN
# ==============================================================================

def plot_sparkline(data: pd.Series, title: str, is_percent: bool = False, lower_is_better: bool = False):
    """
    Crea un minigráfico de línea mejorado para KPIs.
    - lower_is_better: True para métricas como Endeudamiento, donde un valor menor es mejor.
    """
    if data.empty or len(data.dropna()) < 2:
        return go.Figure().update_layout(width=150, height=50, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', annotations=[dict(text="N/A", showarrow=False)])

    last_val = data.iloc[-1]
    first_val = data.iloc[0]
    
    # Lógica de color mejorada
    if (lower_is_better and last_val < first_val) or (not lower_is_better and last_val > first_val):
        color = '#28a745'  # Verde (Mejora)
    else:
        color = '#dc3545'  # Rojo (Empeora)

    fig = go.Figure(go.Scatter(
        x=list(range(len(data))),
        y=data,
        mode='lines',
        line=dict(color=color, width=2.5),
        fill='tozeroy',
        fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)"
    ))
    fig.update_layout(
        width=150, height=50,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=5, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig

def calcular_variaciones_er(df_actual: pd.DataFrame, df_previo: pd.DataFrame, cc_filter: str) -> pd.DataFrame:
    """
    Calcula las variaciones absolutas y porcentuales entre dos periodos para el Estado de Resultados.
    VERSIÓN CORREGIDA para manejar centros de costo que no existen en el periodo previo.
    """
    cuenta_col = COL_CONFIG['ESTADO_DE_RESULTADOS'].get('CUENTA', 'Cuenta')
    desc_col = COL_CONFIG['ESTADO_DE_RESULTADOS'].get('DESCRIPCION_CUENTA', 'Descripción')
    
    # Determinar el nombre de la columna de valor según el filtro
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
    
    utilidad_neta_actual = df_variacion['Valor_actual'].sum()
    utilidad_neta_previa = df_variacion['Valor_previo'].sum()

    variacion_ingresos = df_variacion[df_variacion[cuenta_col].str.startswith('4')]['Variacion_Absoluta'].sum()
    variacion_costos = df_variacion[df_variacion[cuenta_col].str.startswith('6')]['Variacion_Absoluta'].sum()
    variacion_gastos = df_variacion[df_variacion[cuenta_col].str.startswith('5')]['Variacion_Absoluta'].sum()
    
    otras_variaciones = df_variacion['Variacion_Absoluta'].sum() - (variacion_ingresos + variacion_costos + variacion_gastos)

    medidas = ["relative"] * 4
    textos = [f"${v:,.0f}" for v in [variacion_ingresos, variacion_costos, variacion_gastos, otras_variaciones]]

    fig = go.Figure(go.Waterfall(
        name="Variación",
        orientation="v",
        measure=["absolute"] + medidas + ["total"],
        x=["Utilidad Neta " + periodo_previo, "Ingresos", "Costos", "Gastos Op.", "Otros", "Utilidad Neta " + periodo_actual],
        text= [""] + textos + [""],
        y=[utilidad_neta_previa, variacion_ingresos, variacion_costos, variacion_gastos, otras_variaciones, utilidad_neta_actual],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#28a745"}},  # Favorable (disminuye el valor numérico)
        increasing={"marker": {"color": "#dc3545"}},  # Desfavorable (aumenta el valor numérico)
    ))

    fig.update_layout(
        title=f"Puente de Utilidad Neta: {periodo_previo} vs {periodo_actual}",
        showlegend=False,
        yaxis_title="Monto (COP)",
        height=500
    )
    fig.update_yaxes(tickformat="$,.0f")
    return fig

# ==============================================================================
#           CONFIGURACIÓN DE PÁGINA Y AUTENTICACIÓN
# ==============================================================================
st.set_page_config(layout="wide", page_title="Análisis Financiero Inteligente PRO")
st.title("🤖 Dashboard Financiero Profesional con IA")

st.markdown("""
<style>
    .reportview-container { background: #f0f2f6; }
    .kpi-card {
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        background-color: white;
        text-align: center;
    }
    .ai-analysis-text {
        background-color: #e8f0fe;
        border-left: 5px solid #1967d2;
        padding: 15px;
        border-radius: 5px;
        font-size: 1.05em;
    }
</style>
""", unsafe_allow_html=True)

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
#           CARGA DE DATOS AUTOMÁTICA DESDE DROPBOX
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
period_options = ["Análisis de Evolución (Tendencias)"] + sorted_periods
selected_view = st.sidebar.selectbox("Selecciona la vista de análisis:", period_options)

# ==============================================================================
#           VISTA DE ANÁLISIS DE TENDENCIAS (Mejorada)
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
    # Recordar: Ingresos son negativos, por eso usamos .abs() para visualizarlos como barras positivas.
    fig_combinada.add_trace(go.Bar(x=df_tendencia['periodo'], y=df_tendencia['ingresos'].abs(), name='Ingresos', marker_color='#28a745'))
    fig_combinada.add_trace(go.Bar(x=df_tendencia['periodo'], y=df_tendencia['gastos_operativos'].abs(), name='Gastos Operativos', marker_color='#ffc107'))
    fig_combinada.add_trace(go.Scatter(x=df_tendencia['periodo'], y=df_tendencia['utilidad_neta'].abs(), name='Utilidad Neta', mode='lines+markers', line=dict(color='#0d6efd', width=4)))
    fig_combinada.update_layout(
        title='Evolución de Ingresos, Gastos y Utilidad Neta (Valores Absolutos)',
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Monto (COP)"
    )
    st.plotly_chart(fig_combinada, use_container_width=True)

# ==============================================================================
#           VISTA DE PERIODO ÚNICO (CENTRO DE ANÁLISIS PROFUNDO)
# ==============================================================================
else:
    st.header(f"Centro de Análisis para el Periodo: {selected_view}")
    
    # --- Preparación de Datos ---
    data_actual = st.session_state.datos_historicos.get(selected_view)
    if not data_actual:
        st.error(f"No se encontraron datos para el periodo: {selected_view}"); st.stop()

    periodo_actual_idx = sorted_periods.index(selected_view)
    periodo_previo = sorted_periods[periodo_actual_idx + 1] if periodo_actual_idx + 1 < len(sorted_periods) else None
    data_previa = st.session_state.datos_historicos.get(periodo_previo) if periodo_previo else None

    df_er_actual = data_actual['df_er_master']
    df_bg_actual = data_actual['df_bg_master']
    kpis_por_tienda = data_actual['kpis']

    # --- Filtros en Sidebar ---
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
    
    # --- Cálculo de Variaciones ---
    df_variacion_er = None
    if data_previa:
        df_variacion_er = calcular_variaciones_er(df_er_actual, data_previa['df_er_master'], cc_filter)
        if df_variacion_er is not None and not df_variacion_er.empty:
            st.info(f"Análisis comparativo contra el periodo **{periodo_previo}**.")
    else:
        st.warning("No hay un periodo anterior para realizar análisis comparativo.")

    # --- Pestañas de Análisis Detallado ---
    tab_gen, tab_utilidad, tab_ing, tab_gas, tab_roe, tab_rep = st.tabs([
        "📊 Resumen General", 
        "💰 Análisis de Utilidad Neta", 
        "📈 Análisis de Ingresos", 
        "🧾 Análisis de Gastos", 
        "🎯 Análisis ROE (DuPont)", 
        "📋 Reportes Financieros"
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
        # --- CÓDIGO CORREGIDO PARA LA LLAMADA A LA IA ---
        with st.expander("🧠 **Ver Análisis y Consejos del CFO Virtual (IA)**", expanded=True):
            with st.spinner('El CFO Virtual está preparando un análisis profundo...'):
                # Preparamos un contexto mucho más rico para la IA
                contexto_ia = {
                    "kpis": selected_kpis,
                    "periodo": selected_view,
                    "centro_costo": cc_filter,
                    "convencion_contable": "IMPORTANTE: En el Estado de Resultados, los valores NEGATIVOS como ingresos son FAVORABLES. Los valores POSITIVOS como gastos son DESFAVORABLES. Una disminución en un gasto es una mejora.",
                    "variaciones_favorables": [],
                    "variaciones_desfavorables": []
                }
                if df_variacion_er is not None and not df_variacion_er.empty:
                    # Variación < 0 es Favorable (Crecimiento de ingresos o disminución de gastos)
                    top_favorables = df_variacion_er.nsmallest(5, 'Variacion_Absoluta')
                    # Variación > 0 es Desfavorable (Disminución de ingresos o aumento de gastos)
                    top_desfavorables = df_variacion_er.nlargest(5, 'Variacion_Absoluta')
                    contexto_ia["variaciones_favorables"] = top_favorables[['Descripción', 'Variacion_Absoluta']].to_dict('records')
                    contexto_ia["variaciones_desfavorables"] = top_desfavorables[['Descripción', 'Variacion_Absoluta']].to_dict('records')

                # Llamada a la IA con el diccionario de contexto único
                analisis_ia = generar_analisis_avanzado_ia(contexto_ia)
                st.markdown(f"<div class='ai-analysis-text'>{analisis_ia}</div>", unsafe_allow_html=True)

    with tab_utilidad:
        st.subheader(f"💰 Análisis de la Utilidad Neta: ¿Qué movió el resultado?")
        if df_variacion_er is not None and not df_variacion_er.empty:
            st.plotly_chart(plot_waterfall_utilidad_neta(df_variacion_er, selected_view, periodo_previo), use_container_width=True)
            
            st.markdown("#### Principales Motores del Cambio vs. Periodo Anterior")
            col1, col2 = st.columns(2)
            
            top_favorables = df_variacion_er[df_variacion_er['Variacion_Absoluta'] < 0].sort_values('Variacion_Absoluta').head(10)
            top_favorables = top_favorables[['Descripción', 'Valor_previo', 'Valor_actual', 'Variacion_Absoluta']]
            
            top_desfavorables = df_variacion_er[df_variacion_er['Variacion_Absoluta'] > 0].sort_values('Variacion_Absoluta', ascending=False).head(10)
            top_desfavorables = top_desfavorables[['Descripción', 'Valor_previo', 'Valor_actual', 'Variacion_Absoluta']]

            with col1:
                st.markdown("✅ **Impactos Positivos (Ayudaron a la Utilidad)**")
                st.dataframe(top_favorables.style.format('${:,.0f}').background_gradient(cmap='Greens', subset=['Variacion_Absoluta']), use_container_width=True)
            with col2:
                st.markdown("❌ **Impactos Negativos (Perjudicaron la Utilidad)**")
                st.dataframe(top_desfavorables.style.format('${:,.0f}').background_gradient(cmap='Reds', subset=['Variacion_Absoluta']), use_container_width=True)
        else:
            st.info("Se requiere un periodo anterior para este análisis.")

    with tab_ing:
        st.subheader("📈 Análisis Detallado de Ingresos")
        cuenta_col = er_conf.get('CUENTA', 'Cuenta')
        valor_col_nombre = 'Total_Consolidado_ER' if cc_filter == 'Todos' else cc_filter
        
        if df_variacion_er is not None and not df_variacion_er.empty:
            df_ing_var = df_variacion_er[df_variacion_er[cuenta_col].str.startswith('4')]
            st.markdown("##### Comparativo de Ingresos vs. Periodo Anterior")
            st.bar_chart(data=df_ing_var.set_index('Descripción')[['Valor_actual', 'Valor_previo']].abs())
            st.dataframe(df_ing_var[['Descripción', 'Valor_previo', 'Valor_actual', 'Variacion_Absoluta']].style.format('${:,.0f}'), use_container_width=True)
        else:
            df_ingresos = df_er_actual[df_er_actual[cuenta_col].str.startswith('4')]
            if valor_col_nombre in df_ingresos.columns:
                 st.bar_chart(data=df_ingresos.set_index('Descripción')[valor_col_nombre].abs())
                 st.dataframe(df_ingresos[['Descripción', valor_col_nombre]], use_container_width=True)
        
    with tab_gas:
        st.subheader("🧾 Análisis Detallado de Gastos")
        cuenta_col = er_conf.get('CUENTA', 'Cuenta')
        valor_col_nombre = 'Total_Consolidado_ER' if cc_filter == 'Todos' else cc_filter

        df_gastos = df_er_actual[df_er_actual[cuenta_col].str.startswith('5')]
        
        if valor_col_nombre in df_gastos.columns:
            st.markdown("#### Composición de Gastos del Periodo")
            fig_treemap = px.treemap(df_gastos, path=['Descripción'], values=valor_col_nombre,
                                     title='Distribución de Gastos Operacionales',
                                     color=valor_col_nombre,
                                     color_continuous_scale='Reds')
            st.plotly_chart(fig_treemap, use_container_width=True)

        if df_variacion_er is not None and not df_variacion_er.empty:
            st.markdown("#### Comparativo de Gastos vs. Periodo Anterior")
            df_gas_var = df_variacion_er[df_variacion_er[cuenta_col].str.startswith('5')]
            st.bar_chart(data=df_gas_var.set_index('Descripción')[['Valor_actual', 'Valor_previo']])
            st.dataframe(df_gas_var[['Descripción', 'Valor_previo', 'Valor_actual', 'Variacion_Absoluta']].style.format('${:,.0f}'), use_container_width=True)

    with tab_roe:
        st.subheader("🎯 Análisis de Rentabilidad (ROE) con Modelo DuPont")
        kpis_actuales = kpis_por_tienda.get(cc_filter, {})
        
        if data_previa:
            kpis_previos = data_previa['kpis'].get(cc_filter, {})
            
            dupont_data = {
                'Componente': ['Margen Neto', 'Rotación de Activos', 'Apalancamiento Financiero', 'ROE'],
                periodo_actual: [
                    kpis_actuales.get('margen_neto', 0),
                    kpis_actuales.get('rotacion_activos', 0),
                    kpis_actuales.get('apalancamiento', 0),
                    kpis_actuales.get('roe', 0)
                ],
                periodo_previo: [
                    kpis_previos.get('margen_neto', 0),
                    kpis_previos.get('rotacion_activos', 0),
                    kpis_previos.get('apalancamiento', 0),
                    kpis_previos.get('roe', 0)
                ]
            }
            df_dupont = pd.DataFrame(dupont_data)
            df_dupont['Variación'] = df_dupont[periodo_actual] - df_dupont[periodo_previo]
            
            st.markdown("El **Análisis DuPont** descompone el ROE en tres palancas: eficiencia operativa (Margen Neto), eficiencia en el uso de activos (Rotación) y apalancamiento financiero. Permite identificar qué motor de la rentabilidad ha cambiado.")
            
            st.dataframe(df_dupont.style.format({
                periodo_actual: '{:.2%}',
                periodo_previo: '{:.2%}',
                'Variación': '{:+.2%}',
            }).background_gradient(cmap='RdYlGn', subset=['Variación'], low=0.4, high=0.4), use_container_width=True)

        else:
            st.info("Se requiere un periodo anterior para el análisis DuPont comparativo.")

    with tab_rep:
        st.subheader("📊 Reportes Financieros Detallados")
        
        st.markdown("#### Estado de Resultados")
        df_er_display = generate_financial_statement(df_er_actual, 'Estado de Resultados', cc_filter, nivel_seleccionado)
        st.dataframe(df_er_display.style.format({'Valor': "${:,.0f}"}), use_container_width=True, height=600)

        st.markdown("#### Balance General")
        df_bg_display = generate_financial_statement(df_bg_actual, 'Balance General', 99)
        st.dataframe(df_bg_display.style.format({'Valor': "${:,.0f}"}), use_container_width=True, height=600)

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
                st.info(f"No se encontraron cuentas en el ER para '{search_account_input}'.")
            st.write("**Balance General**")
            df_search_bg = df_bg_actual[df_bg_actual[cuenta_col_bg].astype(str).str.startswith(search_account_input)]
            if not df_search_bg.empty:
                st.dataframe(df_search_bg)
            else:
                st.info(f"No se encontraron cuentas en el BG para '{search_account_input}'.")

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
