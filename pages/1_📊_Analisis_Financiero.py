# pages/1_📊_Analisis_Financiero.py
import streamlit as st
import pandas as pd
import plotly.express as px
from analisis_adicional import calcular_analisis_vertical, calcular_analisis_horizontal # Importamos las nuevas funciones
from mi_logica_original import COL_CONFIG, generate_financial_statement
from kpis_y_analisis import generar_analisis_tendencia_ia # Reutilizamos la IA de tendencias

st.set_page_config(layout="wide", page_title="Análisis Financiero Detallado")
st.title("🔬 Análisis Financiero Detallado")

# --- Verificación de autenticación y carga de datos ---
if 'authenticated' not in st.session_state or not st.session_state.authenticated:
    st.warning("Por favor, autentícate en la página principal para continuar.")
    st.stop()
if 'datos_historicos' not in st.session_state or not st.session_state.datos_historicos:
    st.error("No se han cargado datos históricos. Ve a la página principal y refresca los datos.")
    st.stop()

datos_historicos = st.session_state.datos_historicos
sorted_periods = sorted(datos_historicos.keys(), reverse=True)

# --- Pestañas de Análisis ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Análisis Vertical y Horizontal", 
    "🏆 Mejor Periodo del Año (IA)",
    "💰 Análisis de Rentabilidad (DuPont)",
    "🚑 Test Ácido y Ciclo de Efectivo"
])

# ==============================================================================
#                      PESTAÑA 1: ANÁLISIS VERTICAL Y HORIZONTAL
# ==============================================================================
with tab1:
    st.header("Análisis Vertical y Horizontal")
    st.markdown("Compara la estructura interna de los estados financieros (Vertical) y su evolución entre periodos (Horizontal).")

    col1, col2 = st.columns(2)
    with col1:
        periodo_actual_vh = st.selectbox("Selecciona el Periodo Principal:", sorted_periods, key="periodo_vh_actual")
    with col2:
        periodos_anteriores = [p for p in sorted_periods if p < periodo_actual_vh]
        periodo_anterior_vh = st.selectbox(
            "Selecciona el Periodo para Comparar (Horizontal):", 
            periodos_anteriores, 
            key="periodo_vh_anterior",
            disabled=not periodos_anteriores
        )

    # --- Carga de datos para los periodos seleccionados ---
    data_actual = datos_historicos[periodo_actual_vh]
    df_er_actual = data_actual['df_er_master']
    df_bg_actual = data_actual['df_bg_master']
    
    data_anterior = datos_historicos.get(periodo_anterior_vh) if periodo_anterior_vh else None
    df_er_anterior = data_anterior['df_er_master'] if data_anterior else pd.DataFrame()
    df_bg_anterior = data_anterior['df_bg_master'] if data_anterior else pd.DataFrame()

    er_conf = COL_CONFIG['ESTADO_DE_RESULTADOS']
    bg_conf = COL_CONFIG['BALANCE_GENERAL']
    
    # --- Análisis del Estado de Resultados ---
    st.subheader("Estado de Resultados")
    expander_er = st.expander("Ver Análisis Detallado de ER", expanded=True)
    with expander_er:
        # Vertical
        df_er_vf = generate_financial_statement(df_er_actual, 'Estado de Resultados', 'Todos', 99)
        df_er_vertical = calcular_analisis_vertical(df_er_vf, 'Valor', er_conf['CUENTA'], '4')
        st.write("**Análisis Vertical (Periodo Actual)**")
        st.dataframe(df_er_vertical.style.format({'Valor': "${:,.0f}", 'Análisis Vertical (%)': "{:.2f}%"}), use_container_width=True)

        # Horizontal
        if not df_er_anterior.empty:
            df_er_hf = generate_financial_statement(df_er_anterior, 'Estado de Resultados', 'Todos', 99)
            df_er_horizontal = calcular_analisis_horizontal(df_er_vf, df_er_hf, 'Valor', er_conf['CUENTA'])
            st.write("**Análisis Horizontal (Comparativo)**")
            st.dataframe(df_er_horizontal.style.format({
                'Valor Actual': "${:,.0f}", 'Valor Anterior': "${:,.0f}", 
                'Variación Absoluta': "${:,.0f}", 'Variación Relativa (%)': "{:.2f}%"
            }), use_container_width=True)

    # --- Análisis del Balance General ---
    st.subheader("Balance General")
    expander_bg = st.expander("Ver Análisis Detallado de BG", expanded=True)
    with expander_bg:
        # Vertical
        df_bg_vf = generate_financial_statement(df_bg_actual, 'Balance General', 'Todos', 99)
        df_bg_vertical = calcular_analisis_vertical(df_bg_vf, 'Valor', bg_conf['CUENTA'], '1')
        st.write("**Análisis Vertical (Periodo Actual)**")
        st.dataframe(df_bg_vertical.style.format({'Valor': "${:,.0f}", 'Análisis Vertical (%)': "{:.2f}%"}), use_container_width=True)

        # Horizontal
        if not df_bg_anterior.empty:
            df_bg_hf = generate_financial_statement(df_bg_anterior, 'Balance General', 'Todos', 99)
            df_bg_horizontal = calcular_analisis_horizontal(df_bg_vf, df_bg_hf, 'Valor', bg_conf['CUENTA'])
            st.write("**Análisis Horizontal (Comparativo)**")
            st.dataframe(df_bg_horizontal.style.format({
                'Valor Actual': "${:,.0f}", 'Valor Anterior': "${:,.0f}",
                'Variación Absoluta': "${:,.0f}", 'Variación Relativa (%)': "{:.2f}%"
            }), use_container_width=True)

# ==============================================================================
#                      PESTAÑA 2: MEJOR PERIODO DEL AÑO (IA)
# ==============================================================================
with tab2:
    st.header("🏆 Identificación del Mejor Periodo del Año con IA")
    st.markdown("La IA analiza toda la historia financiera para determinar cuál fue el periodo más exitoso y explica las razones clave de dicho éxito.")

    if len(datos_historicos) < 2:
        st.info("Se necesitan al menos dos periodos para realizar una comparación.")
    else:
        df_tendencia = pd.DataFrame([
            {'periodo': p, **d['kpis']['Todos']} for p, d in datos_historicos.items()
        ]).sort_values('periodo').reset_index(drop=True)

        # Criterios para definir el "mejor" periodo (puedes ajustarlos)
        df_tendencia['score'] = (
            df_tendencia['margen_neto'].rank(pct=True) * 0.4 +
            df_tendencia['roe'].rank(pct=True) * 0.3 +
            df_tendencia['ingresos'].rank(pct=True) * 0.2 +
            df_tendencia['razon_corriente'].rank(pct=True) * 0.1
        )
        mejor_periodo_row = df_tendencia.loc[df_tendencia['score'].idxmax()]
        mejor_periodo_key = pd.to_datetime(mejor_periodo_row['periodo']).strftime('%Y-%m')

        st.success(f"**El mejor periodo identificado es: {mejor_periodo_key}**")

        with st.spinner("Generando análisis detallado del porqué..."):
            # Reutilizamos la función de análisis de tendencia pero con un prompt enfocado
            prompt_mejor_periodo = f"""
            **Rol:** Eres un Analista Financiero Senior que debe explicar a la junta directiva por qué el mes de {mejor_periodo_key} fue el mejor del año.
            
            **Contexto:** Se ha determinado que {mejor_periodo_key} fue el periodo de mayor rendimiento financiero basándose en una combinación de rentabilidad, crecimiento y solidez.
            
            **Datos del Mejor Periodo ({mejor_periodo_key}):**
            - Ingresos: ${mejor_periodo_row['ingresos']:,.0f}
            - Utilidad Neta: ${mejor_periodo_row['utilidad_neta']:,.0f}
            - Margen Neto: {mejor_periodo_row['margen_neto']:.2%}
            - ROE: {mejor_periodo_row['roe']:.2%}
            - Razón Corriente: {mejor_periodo_row['razon_corriente']:.2f}

            **Instrucciones:**
            1.  **Título Atractivo:** Comienza con un título como "🏆 {mejor_periodo_key}: Anatomía de un Mes Exitoso".
            2.  **Diagnóstico Ejecutivo:** Explica en un párrafo por qué este mes fue excepcional.
            3.  **Factores Clave del Éxito:** Usa una lista con viñetas y emojis para detallar los 3 principales factores que llevaron a este resultado (Ej: ✅ **Rentabilidad Superior:**, 📈 **Crecimiento Sólido:**, 💧 **Liquidez Robusta:**).
            4.  **Lecciones Aprendidas:** Concluye con 2 lecciones estratégicas que la empresa puede aprender y replicar de este periodo.
            """
            
            # Aquí usamos una llamada directa a la IA similar a tus funciones existentes
            # Nota: Esto es un ejemplo, necesitarías integrar la llamada a `genai` como en `kpis_y_analisis.py`
            # análisis_ia = generar_analisis_con_prompt_especifico(prompt_mejor_periodo)
            # Por ahora, mostraremos un texto de ejemplo:
            
            analisis_ia = f"""
            ### 🏆 {mejor_periodo_key}: Anatomía de un Mes Exitoso

            El mes de **{mejor_periodo_key}** se destaca como el pináculo del rendimiento financiero del año, no solo por un crecimiento notable en los ingresos, sino por una excepcional capacidad de convertir ese crecimiento en rentabilidad real y sostenible, manteniendo una salud financiera envidiable.

            #### Factores Clave del Éxito 🔑

            * **✅ Rentabilidad Máxima:** Se alcanzó un Margen Neto de **{mejor_periodo_row['margen_neto']:.2%}**, lo que indica una gestión de costos y gastos extraordinariamente eficiente durante este periodo. Cada peso de venta generó más utilidad que en cualquier otro mes.
            * **📈 Crecimiento con Calidad:** Los ingresos de **${mejor_periodo_row['ingresos']:,.0f}** no solo fueron altos, sino que se tradujeron directamente en un ROE del **{mejor_periodo_row['roe']:.2%}**, demostrando que la inversión de los accionistas fue altamente productiva.
            * **💧 Solidez Financiera:** Con una Razón Corriente de **{mejor_periodo_row['razon_corriente']:.2f}**, la empresa demostró una capacidad sobresaliente para cubrir sus obligaciones a corto plazo, otorgando una gran tranquilidad y flexibilidad operativa.

            #### Lecciones para Replicar 💡

            1.  **Control de Gastos:** Investigar las iniciativas de control de costos implementadas en {mejor_periodo_key} para convertirlas en políticas permanentes.
            2.  **Mix de Ventas:** Analizar el mix de productos/servicios vendidos ese mes que pudo haber contribuido a mayores márgenes y potenciarlo en el futuro.
            """
            st.markdown(analisis_ia, unsafe_allow_html=True)


# ==============================================================================
#                      PESTAÑA 3: Sugerencia - Análisis DuPont
# ==============================================================================
with tab3:
    st.header("💰 Análisis de Rentabilidad (DuPont)")
    st.markdown("""
    El análisis DuPont descompone el **Retorno sobre el Patrimonio (ROE)** en tres componentes clave para entender qué impulsa la rentabilidad:
    1.  **Margen Neto:** Mide la eficiencia operativa.
    2.  **Rotación de Activos:** Mide la eficiencia en el uso de los activos.
    3.  **Apalancamiento Financiero:** Mide cómo el endeudamiento aumenta la rentabilidad.
    
    **Fórmula:** `ROE = Margen Neto * Rotación de Activos * Apalancamiento Financiero`
    """)

    periodo_dupont = st.selectbox("Selecciona el Periodo para el Análisis DuPont:", sorted_periods, key="periodo_dupont")
    kpis = datos_historicos[periodo_dupont]['kpis']['Todos']
    
    activo_total = kpis['endeudamiento_activo'] and kpis['ingresos'] / kpis['endeudamiento_activo']
    if kpis and activo_total:
        rotacion_activos = kpis['ingresos'] / activo_total if activo_total else 0
        apalancamiento = activo_total / (activo_total - (kpis.get('pasivo_total', activo_total * kpis['endeudamiento_activo']))) if activo_total and kpis.get('patrimonio') else 0

        st.subheader(f"Descomposición del ROE para {periodo_dupont}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ROE (Resultado)", f"{kpis.get('roe', 0):.2%}")
        col2.metric("Margen Neto", f"{kpis.get('margen_neto', 0):.2%}")
        col3.metric("Rotación de Activos", f"{rotacion_activos:.2f}")
        col4.metric("Apalancamiento", f"{apalancamiento:.2f}")
        
        st.info("Un ROE alto es sostenible si proviene de un buen margen y alta rotación. Si depende mucho del apalancamiento, el riesgo es mayor.")

# ==============================================================================
#                      PESTAÑA 4: Sugerencia - Salud Financiera
# ==============================================================================
with tab4:
    st.header("🚑 Test Ácido y Ciclo de Conversión de Efectivo")
    st.markdown("Indicadores avanzados para medir la liquidez inmediata y la eficiencia del capital de trabajo.")

    periodo_salud = st.selectbox("Selecciona el Periodo para el Análisis de Salud:", sorted_periods, key="periodo_salud")
    kpis_salud = datos_historicos[periodo_salud]['kpis']['Todos']

    if kpis_salud:
        bg_actual = datos_historicos[periodo_salud]['df_bg_master']
        saldo_final_col = COL_CONFIG['BALANCE_GENERAL']['SALDO_FINAL']
        cuenta_bg_col = COL_CONFIG['BALANCE_GENERAL']['CUENTA']
        
        activo_corriente = kpis_salud.get('activo_corriente', get_principal_account_value(bg_actual, '1', saldo_final_col, cuenta_bg_col))
        inventario = kpis_salud.get('inventarios', get_principal_account_value(bg_actual, '14', saldo_final_col, cuenta_bg_col))
        pasivo_corriente = kpis_salud.get('pasivo_corriente', get_principal_account_value(bg_actual, '2', saldo_final_col, cuenta_bg_col))
        
        test_acido = (activo_corriente - inventario) / pasivo_corriente if pasivo_corriente else 0
        
        st.subheader("Indicadores de Liquidez Inmediata")
        col1, col2 = st.columns(2)
        col1.metric("Razón Corriente", f"{kpis_salud.get('razon_corriente', 0):.2f}")
        col2.metric("Prueba Ácida (Quick Ratio)", f"{test_acido:.2f}", help="Mide la capacidad de pagar deudas a corto plazo sin depender de la venta de inventarios. Ideal > 1.")
        
        st.info("La **Prueba Ácida** es un indicador más estricto de la liquidez. Si es mucho menor que la Razón Corriente, significa que la empresa depende fuertemente de sus inventarios.")
