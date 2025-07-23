import streamlit as st
import pandas as pd
import plotly.express as px
from analisis_adicional import calcular_analisis_vertical, calcular_analisis_horizontal
from mi_logica_original import COL_CONFIG, generate_financial_statement
# Se importa la función correcta para análisis basados en prompts directos
from kpis_y_analisis import generar_analisis_con_prompt_libre

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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Análisis Vertical y Horizontal",
    "🏆 Mejor Periodo del Año (IA)",
    "💰 Análisis de Rentabilidad (DuPont)",
    "🚑 Test Ácido y Ciclo de Efectivo",
    "📈 Consolidado Histórico"
])

# ==============================================================================
#   PESTAÑA 1: ANÁLISIS VERTICAL Y HORIZONTAL (CON IA CORREGIDA)
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
    df_er_actual_full = data_actual['df_er_master']
    df_bg_actual_full = data_actual['df_bg_master']
    
    df_er_display_actual = generate_financial_statement(df_er_actual_full, 'Estado de Resultados', 'Todos', 99)
    df_bg_display_actual = generate_financial_statement(df_bg_actual_full, 'Balance General', 'Todos', 99)
    
    er_conf = COL_CONFIG['ESTADO_DE_RESULTADOS']
    bg_conf = COL_CONFIG['BALANCE_GENERAL']
    
    # --- IA PARA ANÁLISIS HORIZONTAL ---
    if periodo_anterior_vh:
        data_anterior = datos_historicos[periodo_anterior_vh]
        df_er_anterior_full = data_anterior['df_er_master']
        df_bg_anterior_full = data_anterior['df_bg_master']
        
        df_er_display_anterior = generate_financial_statement(df_er_anterior_full, 'Estado de Resultados', 'Todos', 99)
        
        # Usamos la columna 'Cuenta' del estado de resultados generado
        cuenta_col_display = 'Cuenta' 
        df_horizontal = calcular_analisis_horizontal(df_er_display_actual, df_er_display_anterior, 'Valor', cuenta_col_display)
        
        # Asegurarse de que la columna de variación sea numérica para ordenar
        df_horizontal['Variacion_Abs_Num'] = pd.to_numeric(df_horizontal['Variación Absoluta'], errors='coerce').fillna(0)
        
        # CORRECCIÓN LÓGICA: nsmallest son las MEJORAS (más negativas), nlargest son los DETERIOROS (más positivas)
        bottom_5_variaciones = df_horizontal.nsmallest(5, 'Variacion_Abs_Num')
        top_5_variaciones = df_horizontal.nlargest(5, 'Variacion_Abs_Num')

        prompt_horizontal = f"""
        **Rol:** Eres un Contralor Financiero experto. Analizas la comparación entre dos periodos ({periodo_actual_vh} vs {periodo_anterior_vh}) para un comité de gerencia. Tu lenguaje debe ser profesional, claro y directo.

        **LÓGICA DE NEGOCIO CRÍTICA:**
        - Una **Variación Absoluta NEGATIVA** es una **MEJORA** o **IMPACTO POSITIVO** (ej: un ingreso que crece o un gasto que disminuye).
        - Una **Variación Absoluta POSITIVA** es un **DETERIORO** o **IMPACTO NEGATIVO** (ej: un ingreso que cae o un gasto que aumenta).
        - Tu análisis DEBE seguir esta lógica.

        **Contexto:** Se está evaluando la evolución del negocio. Aquí están las variaciones más significativas en el Estado de Resultados:
        - **Mayores Mejoras (Impacto Positivo):**
        {bottom_5_variaciones[[cuenta_col_display, 'Valor Actual', 'Valor Anterior', 'Variación Absoluta', 'Variación Relativa (%)']].to_string()}
        - **Mayores Deterioros (Impacto Negativo):**
        {top_5_variaciones[[cuenta_col_display, 'Valor Actual', 'Valor Anterior', 'Variación Absoluta', 'Variación Relativa (%)']].to_string()}

        **Instrucciones:**
        1.  **Diagnóstico General (1 párrafo):** Ofrece un veredicto sobre la evolución. ¿La empresa mejoró o empeoró financieramente entre estos periodos? Sé directo y fundamenta tu respuesta en las cifras, usando la lógica de negocio descrita.
        2.  **Puntos Positivos a Destacar ✅:** Lista 2-3 puntos buenos de la comparación. Explica la implicación de negocio. (Ej: "La reducción del costo de ventas sugiere una mejora en las negociaciones con proveedores.").
        3.  **Focos Rojos de Alerta ⚠️:** Lista 2-3 puntos negativos. Explica el riesgo o problema. (Ej: "El aumento desproporcionado de gastos administrativos indica una posible pérdida de eficiencia.").
        4.  **Recomendación Estratégica 🎯:** Basado en el análisis, ofrece una recomendación clave, concreta y accionable.
        """
        
        with st.expander("🧠 Ver Análisis Comparativo por IA", expanded=True):
            with st.spinner("El Contralor IA está comparando los periodos..."):
                # LLAMADA CORREGIDA: Se usa la función que solo necesita el prompt
                analisis_comparativo = generar_analisis_con_prompt_libre(prompt_horizontal)
                st.markdown(analisis_comparativo, unsafe_allow_html=True)
                
    # --- Visualización de tablas ---
    st.subheader("Estado de Resultados")
    expander_er = st.expander("Ver Análisis Detallado de ER", expanded=True)
    with expander_er:
        df_er_vertical = calcular_analisis_vertical(df_er_display_actual, 'Valor', 'Cuenta', '4')
        st.write("**Análisis Vertical (Periodo Actual)**")
        st.dataframe(df_er_vertical.style.format({'Valor': "${:,.0f}", 'Análisis Vertical (%)': "{:.2f}%"}), use_container_width=True)

        if periodo_anterior_vh:
            df_er_horizontal = calcular_analisis_horizontal(df_er_display_actual, df_er_display_anterior, 'Valor', 'Cuenta')
            st.write("**Análisis Horizontal (Comparativo)**")
            st.dataframe(df_er_horizontal.style.format({
                'Valor Actual': "${:,.0f}", 'Valor Anterior': "${:,.0f}",
                'Variación Absoluta': "${:,.0f}", 'Variación Relativa (%)': "{:.2f}%"
            }), use_container_width=True)

    st.subheader("Balance General")
    expander_bg = st.expander("Ver Análisis Detallado de BG", expanded=True)
    with expander_bg:
        df_bg_vertical = calcular_analisis_vertical(df_bg_display_actual, 'Valor', 'Cuenta', '1')
        st.write("**Análisis Vertical (Periodo Actual)**")
        st.dataframe(df_bg_vertical.style.format({'Valor': "${:,.0f}", 'Análisis Vertical (%)': "{:.2f}%"}), use_container_width=True)

        if periodo_anterior_vh:
            df_bg_display_anterior = generate_financial_statement(df_bg_anterior_full, 'Balance General', 'Todos', 99)
            df_bg_horizontal = calcular_analisis_horizontal(df_bg_display_actual, df_bg_display_anterior, 'Valor', 'Cuenta')
            st.write("**Análisis Horizontal (Comparativo)**")
            st.dataframe(df_bg_horizontal.style.format({
                'Valor Actual': "${:,.0f}", 'Valor Anterior': "${:,.0f}",
                'Variación Absoluta': "${:,.0f}", 'Variación Relativa (%)': "{:.2f}%"
            }), use_container_width=True)


# ==============================================================================
#             PESTAÑA 2: MEJOR PERIODO DEL AÑO (CON IA CORREGIDA)
# ==============================================================================
with tab2:
    st.header("🏆 Identificación del Periodo de Mejor Desempeño Relativo")
    st.markdown("La IA analiza la historia para identificar el periodo que, aunque no sea perfecto, mostró las señales más prometedoras o el menor deterioro, y extrae lecciones clave.")

    if len(datos_historicos) < 2:
        st.info("Se necesitan al menos dos periodos para realizar una comparación.")
    else:
        df_tendencia = pd.DataFrame([
            {'periodo': p, **d['kpis']['Todos']} for p, d in datos_historicos.items()
        ]).sort_values('periodo').reset_index(drop=True)
        
        df_tendencia['score'] = (
            df_tendencia['margen_neto'].rank(pct=True) * 0.4 +
            df_tendencia['roe'].rank(pct=True) * 0.3 +
            abs(df_tendencia['ingresos']).rank(pct=True) * 0.2 + # Usamos abs para rankear el tamaño del ingreso
            df_tendencia['razon_corriente'].rank(pct=True) * 0.1
        )
        mejor_periodo_row = df_tendencia.loc[df_tendencia['score'].idxmax()]
        mejor_periodo_key = pd.to_datetime(mejor_periodo_row['periodo']).strftime('%Y-%m')

        st.info(f"**Periodo de mejor desempeño relativo identificado: {mejor_periodo_key}**")

        es_negativo = mejor_periodo_row['utilidad_neta'] > 0 or mejor_periodo_row['margen_neto'] < 0

        titulo_analisis = f"🏅 {mejor_periodo_key}: Anatomía de Nuestro Mejor Esfuerzo" if es_negativo else f"🏆 {mejor_periodo_key}: Anatomía de un Mes Exitoso"
        contexto_analisis = "Aunque los resultados generales siguen siendo un desafío, este fue el mes en que mostramos el desempeño más sólido y logramos contener mejor las pérdidas." if es_negativo else "Este mes representa un pináculo en el rendimiento financiero del año, demostrando un crecimiento y rentabilidad saludables."

        prompt_mejor_periodo_realista = f"""
        **Rol:** Eres un Asesor Financiero Estratégico (CFO) que se dirige al equipo directivo. Tu tono es realista, didáctico y orientado a la acción. Debes ser brutalmente honesto pero constructivo.

        **Lógica de Negocio Clave:** Recuerda que una utilidad neta negativa es una GANANCIA. Un margen neto positivo es bueno.

        **Contexto General:** {contexto_analisis}

        **Datos Clave del Periodo ({mejor_periodo_key}):**
        - Ingresos: ${mejor_periodo_row['ingresos']:,.0f}
        - Utilidad Neta (Negativo=Ganancia): ${mejor_periodo_row['utilidad_neta']:,.0f}
        - Margen Neto: {mejor_periodo_row['margen_neto']:.2%}
        - ROE: {mejor_periodo_row['roe']:.2%}
        - Razón Corriente: {mejor_periodo_row['razon_corriente']:.2f}

        **Instrucciones:**
        1.  **Título:** Usa este título: "{titulo_analisis}".
        2.  **Diagnóstico Sincero (1 párrafo):** Explica por qué este mes fue el 'mejor' en términos relativos. Si los números de utilidad son negativos (lo cual es bueno), explícalo como un éxito.
        3.  **Lecciones Clave del Periodo:**
            - Usa una lista con viñetas y emojis (🌱, 💡, ⚠️).
            - **Analiza cada KPI clave de forma crítica:**
                - **Rentabilidad:** "🌱 **Nivel de Rentabilidad:** Un margen de {mejor_periodo_row['margen_neto']:.2%} es nuestro punto de referencia. Debemos investigar qué impulsó este resultado para replicarlo".
                - **Crecimiento:** "💡 **Nivel de Ingresos:** Alcanzar ${mejor_periodo_row['ingresos']:,.0f} en ventas es un logro, pero la pregunta clave es: ¿a qué costo? Necesitamos que este nivel de ventas se traduzca en utilidad positiva."
                - **Solidez/Liquidez:** "⚠️ **Alerta de Liquidez:** Una Razón Corriente de {mejor_periodo_row['razon_corriente']:.2f} puede ser un punto de atención. Si es menor a 1.5, indica un riesgo que debemos monitorear de cerca."
        4.  **Plan de Acción Prioritario (2-3 puntos):** Ofrece recomendaciones concretas y urgentes basadas en las debilidades detectadas, incluso en el "mejor" mes.
        """

        with st.spinner("El Asesor Financiero IA está preparando un análisis realista..."):
            # LLAMADA CORREGIDA: Se usa la función que solo necesita el prompt
            analisis_ia_realista = generar_analisis_con_prompt_libre(prompt_mejor_periodo_realista)
            st.markdown(analisis_ia_realista, unsafe_allow_html=True)


# ==============================================================================
#             PESTAÑA 3: Sugerencia - Análisis DuPont
# ==============================================================================
with tab3:
    st.header("💰 Análisis de Rentabilidad (DuPont)")
    st.markdown("""
    El análisis DuPont descompone el **Retorno sobre el Patrimonio (ROE)** en tres componentes clave para entender qué impulsa la rentabilidad:
    1.  **Margen Neto:** Mide la eficiencia operativa (rentabilidad por venta).
    2.  **Rotación de Activos:** Mide la eficiencia en el uso de los activos para generar ventas.
    3.  **Apalancamiento Financiero:** Mide cómo el endeudamiento aumenta la rentabilidad de los accionistas.
    
    **Fórmula:** `ROE = Margen Neto * Rotación de Activos * Apalancamiento Financiero`
    """)

    periodo_dupont = st.selectbox("Selecciona el Periodo para el Análisis DuPont:", sorted_periods, key="periodo_dupont")
    kpis = datos_historicos[periodo_dupont]['kpis']['Todos']
    
    if kpis:
        st.subheader(f"Descomposición del ROE para {periodo_dupont}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ROE (Resultado)", f"{kpis.get('roe', 0):.2%}")
        col2.metric("Margen Neto", f"{kpis.get('margen_neto', 0):.2%}", help="Utilidad Neta / Ventas")
        col3.metric("Rotación de Activos", f"{kpis.get('rotacion_activos', 0):.2f}", help="Ventas / Activo Total")
        col4.metric("Apalancamiento", f"{kpis.get('apalancamiento', 0):.2f}", help="Activo Total / Patrimonio")
        
        st.info("Un ROE alto es más sostenible si proviene de un buen margen y una alta rotación. Si depende demasiado del apalancamiento, el riesgo financiero es mayor.")

# ==============================================================================
#             PESTAÑA 4: Sugerencia - Salud Financiera
# ==============================================================================
with tab4:
    st.header("🚑 Test Ácido y Ciclo de Conversión de Efectivo")
    st.markdown("Indicadores avanzados para medir la liquidez inmediata y la eficiencia del capital de trabajo.")

    periodo_salud = st.selectbox("Selecciona el Periodo para el Análisis de Salud:", sorted_periods, key="periodo_salud")
    kpis_salud = datos_historicos[periodo_salud]['kpis']['Todos']

    if kpis_salud:
        activo_corriente = kpis_salud.get('activo_corriente', 0)
        inventario = kpis_salud.get('inventarios', 0)
        pasivo_corriente = kpis_salud.get('pasivo_corriente', 0)
        
        test_acido = (activo_corriente - inventario) / pasivo_corriente if pasivo_corriente > 0 else 0
        
        st.subheader("Indicadores de Liquidez Inmediata")
        col1, col2 = st.columns(2)
        col1.metric("Razón Corriente", f"{kpis_salud.get('razon_corriente', 0):.2f}")
        col2.metric("Prueba Ácida (Quick Ratio)", f"{test_acido:.2f}", help="Mide la capacidad de pagar deudas a corto plazo sin depender de la venta de inventarios. Ideal > 1.")
        
        if test_acido < 1:
            st.warning(f"**Atención:** Un Test Ácido de {test_acido:.2f} es bajo. Indica que sin vender los inventarios, la empresa podría tener dificultades para cubrir sus obligaciones más urgentes. La dependencia del inventario es alta.")
        else:
            st.success(f"**Buena señal:** Un Test Ácido de {test_acido:.2f} es saludable. Muestra que la empresa tiene suficientes activos líquidos para operar con tranquilidad a corto plazo, incluso sin contar con la venta de su inventario.")

        st.info("La **Prueba Ácida** es un indicador más estricto de la liquidez. Si es mucho menor que la Razón Corriente, significa que la empresa depende fuertemente de sus inventarios para mantenerse a flote.")


# ==============================================================================
#      PESTAÑA 5: ESTADOS FINANCIEROS CONSOLIDADOS CON FILTRO
# ==============================================================================
with tab5:
    st.header("📈 Estados Financieros Consolidados por Periodo y Centro de Costo")
    st.markdown("""
    Esta sección te permite consolidar (sumar) todos los periodos históricos para obtener una visión acumulada. 
    Puedes filtrar por un **Centro de Costo** específico o ver el **Consolidado General** de la compañía.
    """)

    if 'datos_historicos' in st.session_state and st.session_state.datos_historicos:
        
        # --- Lógica para obtener la lista de Centros de Costo ---
        # Se asume que los dataframes brutos se llaman 'df_er' y 'df_bg' dentro de 'datos_historicos'
        # y que contienen la columna 'Centro de Costo'.
        all_cost_centers = set()
        for period_data in datos_historicos.values():
            # Intentar obtener centros de costo del estado de resultados
            if 'df_er' in period_data and 'Centro de Costo' in period_data['df_er'].columns:
                all_cost_centers.update(period_data['df_er']['Centro de Costo'].unique())
            # Intentar obtener centros de costo del balance general
            if 'df_bg' in period_data and 'Centro de Costo' in period_data['df_bg'].columns:
                 all_cost_centers.update(period_data['df_bg']['Centro de Costo'].unique())
        
        # Crear la lista de opciones para el filtro, con "Consolidado" al principio
        lista_filtros_cc = ["Consolidado"] + sorted(list(all_cost_centers))
        
        cc_seleccionado = st.selectbox(
            "Selecciona un Centro de Costo o el Consolidado General:",
            options=lista_filtros_cc,
            key="cc_consolidado_historico"
        )
        
        st.info(f"Mostrando datos para: **{cc_seleccionado}**")

        # --- Lógica de consolidación ---
        lista_er = []
        lista_bg = []

        if cc_seleccionado == "Consolidado":
            # Si se elige "Consolidado", se usan los dataframes master (totales)
            for periodo, data in datos_historicos.items():
                if 'df_er_master' in data:
                    lista_er.append(data['df_er_master'])
                if 'df_bg_master' in data:
                    lista_bg.append(data['df_bg_master'])
        else:
            # Si se elige un Centro de Costo, se filtran los dataframes brutos
            for periodo, data in datos_historicos.items():
                # Filtrar Estado de Resultados
                if 'df_er' in data:
                    df_er_filtrado = data['df_er'][data['df_er']['Centro de Costo'] == cc_seleccionado]
                    if not df_er_filtrado.empty:
                        lista_er.append(df_er_filtrado)
                # Filtrar Balance General
                if 'df_bg' in data:
                    df_bg_filtrado = data['df_bg'][data['df_bg']['Centro de Costo'] == cc_seleccionado]
                    if not df_bg_filtrado.empty:
                        lista_bg.append(df_bg_filtrado)

        # --- Visualización de resultados ---
        col_er, col_bg = st.columns(2)

        with col_er:
            st.subheader("Estado de Resultados (Acumulado)")
            if lista_er:
                df_er_completo = pd.concat(lista_er, ignore_index=True)
                df_er_consolidado = df_er_completo.groupby('Cuenta')['Valor'].sum().reset_index()
                st.dataframe(
                    df_er_consolidado.style.format({'Valor': "${:,.0f}"}),
                    use_container_width=True,
                    height=600 # Altura ajustable
                )
            else:
                st.warning(f"No se encontraron datos del Estado de Resultados para '{cc_seleccionado}'.")

        with col_bg:
            st.subheader("Balance General (Acumulado)")
            st.markdown("<small>Nota: La suma de balances de diferentes periodos es conceptualmente inusual.</small>", unsafe_allow_html=True)
            if lista_bg:
                df_bg_completo = pd.concat(lista_bg, ignore_index=True)
                df_bg_consolidado = df_bg_completo.groupby('Cuenta')['Valor'].sum().reset_index()
                st.dataframe(
                    df_bg_consolidado.style.format({'Valor': "${:,.0f}"}),
                    use_container_width=True,
                    height=600 # Altura ajustable
                )
            else:
                st.warning(f"No se encontraron datos del Balance General para '{cc_seleccionado}'.")

    else:
        st.error("No hay datos históricos cargados para generar el consolidado.")
