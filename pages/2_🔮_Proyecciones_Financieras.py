# pages/2_🔮_Proyecciones_Financieras.py
import streamlit as st
import pandas as pd
import numpy as np
from analisis_adicional import construir_flujo_de_caja # Importamos la nueva función
from mi_logica_original import COL_CONFIG, get_principal_account_value

st.set_page_config(layout="wide", page_title="Proyecciones Financieras")
st.title("🔮 Proyecciones y Flujo de Caja")

# --- Verificación de autenticación y carga de datos ---
if 'authenticated' not in st.session_state or not st.session_state.authenticated:
    st.warning("Por favor, autentícate en la página principal para continuar.")
    st.stop()
if 'datos_historicos' not in st.session_state or not st.session_state.datos_historicos:
    st.error("No se han cargado datos históricos. Ve a la página principal y refresca los datos.")
    st.stop()

datos_historicos = st.session_state.datos_historicos
sorted_periods = sorted(datos_historicos.keys(), reverse=True)
ultimo_periodo = sorted_periods[0]

# --- Pestañas de Proyecciones ---
tab1, tab2 = st.tabs(["💸 Estado de Flujo de Caja", "🚀 Proyecciones a Futuro"])

# ==============================================================================
#                        PESTAÑA 1: ESTADO DE FLUJO DE CAJA
# ==============================================================================
with tab1:
    st.header("Análisis del Flujo de Caja (Método Indirecto)")
    st.markdown("Entiende cómo las actividades de operación, inversión y financiación impactan la posición de efectivo de la empresa.")
    
    col1, col2 = st.columns(2)
    with col1:
        periodo_actual_fc = st.selectbox("Periodo a Analizar:", sorted_periods, key="periodo_fc_actual")
    with col2:
        periodos_anteriores_fc = [p for p in sorted_periods if p < periodo_actual_fc]
        periodo_anterior_fc = st.selectbox(
            "Periodo Anterior (para cálculo de variaciones):", 
            periodos_anteriores_fc, 
            key="periodo_fc_anterior",
            disabled=not periodos_anteriores_fc
        )

    if periodo_anterior_fc:
        # Cargar datos necesarios
        df_er_actual = datos_historicos[periodo_actual_fc]['df_er_master']
        df_bg_actual = datos_historicos[periodo_actual_fc]['df_bg_master']
        df_bg_anterior = datos_historicos[periodo_anterior_fc]['df_bg_master']
        
        er_conf = COL_CONFIG['ESTADO_DE_RESULTADOS']
        bg_conf = COL_CONFIG['BALANCE_GENERAL']
        total_col_name = er_conf.get('CENTROS_COSTO_COLS', {}).get('Total', 'Total_Consolidado_ER')

        with st.spinner("Construyendo el Estado de Flujo de Caja..."):
            df_flujo_caja = construir_flujo_de_caja(
                df_er_actual, df_bg_actual, df_bg_anterior,
                val_col_er=total_col_name,
                cuenta_er=er_conf['CUENTA'],
                saldo_final_bg=bg_conf['SALDO_FINAL'],
                cuenta_bg=bg_conf['CUENTA']
            )
        
        st.subheader("Estado de Flujo de Caja")
        st.dataframe(df_flujo_caja.style.format({'Valor': "${:,.0f}"}), use_container_width=True)

        # Análisis con IA del Flujo de Caja
        with st.expander("🧠 Ver Análisis del Flujo de Caja por IA", expanded=True):
            fco = df_flujo_caja.loc[df_flujo_caja['Concepto'] == '**Flujo de Efectivo de Operación (FCO)**', 'Valor'].iloc[0]
            fci = df_flujo_caja.loc[df_flujo_caja['Concepto'] == '**Flujo de Efectivo de Inversión (FCI)**', 'Valor'].iloc[0]
            fcf = df_flujo_caja.loc[df_flujo_caja['Concepto'] == '**Flujo de Efectivo de Financiación (FCF)**', 'Valor'].iloc[0]
            
            prompt_flujo_caja = f"""
            **Rol:** Eres un CFO virtual analizando el flujo de caja. Tu análisis debe ser directo y accionable.
            **Contexto:** Se ha generado el siguiente resumen de flujo de caja para el periodo {periodo_actual_fc}:
            - Flujo de Efectivo de Operación (FCO): ${fco:,.0f}
            - Flujo de Efectivo de Inversión (FCI): ${fci:,.0f}
            - Flujo de Efectivo de Financiación (FCF): ${fcf:,.0f}

            **Instrucciones:**
            1.  **Diagnóstico General:** ¿La empresa genera o consume efectivo? ¿De dónde proviene y a dónde va el dinero?
            2.  **Análisis por Componente:**
                - **Operación (FCO):** Si es positivo, la operación es sana. Si es negativo, es una señal de alerta.
                - **Inversión (FCI):** Si es negativo, la empresa está invirtiendo en crecer (bueno). Si es positivo, podría estar desinvirtiendo.
                - **Financiación (FCF):** Si es positivo, está entrando dinero por deuda o capital. Si es negativo, se está pagando deuda o dividendos.
            3.  **Veredicto Final:** Concluye con un veredicto sobre la salud del flujo de caja. ¿El patrón es sostenible? (Ej: "Patrón saludable de crecimiento", "Dependencia de financiación externa", "Alerta de quema de efectivo operativo").
            """
            # Aquí, de nuevo, iría la llamada a la IA. Mostramos un ejemplo.
            analisis_ia_flujo = f"""
            ###  Diagnóstico del Flujo de Caja 🎯

            #### Análisis por Componente
            * **💧 Operación (FCO):** {'Genera' if fco >= 0 else 'Consume'} **${abs(fco):,.0f}**. Un FCO positivo indica que el negocio principal es autosuficiente y genera el efectivo necesario para operar, lo cual es una señal muy saludable.
            * **🏗️ Inversión (FCI):** Se {'invirtieron' if fci < 0 else 'recibieron'} **${abs(fci):,.0f}**. Un valor negativo aquí es típicamente positivo, ya que sugiere que la empresa está invirtiendo en activos fijos para su crecimiento futuro.
            * **🏦 Financiación (FCF):** Se {'obtuvieron' if fcf >= 0 else 'pagaron'} **${abs(fcf):,.0f}** a través de actividades de financiación. Esto puede indicar nueva deuda, pago de préstamos o dividendos.

            #### Veredicto Final 📜
            El perfil de flujo de caja actual sugiere un **patrón de crecimiento saludable**. La empresa genera efectivo a través de sus operaciones, lo cual utiliza para financiar su expansión (inversión) y, posiblemente, para gestionar su estructura de capital. Este es el escenario ideal para una empresa en fase de crecimiento.
            """
            st.markdown(analisis_ia_flujo)

    else:
        st.info("Selecciona un periodo anterior para poder calcular las variaciones y construir el flujo de caja.")

# ==============================================================================
#                           PESTAÑA 2: PROYECCIONES
# ==============================================================================
with tab2:
    st.header("Proyección de Estados Financieros")
    st.markdown("Estima los resultados futuros basándote en supuestos de crecimiento y comportamiento histórico.")

    # --- Parámetros de Proyección ---
    st.sidebar.header("Parámetros de Proyección")
    crecimiento_ingresos = st.sidebar.slider("Crecimiento de Ingresos (%) para el próximo periodo", -20.0, 50.0, 5.0, 0.5) / 100
    porc_costo_venta = st.sidebar.slider("% Costo de Venta sobre Ingresos", 30.0, 90.0, 60.0, 0.5) / 100
    porc_gasto_op = st.sidebar.slider("% Gastos Operativos sobre Ingresos", 10.0, 50.0, 25.0, 0.5) / 100

    # --- Lógica de Proyección (Simplificada) ---
    kpis_ultimo_periodo = datos_historicos[ultimo_periodo]['kpis']['Todos']
    
    # Proyección ER
    ingresos_proy = kpis_ultimo_periodo['ingresos'] * (1 + crecimiento_ingresos)
    costo_venta_proy = ingresos_proy * porc_costo_venta * -1
    utilidad_bruta_proy = ingresos_proy + costo_venta_proy
    gasto_op_proy = ingresos_proy * porc_gasto_op * -1
    utilidad_op_proy = utilidad_bruta_proy + gasto_op_proy
    utilidad_neta_proy = utilidad_op_proy # Simplificado
    
    # Proyección BG (muy simplificada, se necesita más lógica para un modelo robusto)
    df_bg_ultimo = datos_historicos[ultimo_periodo]['df_bg_master']
    saldo_final_col = COL_CONFIG['BALANCE_GENERAL']['SALDO_FINAL']
    cuenta_bg_col = COL_CONFIG['BALANCE_GENERAL']['CUENTA']
    
    caja_inicial = get_principal_account_value(df_bg_ultimo, '11', saldo_final_col, cuenta_bg_col)
    caja_final_proy = caja_inicial + utilidad_neta_proy # Supuesto ultra-simplificado
    activo_proy = get_principal_account_value(df_bg_ultimo, '1', saldo_final_col, cuenta_bg_col) + (caja_final_proy - caja_inicial)
    pasivo_proy = get_principal_account_value(df_bg_ultimo, '2', saldo_final_col, cuenta_bg_col)
    patrimonio_proy = get_principal_account_value(df_bg_ultimo, '3', saldo_final_col, cuenta_bg_col) + utilidad_neta_proy

    # --- Mostrar Resultados ---
    st.subheader("Estado de Resultados Proyectado")
    df_proy_er = pd.DataFrame({
        'Concepto': ['Ingresos', 'Costo de Ventas', '**Utilidad Bruta**', 'Gastos Operativos', '**Utilidad Neta Proyectada**'],
        'Valor': [ingresos_proy, costo_venta_proy, utilidad_bruta_proy, gasto_op_proy, utilidad_neta_proy]
    })
    st.dataframe(df_proy_er.style.format({'Valor': "${:,.0f}"}), use_container_width=True)

    st.subheader("Balance General Proyectado")
    df_proy_bg = pd.DataFrame({
        'Concepto': ['Activo', 'Pasivo', 'Patrimonio', '**Verificación (A=P+Pt)**'],
        'Valor': [activo_proy, pasivo_proy, patrimonio_proy, activo_proy - (pasivo_proy + patrimonio_proy)]
    })
    st.dataframe(df_proy_bg.style.format({'Valor': "${:,.0f}"}), use_container_width=True)
    
    st.warning("Nota: Las proyecciones del Balance General y Flujo de Caja son simplificadas. Un modelo completo requeriría supuestos detallados sobre capital de trabajo, inversiones y financiación.")
