# kpis_y_analisis.py
# Archivo completo y corregido para análisis financiero avanzado.

import pandas as pd
import streamlit as st
import google.generativeai as genai
from mi_logica_original import get_principal_account_value, COL_CONFIG
import numpy as np

def calcular_kpis_periodo(df_er: pd.DataFrame, df_bg: pd.DataFrame, cc_filter: str = 'Todos') -> dict:
    """
    Calcula un conjunto completo de KPIs financieros para un período específico.
    Esta función es la ÚNICA fuente de verdad para los cálculos numéricos.

    *** LÓGICA DE CÁLCULO 100% AJUSTADA A LÓGICA MIXTA ***
    - ESTADO DE RESULTADOS (Lógica Estándar): Ingresos (+), Costos/Gastos (-).
    - BALANCE GENERAL (Lógica de Sistema): Activos (+), Pasivos (-), Patrimonio (-).
    """
    kpis = {}
    er_conf = COL_CONFIG.get('ESTADO_DE_RESULTADOS', {})
    bg_conf = COL_CONFIG.get('BALANCE_GENERAL', {})

    # --- 1. Determinar la columna de valores a utilizar ---
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
            ind_cc_cols = [
                v for k, v in er_conf.get('CENTROS_COSTO_COLS', {}).items()
                if str(k).lower() not in ['total', 'sin centro de coste'] and v in df_er.columns
            ]
            if ind_cc_cols:
                df_er['__temp_sum_kpi'] = df_er.loc[:, ind_cc_cols].sum(axis=1)
                val_col_kpi = '__temp_sum_kpi'
            else:
                scc_name = er_conf.get('CENTROS_COSTO_COLS', {}).get('Sin centro de coste')
                if scc_name and scc_name in df_er.columns:
                    val_col_kpi = scc_name
                elif 'Total_Consolidado_ER' in df_er.columns:
                    val_col_kpi = 'Total_Consolidado_ER'

    if not val_col_kpi or val_col_kpi not in df_er.columns:
        return {}

    # --- 2. Extracción y Cálculo del Estado de Resultados (Lógica Estándar) ---
    cuenta_er = er_conf['CUENTA']
    ingresos = get_principal_account_value(df_er, '4', val_col_kpi, cuenta_er)      # Positivo (+)
    costo_ventas = get_principal_account_value(df_er, '6', val_col_kpi, cuenta_er)  # Negativo (-)
    gastos_admin = get_principal_account_value(df_er, '51', val_col_kpi, cuenta_er) # Negativo (-)
    gastos_ventas = get_principal_account_value(df_er, '52', val_col_kpi, cuenta_er)# Negativo (-)
    costos_prod = get_principal_account_value(df_er, '7', val_col_kpi, cuenta_er)   # Negativo (-)
    gastos_no_op = get_principal_account_value(df_er, '53', val_col_kpi, cuenta_er) # Negativo (-)
    impuestos = get_principal_account_value(df_er, '54', val_col_kpi, cuenta_er)    # Negativo (-)

    # El cálculo es una suma directa gracias a la lógica estándar de signos.
    gastos_operativos = gastos_admin + gastos_ventas + costos_prod
    utilidad_bruta = ingresos + costo_ventas
    utilidad_operacional = utilidad_bruta + gastos_operativos
    utilidad_neta = utilidad_operacional + gastos_no_op + impuestos

    kpis['ingresos'] = ingresos
    kpis['costo_ventas'] = abs(costo_ventas) # Se muestra en positivo por convención
    kpis['gastos_operativos'] = abs(gastos_operativos) # Se muestra en positivo
    kpis['utilidad_bruta'] = utilidad_bruta
    kpis['utilidad_operacional'] = utilidad_operacional
    kpis['utilidad_neta'] = utilidad_neta

    # --- 3. Extracción de Balance General (Lógica de Sistema: Pas/Pat Negativos) ---
    cuenta_bg = bg_conf['CUENTA']
    saldo_final_col = bg_conf['SALDO_FINAL']

    activo_raw = get_principal_account_value(df_bg, '1', saldo_final_col, cuenta_bg)      # Positivo (+)
    pasivo_raw = get_principal_account_value(df_bg, '2', saldo_final_col, cuenta_bg)      # Negativo (-)
    patrimonio_raw = get_principal_account_value(df_bg, '3', saldo_final_col, cuenta_bg)  # Negativo (-)

    # Se usa abs() para obtener la magnitud real para los ratios.
    activo = activo_raw
    pasivo = abs(pasivo_raw)
    patrimonio = abs(patrimonio_raw)

    kpis['activo_raw'] = activo_raw
    kpis['pasivo_raw'] = pasivo_raw
    kpis['patrimonio_raw'] = patrimonio_raw
    kpis['activo'] = activo
    kpis['pasivo'] = pasivo
    kpis['patrimonio'] = patrimonio

    # --- 4. KPIs Adicionales para el Tablero ---
    activo_corriente = sum([get_principal_account_value(df_bg, c, saldo_final_col, cuenta_bg) for c in ['11','12','13','14']])
    inventarios = get_principal_account_value(df_bg, '14', saldo_final_col, cuenta_bg)
    pasivo_corriente_raw = sum([get_principal_account_value(df_bg, c, saldo_final_col, cuenta_bg) for c in ['21','22','23']])
    pasivo_corriente = abs(pasivo_corriente_raw)

    kpis['activo_corriente'] = activo_corriente
    kpis['pasivo_corriente'] = pasivo_corriente
    kpis['inventarios'] = inventarios

    # KPI de Diagnóstico: Ecuación Contable. Debe ser cercano a cero.
    kpis['descuadre_contable'] = activo_raw + pasivo_raw + patrimonio_raw # Ej: 1000 + (-700) + (-300) = 0

    # --- 5. Cálculo de Ratios Financieros ---
    def safe_divide(numerator, denominator):
        denom_abs = abs(denominator)
        if denom_abs == 0: return 0.0
        result = np.divide(numerator, denom_abs)
        return result if np.isfinite(result) else 0.0

    kpis['razon_corriente'] = safe_divide(activo_corriente, pasivo_corriente)
    kpis['prueba_acida'] = safe_divide(activo_corriente - inventarios, pasivo_corriente)
    kpis['endeudamiento_activo'] = safe_divide(pasivo, activo)
    kpis['endeudamiento_patrimonio'] = safe_divide(pasivo, patrimonio)
    kpis['apalancamiento'] = safe_divide(activo, patrimonio)
    kpis['margen_neto'] = safe_divide(utilidad_neta, ingresos)
    kpis['margen_operacional'] = safe_divide(utilidad_operacional, ingresos)
    kpis['roe'] = safe_divide(utilidad_neta, patrimonio)
    kpis['roa'] = safe_divide(utilidad_neta, activo)
    kpis['rotacion_activos'] = safe_divide(ingresos, activo)

    if '__temp_sum_kpi' in df_er.columns:
        df_er.drop(columns=['__temp_sum_kpi'], inplace=True)

    return kpis

def preparar_datos_tendencia(datos_historicos: dict) -> pd.DataFrame:
    """Convierte el diccionario de datos históricos en un DataFrame para graficar tendencias."""
    lista_periodos = [
        dict(periodo=periodo, **data['kpis']['Todos'])
        for periodo, data in datos_historicos.items()
        if 'kpis' in data and 'Todos' in data['kpis']
    ]
    if not lista_periodos:
        return pd.DataFrame()

    df_tendencia = pd.DataFrame(lista_periodos)
    df_tendencia['periodo'] = pd.to_datetime(df_tendencia['periodo'], format='%Y-%m')
    df_tendencia = df_tendencia.sort_values(by='periodo').reset_index(drop=True)
    return df_tendencia

@st.cache_data(show_spinner=False)
def generar_analisis_avanzado_ia(contexto_ia: dict):
    """
    Genera un análisis con IA, instruido para interpretar la LÓGICA MIXTA
    y la condición de patrimonio negativo.
    """
    try:
        api_key = st.secrets["google_ai"]["api_key"]
        genai.configure(api_key=api_key)
    except Exception:
        return "🔴 **Error:** No se pudo configurar la API de Google AI."

    kpis = contexto_ia.get("kpis", {})
    periodo = contexto_ia.get("periodo", "N/A")
    cc = contexto_ia.get("centro_costo", "Consolidado")

    patrimonio_es_negativo = kpis.get('patrimonio_raw', 0) < 0
    situacion_patrimonial = (
        "**CRÍTICO: La empresa opera con un PATRIMONIO NETO NEGATIVO (déficit patrimonial). Esto indica un estado de insolvencia técnica.**"
        if patrimonio_es_negativo
        else "La empresa opera con un patrimonio neto positivo."
    )

    prompt = f"""
    **Rol:** Eres un Director Financiero (CFO) y experto en reestructuración de empresas. Tu análisis debe ser agudo, detectar las causas raíz y proponer soluciones.

    **REGLA DE ORO: INTERPRETACIÓN DE LÓGICA MIXTA (¡INSTRUCCIÓN CRÍTICA!)**
    Estás analizando datos con una convención de signos mixta:
    1.  **Estado de Resultados (P&L):** Usa la lógica financiera estándar. Ingresos son POSITIVOS (+), y Costos/Gastos son NEGATIVOS (-).
    2.  **Balance General (BS):** Usa una lógica de sistema contable. Activos son POSITIVOS (+), pero **Pasivos y Patrimonio son NEGATIVOS (-)**.

    **Tu Misión:** Conecta los resultados del P&L (ej. una pérdida neta) con sus consecuencias directas en el Balance General (ej. el empeoramiento del patrimonio negativo).

    **Contexto del Análisis:**
    - **Periodo:** {periodo}
    - **Unidad de Negocio:** "{cc}"
    - **SITUACIÓN PATRIMONIAL:** {situacion_patrimonial}

    **Indicadores Clave (KPIs) del Periodo:**
    - **Resultado Neto:** ${kpis.get('utilidad_neta', 0):,.0f}
    - **ROE (Calculado sobre magnitud):** {kpis.get('roe', 0):.2%} (¡Interpretar con extrema cautela si el patrimonio es negativo!)
    - **ROA (Retorno sobre Activos):** {kpis.get('roa', 0):.2%}
    - **Margen Neto:** {kpis.get('margen_neto', 0):.2%}
    - **Liquidez (Razón Corriente):** {kpis.get('razon_corriente', 0):.2f}
    - **Endeudamiento (Pasivo / Activo):** {kpis.get('endeudamiento_activo', 0):.2%}
    - **Endeudamiento (Pasivo / Patrimonio):** {kpis.get('endeudamiento_patrimonio', 0):.2f} (Un valor alto aquí es una señal de alerta máxima)
    - **Descuadre Contable:** ${kpis.get('descuadre_contable', 0):,.0f} (Debe ser cero)

    **Instrucciones de Respuesta:**
    Con base en TODA la información, y poniendo especial atención a la situación patrimonial, genera un informe de diagnóstico y estrategia.

    ### Diagnóstico de Viabilidad 🩺
    (Ofrece un veredicto directo. ¿La empresa es viable? Comienza con la situación patrimonial y el resultado neto. Ej: "Diagnóstico: La empresa se encuentra en un estado de **insolvencia técnica**, con un déficit patrimonial de ${kpis.get('patrimonio', 0):,.0f}. La operación del periodo agravó esta situación al generar una **pérdida neta de ${kpis.get('utilidad_neta', 0):,.0f}**, que se restó directamente del ya negativo patrimonio.")

    ### Análisis Causa-Raíz 🔬
    (Conecta los puntos. ¿La rentabilidad del P&L es la causa de la insolvencia del BS?
    1.  **Rentabilidad Operativa:** El margen neto de {kpis.get('margen_neto', 0):.2%} indica si el negocio es rentable o no. ¿Esta rentabilidad (o falta de ella) explica la situación patrimonial?
    2.  **Estructura de Capital:** El ratio Pasivo/Patrimonio de {kpis.get('endeudamiento_patrimonio', 0):.2f} muestra la dependencia de la deuda. ¿Las pérdidas acumuladas han erosionado el patrimonio hasta hacerlo negativo?
    3.  **Liquidez vs. Solvencia:** La Razón Corriente es de {kpis.get('razon_corriente', 0):.2f}. ¿Puede la empresa pagar sus deudas a corto plazo, a pesar de ser insolvente a largo plazo?)

    ### Plan de Acción Estratégico 💡
    (Proporciona 3 recomendaciones jerarquizadas para abordar la situación.
    1.  **Acción Inmediata (Frenar la Pérdida):** ¿Qué ajuste operativo, basado en el P&L, es prioritario? (Ej: Reducir el costo de ventas en un 5% para mejorar el margen bruto; recortar gastos de ventas que no generan retorno).
    2.  **Acción a Mediano Plazo (Restaurar la Solvencia):** ¿Cómo se soluciona el problema del Balance General? (Ej: Plan de desinversión de activos no estratégicos para pagar deuda y reducir el pasivo; negociar una quita de deuda con acreedores).
    3.  **Acción a Largo Plazo (Recapitalización):** ¿De dónde vendrá el dinero nuevo? (Ej: Búsqueda activa de un socio capitalista para una inyección de capital de $Y, necesaria para restaurar el patrimonio a positivo).)
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        cleaned_response = response.text.replace('•', '*')
        return cleaned_response
    except Exception as e:
        return f"🔴 **Error al contactar la IA:** {e}"

@st.cache_data(show_spinner=False)
def generar_analisis_tendencia_ia(_df_tendencia: pd.DataFrame):
    """
    Genera un análisis de EVOLUCIÓN y TENDENCIA con IA, instruida sobre la lógica mixta.
    """
    try:
        api_key = st.secrets["google_ai"]["api_key"]
        genai.configure(api_key=api_key)
    except Exception:
        return "🔴 **Error:** No se encontró la clave de API de Google AI."

    if _df_tendencia.empty or len(_df_tendencia) < 2:
        return "ℹ️ Se necesitan al menos dos periodos para un análisis de tendencia."

    primer_periodo = _df_tendencia.iloc[0]
    ultimo_periodo = _df_tendencia.iloc[-1]
    
    # La trayectoria de la insolvencia es el dato más importante
    evolucion_patrimonial = f"El patrimonio neto (déficit) evolucionó de ${primer_periodo['patrimonio_raw']:,.0f} a ${ultimo_periodo['patrimonio_raw']:,.0f}."

    resumen_datos = f"""
    - **Horizonte de Análisis:** De {primer_periodo['periodo'].strftime('%Y-%m')} a {ultimo_periodo['periodo'].strftime('%Y-%m')}.
    - **Evolución del Patrimonio (Insolvencia):** {evolucion_patrimonial}
    - **Utilidad Neta:** Evolucionó de ${primer_periodo['utilidad_neta']:,.0f} a ${ultimo_periodo['utilidad_neta']:,.0f}.
    - **Ingresos:** Crecieron de ${primer_periodo['ingresos']:,.0f} a ${ultimo_periodo['ingresos']:,.0f}.
    - **Margen Neto:** Cambió de {primer_periodo['margen_neto']:.2%} a {ultimo_periodo['margen_neto']:.2%}.
    - **Endeudamiento (Pasivo/Patrimonio):** Varió de {primer_periodo['endeudamiento_patrimonio']:.2f} a {ultimo_periodo['endeudamiento_patrimonio']:.2f}.
    """

    prompt = f"""
    **Rol:** Eres un Analista Financiero Senior evaluando la trayectoria de una empresa en crisis.

    **REGLA DE ORO: INTERPRETACIÓN DE LÓGICA MIXTA (¡INSTRUCCIÓN CRÍTICA!)**
    - El **Estado de Resultados** usa lógica estándar (Ingresos +, Gastos -).
    - El **Balance General** usa una lógica de sistema (Activos +, Pasivos/Patrimonio -). El dato 'Evolución del Patrimonio' muestra el valor crudo negativo.

    **Tu Misión:** Analiza si la tendencia de rentabilidad está mejorando o empeorando la situación de insolvencia de la empresa.

    **Resumen Ejecutivo de la Evolución Financiera:**
    {resumen_datos}

    **Instrucciones de Respuesta:**
    Genera un informe de evolución estratégica, enfocándote en la viabilidad a largo plazo.

    ### Veredicto Estratégico de la Trayectoria 📜
    (En un párrafo, da un veredicto claro sobre la tendencia. ¿La empresa se acerca a la viabilidad o se hunde más en la insolvencia? Ej: "La trayectoria de la compañía es preocupante. Aunque los ingresos muestran un ligero crecimiento, la rentabilidad sigue siendo negativa, lo que ha provocado que el déficit patrimonial se incremente de X a Y, haciendo a la empresa aún más insolvente.")

    ### Análisis de Evolución por Dimensión 🔍
    - **Solvencia y Rentabilidad:** ¿La tendencia de la utilidad neta es positiva? ¿Es suficiente para revertir el déficit patrimonial? ¿O las pérdidas continúan erosionando la base de la empresa?
    - **Estructura de Capital:** ¿Cómo ha evolucionado el ratio Pasivo/Patrimonio? ¿La dependencia de la deuda está aumentando o disminuyendo?
    - **Operación:** ¿La tendencia del margen neto muestra una mejora en la eficiencia operativa o un deterioro?

    ### Prioridades Estratégicas Basadas en la Tendencia 🎯
    (Basado en la evolución, define 2-3 prioridades. Ej: "1. **Revertir la Tendencia de Pérdidas:** Es la máxima prioridad. Implementar un plan de choque para alcanzar el punto de equilibrio en 6 meses. 2. **Negociar con Acreedores:** La tendencia muestra que la deuda es insostenible; es crucial iniciar un proceso de reestructuración de pasivos.")
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        cleaned_response = response.text.replace('•', '*')
        return cleaned_response
    except Exception as e:
        return f"🔴 **Error al contactar la IA:** {e}"

@st.cache_data(show_spinner=False)
def generar_analisis_con_prompt_libre(prompt_personalizado: str):
    """
    Genera un análisis de IA a partir de un prompt libre y directo del usuario.
    """
    try:
        api_key = st.secrets["google_ai"]["api_key"]
        genai.configure(api_key=api_key)
    except Exception:
        return "🔴 **Error:** No se encontró la clave de API de Google AI."

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt_personalizado)
        cleaned_response = response.text.replace('•', '*')
        return cleaned_response
    except Exception as e:
        return f"🔴 **Error al contactar la IA:** {e}"
