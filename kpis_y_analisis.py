# kpis_y_analisis.py
import pandas as pd
import streamlit as st
import google.generativeai as genai
from mi_logica_original import get_principal_account_value, COL_CONFIG
import numpy as np

def calcular_kpis_periodo(df_er: pd.DataFrame, df_bg: pd.DataFrame, cc_filter: str = 'Todos') -> dict:
    """
    Calcula un conjunto de KPIs financieros para un período específico.
    
    *** LÓGICA DE CÁLCULO 100% CORREGIDA Y ESTANDARIZADA ***
    - Esta función es ahora la ÚNICA fuente de verdad para los cálculos.
    - Asume la lógica financiera estándar:
        - Ingresos (cuenta '4') son POSITIVOS.
        - Costos y Gastos (cuentas '5', '6', '7') son NEGATIVOS.
    - Se adapta para calcular el consolidado ('Todos') o un centro de costo específico.
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
        # Busca la columna 'Total' o la suma de las columnas de CC
        total_col_name = er_conf.get('CENTROS_COSTO_COLS', {}).get('Total')
        if total_col_name and total_col_name in df_er.columns:
            val_col_kpi = total_col_name
        else:
            # Suma dinámica si no hay una columna de total explícita
            ind_cc_cols = [
                v for k, v in er_conf.get('CENTROS_COSTO_COLS', {}).items() 
                if str(k).lower() not in ['total', 'sin centro de coste'] and v in df_er.columns
            ]
            if ind_cc_cols:
                df_er['__temp_sum_kpi'] = df_er.loc[:, ind_cc_cols].sum(axis=1)
                val_col_kpi = '__temp_sum_kpi'
            else:
                # Fallback a otras posibles columnas de total
                scc_name = er_conf.get('CENTROS_COSTO_COLS', {}).get('Sin centro de coste')
                if scc_name and scc_name in df_er.columns:
                    val_col_kpi = scc_name
                elif 'Total_Consolidado_ER' in df_er.columns:
                    val_col_kpi = 'Total_Consolidado_ER'

    if not val_col_kpi or val_col_kpi not in df_er.columns:
        return {} # Retorna KPIs vacíos si no se encuentra la columna de datos

    # --- 2. Extracción de Estado de Resultados (Lógica de Signos Estándar) ---
    cuenta_er = er_conf['CUENTA']
    # Ingresos son positivos (+)
    ingresos = get_principal_account_value(df_er, '4', val_col_kpi, cuenta_er)
    # Costos y Gastos son negativos (-)
    costo_ventas = get_principal_account_value(df_er, '6', val_col_kpi, cuenta_er)
    gastos_admin = get_principal_account_value(df_er, '51', val_col_kpi, cuenta_er)
    gastos_ventas = get_principal_account_value(df_er, '52', val_col_kpi, cuenta_er)
    costos_prod = get_principal_account_value(df_er, '7', val_col_kpi, cuenta_er)
    gastos_no_op = get_principal_account_value(df_er, '53', val_col_kpi, cuenta_er)
    impuestos = get_principal_account_value(df_er, '54', val_col_kpi, cuenta_er)
    
    # --- 3. Cálculo de Utilidades (Lógica Directa y Estándar) ---
    # La suma funciona directamente porque los gastos/costos ya son negativos.
    utilidad_bruta = ingresos + costo_ventas # Ej: 1000 + (-400) = 600
    gastos_operativos = gastos_admin + gastos_ventas + costos_prod
    utilidad_operacional = utilidad_bruta + gastos_operativos # Ej: 600 + (-300) = 300
    utilidad_antes_imp = utilidad_operacional + gastos_no_op
    utilidad_neta = utilidad_antes_imp + impuestos # Ej: 300 + (-50) = 250

    kpis['ingresos'] = ingresos
    kpis['costo_ventas'] = abs(costo_ventas) # Se almacena como valor absoluto para claridad en display
    kpis['gastos_operativos'] = abs(gastos_operativos) # Se almacena como valor absoluto
    kpis['utilidad_bruta'] = utilidad_bruta
    kpis['utilidad_operacional'] = utilidad_operacional
    kpis['utilidad_neta'] = utilidad_neta
    
    # --- 4. Extracción de Balance General (Valores Positivos por Naturaleza) ---
    cuenta_bg = bg_conf['CUENTA']
    saldo_final_col = bg_conf['SALDO_FINAL']
    
    # Función auxiliar para simplificar la extracción
    def get_bg_val(codes):
        if not isinstance(codes, list):
            codes = [codes]
        return sum(get_principal_account_value(df_bg, c, saldo_final_col, cuenta_bg) for c in codes)

    activo = get_bg_val('1')
    pasivo = get_bg_val('2')
    patrimonio = get_bg_val('3')
    activo_corriente = get_bg_val(['11', '12', '13', '14'])
    pasivo_corriente = get_bg_val(['21', '22', '23'])
    inventarios = get_bg_val('14')

    kpis['activo'] = activo
    kpis['pasivo'] = pasivo
    kpis['patrimonio'] = patrimonio
    kpis['activo_corriente'] = activo_corriente
    kpis['pasivo_corriente'] = pasivo_corriente
    kpis['inventarios'] = inventarios
    
    # --- 5. Cálculo de Ratios Financieros (Con Manejo Seguro de Ceros) ---
    # np.divide maneja divisiones por cero devolviendo 0 (o np.inf, que filtramos).
    # abs() se usa en denominadores para evitar problemas con valores residuales negativos.
    def safe_divide(numerator, denominator):
        if denominator == 0:
            return 0.0
        result = np.divide(numerator, abs(denominator))
        return result if np.isfinite(result) else 0.0

    kpis['razon_corriente'] = safe_divide(activo_corriente, pasivo_corriente)
    kpis['endeudamiento_activo'] = safe_divide(pasivo, activo)
    kpis['roe'] = safe_divide(utilidad_neta, patrimonio)
    kpis['roa'] = safe_divide(utilidad_neta, activo)
    kpis['margen_neto'] = safe_divide(utilidad_neta, ingresos)
    kpis['margen_operacional'] = safe_divide(utilidad_operacional, ingresos)
    kpis['rotacion_activos'] = safe_divide(ingresos, activo)
    kpis['apalancamiento'] = safe_divide(activo, patrimonio)

    # Limpieza de columna temporal
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
    Genera un análisis financiero profundo usando IA, basado en datos estandarizados.
    *** PROMPT MEJORADO PARA ANÁLISIS ESTRATÉGICO ***
    """
    try:
        api_key = st.secrets["google_ai"]["api_key"]
        genai.configure(api_key=api_key)
    except Exception:
        return "🔴 **Error:** No se pudo configurar la API de Google AI. Verifica la clave."

    kpis = contexto_ia.get("kpis", {})
    periodo = contexto_ia.get("periodo", "N/A")
    cc = contexto_ia.get("centro_costo", "Consolidado")
    favorables = contexto_ia.get("variaciones_favorables", [])
    desfavorables = contexto_ia.get("variaciones_desfavorables", [])

    # Formateo de impactos para el prompt
    fav_str = "\n".join([f"- {item['Descripción']}: ${item['Variacion_Absoluta']:,.0f}" for item in favorables]) if favorables else "No se identificaron impactos positivos significativos."
    des_str = "\n".join([f"- {item['Descripción']}: ${item['Variacion_Absoluta']:,.0f}" for item in desfavorables]) if desfavorables else "No se identificaron impactos negativos significativos."

    prompt = f"""
    **Rol:** Eres un Director Financiero (CFO) virtual y asesor estratégico. Tu análisis debe ser agudo, directo y enfocado en la toma de decisiones.

    **Lógica Financiera (Regla Fundamental):**
    Toda la información proporcionada sigue la contabilidad estándar:
    - **Utilidad/Ingresos:** Valores positivos son ganancias ✅.
    - **Pérdidas/Gastos:** Valores negativos son pérdidas 🔻.
    - Los KPIs (ROE, márgenes, etc.) están calculados bajo esta lógica universal.

    **Contexto del Análisis:**
    - **Periodo:** {periodo}
    - **Unidad de Negocio/Centro de Costo:** "{cc}"

    **Indicadores Clave de Desempeño (KPIs) del Periodo:**
    - **Utilidad Neta:** ${kpis.get('utilidad_neta', 0):,.0f}
    - **Margen Neto:** {kpis.get('margen_neto', 0):.2%}
    - **ROE (Retorno sobre Patrimonio):** {kpis.get('roe', 0):.2%}
    - **ROA (Retorno sobre Activos):** {kpis.get('roa', 0):.2%}
    - **Razón Corriente (Liquidez):** {kpis.get('razon_corriente', 0):.2f}
    - **Nivel de Endeudamiento (Pasivo/Activo):** {kpis.get('endeudamiento_activo', 0):.2%}

    **Causas del Resultado (Análisis de Variaciones vs. Periodo Anterior):**
    - **Impulsores Positivos (Cuentas que mejoraron el resultado):**
    {fav_str}
    - **Impulsores Negativos (Cuentas que empeoraron el resultado):**
    {des_str}

    **Instrucciones de Respuesta:**
    Basado en TODA la información, genera un informe ejecutivo conciso y accionable. Utiliza emojis para resaltar puntos (📈, ⚠️, ✅, 💡, 🔻).

    ### Diagnóstico General 🎯
    (Ofrece un veredicto claro sobre la salud financiera del periodo. Comienza con el resultado final (ganancia o pérdida) y luego explica las causas principales de forma directa. Ej: "El periodo cerró con una [ganancia/pérdida] neta de $X. Este resultado fue impulsado por [mencionar 1-2 impulsores positivos], pero se vio mermado por [mencionar 1-2 impulsores negativos].")

    ### Puntos Clave y Conexiones 🔑
    (Detalla 2-3 observaciones importantes, conectando los KPIs con las causas. Ej: "La rentabilidad, medida por el Margen Neto del {kpis.get('margen_neto', 0):.2%}, se vio afectada directamente por el aumento en los costos de ventas. A pesar de un crecimiento en ingresos, el margen se contrajo." o "La liquidez (Razón Corriente: {kpis.get('razon_corriente', 0):.2f}) se mantiene sólida, lo que provee un buen colchón para...")

    ### Plan de Acción Recomendado 💡
    (Proporciona 2-3 recomendaciones específicas, priorizadas y cuantificables basadas en el diagnóstico. Sé directo. Ej: "1. **Optimizar Costos de Ventas:** Investigar las causas del aumento de X y renegociar con proveedores para mejorar el margen bruto en un 2%. 2. **Controlar Gastos Operativos:** Implementar un presupuesto base cero para el área Y, con el objetivo de reducir los gastos en un 5% el próximo trimestre.")
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        # Limpieza simple para mejor formato en Markdown
        cleaned_response = response.text.replace('•', '*')
        return cleaned_response
    except Exception as e:
        st.error(f"Error al contactar la IA: {e}")
        return f"🔴 **Error al contactar la IA:** {e}"

@st.cache_data(show_spinner=False)
def generar_analisis_tendencia_ia(_df_tendencia: pd.DataFrame):
    """
    Genera un análisis de EVOLUCIÓN y TENDENCIA con IA y prompts mejorados.
    """
    try:
        api_key = st.secrets["google_ai"]["api_key"]
        genai.configure(api_key=api_key)
    except Exception:
        return "🔴 **Error:** No se pudo configurar la API de Google AI. Verifica la clave."

    if _df_tendencia.empty or len(_df_tendencia) < 2:
        return "ℹ️ Se necesitan al menos dos periodos de datos para realizar un análisis de tendencia."

    primer_periodo = _df_tendencia.iloc[0]
    ultimo_periodo = _df_tendencia.iloc[-1]
    
    resumen_datos = f"""
    - **Horizonte de Análisis:** De {primer_periodo['periodo'].strftime('%Y-%m')} a {ultimo_periodo['periodo'].strftime('%Y-%m')}.
    - **Ingresos:** Crecieron de ${primer_periodo['ingresos']:,.0f} a ${ultimo_periodo['ingresos']:,.0f}.
    - **Utilidad Neta:** Evolucionó de ${primer_periodo['utilidad_neta']:,.0f} a ${ultimo_periodo['utilidad_neta']:,.0f}.
    - **Margen Neto:** Cambió de {primer_periodo['margen_neto']:.2%} a {ultimo_periodo['margen_neto']:.2%}.
    - **ROE:** Se movió de {primer_periodo['roe']:.2%} a {ultimo_periodo['roe']:.2%}.
    - **Liquidez (Razón Corriente):** Varió de {primer_periodo['razon_corriente']:.2f} a {ultimo_periodo['razon_corriente']:.2f}.
    - **Endeudamiento (Pasivo/Activo):** Varió de {primer_periodo['endeudamiento_activo']:.2%} a {ultimo_periodo['endeudamiento_activo']:.2%}.
    """

    prompt = f"""
    **Rol:** Eres un Analista Financiero Senior y Asesor de Estrategia Corporativa. Tu objetivo es evaluar la trayectoria del negocio.

    **Lógica de Datos:**
    El resumen de evolución proporcionado utiliza datos financieros estandarizados (Ganancia > 0, Pérdida < 0). Tu análisis debe enfocarse en la magnitud y dirección de los cambios.

    **Resumen Ejecutivo de la Evolución Financiera:**
    {resumen_datos}
    
    **Instrucciones de Respuesta:**
    Genera un informe de evolución estratégica. Sé crítico y prospectivo.

    ### Veredicto Estratégico de la Trayectoria 📜
    (En un párrafo, da un veredicto claro sobre la tendencia general. ¿La empresa está en una trayectoria de fortalecimiento o debilitamiento? ¿El crecimiento es rentable? Ej: "La compañía muestra una clara tendencia de crecimiento en ingresos, sin embargo, la rentabilidad se ha estancado/deteriorado, como lo demuestra la compresión del margen neto. Esto sugiere que el crecimiento actual no es sostenible o que los costos están fuera de control.")

    ### Análisis de Evolución por Dimensión 🔍
    - **Rentabilidad (Utilidad, Márgenes, ROE):** ¿La capacidad de generar ganancias ha mejorado o empeorado? ¿El retorno para los accionistas (ROE) está creciendo a un ritmo adecuado? ¿Qué implica la tendencia del margen?
    - **Crecimiento y Eficiencia Operativa:** ¿El crecimiento de los ingresos es saludable? ¿La empresa es más o menos eficiente en convertir esos ingresos en utilidad operacional? Compara la tasa de crecimiento de ingresos con la de los costos.
    - **Salud y Riesgo Financiero (Liquidez y Endeudamiento):** ¿La posición de liquidez ha mejorado o se ha vuelto más riesgosa? ¿La dependencia de la deuda ha aumentado o disminuido? ¿Qué riesgos u oportunidades presenta esta tendencia?

    ### Prioridades Estratégicas para el Futuro 🎯
    (Basado en la evolución, define 2-3 prioridades clave. Deben ser acciones para capitalizar fortalezas o mitigar las debilidades reveladas en la tendencia. Ej: "1. **Foco en Rentabilidad, no solo Crecimiento:** Lanzar una iniciativa de revisión de precios. 2. **Fortalecer la Posición de Liquidez:** Optimizar el ciclo de conversión de efectivo.")
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        cleaned_response = response.text.replace('•', '*')
        return cleaned_response
    except Exception as e:
        st.error(f"Error al contactar la IA: {e}")
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
        return "🔴 **Error:** No se pudo configurar la API de Google AI. Verifica la clave."

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt_personalizado)
        cleaned_response = response.text.replace('•', '*')
        return cleaned_response
    except Exception as e:
        st.error(f"Error al contactar la IA: {e}")
        return f"🔴 **Error al contactar la IA:** {e}"
