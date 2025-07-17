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

    # --- Enriquecer el diccionario de KPIs ---
    kpis['activo'] = activo
    kpis['pasivo'] = pasivo
    kpis['patrimonio'] = patrimonio
    kpis['activo_corriente'] = activo_corriente
    kpis['pasivo_corriente'] = pasivo_corriente
    kpis['inventarios'] = inventarios
    
    # --- CÁLCULOS DE RATIOS CORREGIDOS ---
    kpis['razon_corriente'] = activo_corriente / pasivo_corriente if pasivo_corriente != 0 else 0
    kpis['endeudamiento_activo'] = pasivo / activo if activo != 0 else 0
    
    # ROE: Usar abs() en utilidad_neta para asegurar un resultado positivo.
    kpis['roe'] = abs(utilidad_neta) / patrimonio if patrimonio != 0 else 0
    
    # Márgenes: utilidad e ingresos son negativos, la división ya da un resultado positivo. No se necesita abs().
    kpis['margen_neto'] = utilidad_neta / ingresos if ingresos != 0 else 0
    kpis['margen_operacional'] = utilidad_operacional / ingresos if ingresos != 0 else 0
    
    # KPIs para DuPont (deben ser positivos)
    kpis['rotacion_activos'] = abs(ingresos) / activo if activo != 0 else 0
    kpis['apalancamiento'] = activo / patrimonio if patrimonio != 0 else 0

    if '__temp_sum_kpi' in df_er.columns:
        df_er.drop(columns=['__temp_sum_kpi'], inplace=True)
        
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
def generar_analisis_avanzado_ia(contexto_ia: dict):
    """Genera un análisis financiero profundo usando un contexto enriquecido."""
    try:
        api_key = st.secrets["google_ai"]["api_key"]
        genai.configure(api_key=api_key)
    except Exception:
        return "🔴 **Error:** No se encontró la clave de API de Google AI."

    # Extraemos los datos del contexto
    kpis = contexto_ia.get("kpis", {})
    periodo = contexto_ia.get("periodo", "N/A")
    cc = contexto_ia.get("centro_costo", "N/A")
    convencion = contexto_ia.get("convencion_contable", "")
    favorables = contexto_ia.get("variaciones_favorables", [])
    desfavorables = contexto_ia.get("variaciones_desfavorables", [])

    # Formateamos las variaciones para el prompt
    fav_str = "\n".join([f"- {item['Descripción']}: ${item['Variacion_Absoluta']:,.0f}" for item in favorables]) if favorables else "No hubo variaciones favorables significativas."
    des_str = "\n".join([f"- {item['Descripción']}: ${item['Variacion_Absoluta']:,.0f}" for item in desfavorables]) if desfavorables else "No hubo variaciones desfavorables significativas."

    prompt = f"""
    **Rol:** Actúa como un Asesor Financiero Estratégico y CFO virtual. Tu análisis debe ser agudo, directo y accionable.

    **¡REGLA DE ORO PARA EL ANÁLISIS!** {convencion}

    **Contexto de Análisis:**
    - **Periodo:** {periodo}
    - **Centro de Costo / Unidad de Negocio:** "{cc}"

    **Indicadores Clave (KPIs) del Periodo:**
    - **Utilidad Neta:** ${kpis.get('utilidad_neta', 0):,.0f}
    - **Margen Neto:** {kpis.get('margen_neto', 0):.2%}
    - **ROE (Rentabilidad sobre Patrimonio):** {kpis.get('roe', 0):.2%}
    - **Razón Corriente (Liquidez):** {kpis.get('razon_corriente', 0):.2f}
    - **Nivel de Endeudamiento:** {kpis.get('endeudamiento_activo', 0):.2%}

    **Análisis Comparativo vs. Periodo Anterior:**
    - **Principales Impactos Positivos (que ayudaron a la utilidad):**
    {fav_str}
    - **Principales Impactos Negativos (que perjudicaron la utilidad):**
    {des_str}

    **Instrucciones:**
    Con base en TODA la información anterior, genera un informe ejecutivo conciso. Usa emojis para resaltar puntos (ej: 📈, ⚠️, ✅, 💡). La estructura debe ser:

    ### Diagnóstico General 🎯
    (Ofrece un veredicto claro y directo sobre la salud financiera para este centro de costo en este periodo. ¿Fue un buen o mal periodo y por qué?)

    ### Puntos Clave del Análisis 🔑
    (En una lista, detalla las 3 observaciones más importantes. Conecta los KPIs con las variaciones. Por ejemplo: "La caída en el margen neto se explica principalmente por el aumento inesperado en [Gasto X], como se ve en los impactos negativos.")

    ### Plan de Acción Recomendado 💡
    (Proporciona 2 o 3 recomendaciones específicas, prácticas y accionables basadas en tu diagnóstico para mejorar los resultados en el siguiente periodo.)
    """

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        cleaned_response = response.text.replace('•', '*')
        return cleaned_response
    except Exception as e:
        return f"🔴 **Error al contactar la IA:** {e}."

@st.cache_data(show_spinner=False)
def generar_analisis_tendencia_ia(_df_tendencia: pd.DataFrame):
    """Genera un análisis de EVOLUCIÓN y TENDENCIA para un comité directivo."""
    try:
        api_key = st.secrets["google_ai"]["api_key"]
        genai.configure(api_key=api_key)
    except Exception:
        return "🔴 **Error:** No se encontró la clave de API de Google AI."

    if _df_tendencia.empty or len(_df_tendencia) < 2:
        return "ℹ️ Se necesitan al menos dos periodos para un análisis de tendencia."

    primer_periodo = _df_tendencia.iloc[0]
    ultimo_periodo = _df_tendencia.iloc[-1]
    
    resumen_datos = f"""
    - **Periodo Analizado:** De {primer_periodo['periodo'].strftime('%Y-%m')} a {ultimo_periodo['periodo'].strftime('%Y-%m')}.
    - **Ingresos:** Crecieron de ${primer_periodo['ingresos']:,.0f} a ${ultimo_periodo['ingresos']:,.0f}.
    - **Utilidad Neta:** Pasó de ${primer_periodo['utilidad_neta']:,.0f} a ${ultimo_periodo['utilidad_neta']:,.0f}.
    - **Margen Neto:** Evolucionó de {primer_periodo['margen_neto']:.2%} a {ultimo_periodo['margen_neto']:.2%}.
    - **Razón Corriente (Liquidez):** Varió de {primer_periodo['razon_corriente']:.2f} a {ultimo_periodo['razon_corriente']:.2f}.
    - **Endeudamiento (sobre Activo):** Cambió de {primer_periodo['endeudamiento_activo']:.2%} a {ultimo_periodo['endeudamiento_activo']:.2%}.
    - **ROE:** Se movió de {primer_periodo['roe']:.2%} a {ultimo_periodo['roe']:.2%}.
    """

    prompt = f"""
    **Rol:** Eres un Analista Financiero Senior y Asesor Estratégico presentando un informe de evolución de negocio a un comité directivo. Tu análisis debe ser agudo, orientado a la acción y fácil de entender.

    **¡NOTACIÓN CONTABLE CRÍTICA!** Para tu análisis, ten en cuenta esta regla de oro:
    - **Valores NEGATIVOS en Ingresos y Utilidades son FAVORABLES (representan ganancias).**
    - **Valores POSITIVOS en Gastos y Costos son DESFAVORABLES.**
    - Ejemplo de interpretación: Si la utilidad neta evoluciona de -500 a -600, es una **MEJORA** en la rentabilidad. Si un gasto evoluciona de 100 a 80, es una **MEJORA** en la eficiencia.

    **Contexto:** Has analizado la evolución financiera consolidada de la compañía durante varios periodos. Aquí está el resumen de la trayectoria:
    {resumen_datos}
    
    **Instrucciones de Formato y Contenido:**
    Tu respuesta debe ser un informe de evolución de alto nivel, visualmente organizado con Markdown y emojis (📈, 📉, ⚠️, ✅, 💡). La estructura debe ser la siguiente:
    
    ### Veredicto Estratégico 📜
    (En un párrafo, da un veredicto sobre la trayectoria general de la compañía. ¿La tendencia es positiva y sostenible, muestra signos de estancamiento, o hay señales de alerta preocupantes? Justifica tu conclusión basándote en la regla de notación contable.)
    
    ### Análisis de Evolución por Área 🔍
    (Presenta un análisis en formato de lista. Para cada área, describe la tendencia observada y su implicación estratégica, siempre interpretando los signos correctamente.)
    - **Crecimiento y Rentabilidad:** ¿El crecimiento de los ingresos (valores negativos más grandes) se traduce en una mayor rentabilidad (utilidad neta negativa más grande y márgenes positivos más altos)? ¿O están creciendo los ingresos a costa de los márgenes?
    - **Eficiencia Operativa:** ¿Cómo ha evolucionado la relación entre ingresos y gastos operativos a lo largo del tiempo? ¿La empresa se está volviendo más o menos eficiente?
    - **Salud y Riesgo Financiero:** ¿La posición de liquidez (Razón Corriente) ha mejorado o empeorado? ¿El nivel de endeudamiento es sostenible o representa un riesgo creciente?
    
    ### Prioridades para el Próximo Trimestre 🎯
    (Basado en la evolución, proporciona una lista de 2 a 3 prioridades estratégicas y accionables que la dirección debería enfocarse en los próximos 3 meses.)
    """

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        cleaned_response = response.text.replace('•', '*')
        return cleaned_response
    except Exception as e:
        return f"🔴 **Error al contactar la IA:** {e}."
