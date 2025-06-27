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
    """Genera un análisis financiero profundo para un periodo único."""
    try:
        api_key = st.secrets["google_ai"]["api_key"]
        genai.configure(api_key=api_key)
    except Exception:
        return "🔴 **Error:** No se encontró la clave de API de Google AI."

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
    - Ingresos: ${_kpis_actuales.get('ingresos', 0):,.0f}
    - Utilidad Neta: ${_kpis_actuales.get('utilidad_neta', 0):,.0f}
    - Margen Neto: {_kpis_actuales.get('margen_neto', 0):.2%}
    - ROE: {_kpis_actuales.get('roe', 0):.2%}
    - Razón Corriente: {_kpis_actuales.get('razon_corriente', 0):.2f}
    - Endeudamiento: {_kpis_actuales.get('endeudamiento_activo', 0):.2%}
    **Top 5 Gastos Operativos:**
    {top_5_gastos_str}
    **Instrucciones:** Tu respuesta debe ser un informe gerencial profesional, fácil de leer y visualmente organizado. Usa emojis (ej: 📈, ⚠️, ✅, 💡) para enfatizar puntos. La estructura debe ser:
    ### Diagnóstico General 🎯
    (Veredicto claro y directo sobre la salud financiera.)
    ### Puntos Clave del Periodo 🔑
    (Lista con la implicación de negocio de la Rentabilidad, Costos y Solvencia.)
    ### Plan de Acción Recomendado 💡
    (Lista de 2-3 recomendaciones específicas y accionables.)
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
    **Contexto:** Has analizado la evolución financiera consolidada de la compañía durante varios periodos. Aquí está el resumen de la trayectoria:
    {resumen_datos}
    **Instrucciones de Formato y Contenido:**
    Tu respuesta debe ser un informe de evolución de alto nivel, visualmente organizado con Markdown y emojis (📈, 📉, ⚠️, ✅, 💡). La estructura debe ser la siguiente:
    ### Veredicto Estratégico 📜
    (En un párrafo, da un veredicto sobre la trayectoria general de la compañía. ¿La tendencia es positiva y sostenible, muestra signos de estancamiento, o hay señales de alerta preocupantes? Justifica tu conclusión.)
    ### Análisis de Evolución por Área 🔍
    (Presenta un análisis en formato de lista. Para cada área, describe la tendencia observada y su implicación estratégica.)
    - **Crecimiento y Rentabilidad:** ¿El crecimiento de los ingresos se traduce en una mayor rentabilidad (margen neto, ROE)? ¿O están creciendo los ingresos a costa de los márgenes?
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
