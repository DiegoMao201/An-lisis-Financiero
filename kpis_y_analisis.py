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
