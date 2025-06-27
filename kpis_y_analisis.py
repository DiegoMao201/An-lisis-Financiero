# kpis_y_analisis.py
import pandas as pd
import streamlit as st
import google.generativeai as genai
from mi_logica_original import get_principal_account_value, COL_CONFIG

def calcular_kpis_periodo(df_er: pd.DataFrame, df_bg: pd.DataFrame, cc_filter: str = 'Todos') -> dict:
    """Calcula un set de KPIs para un único periodo, opcionalmente filtrado por centro de costo."""
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
        # Lógica para encontrar la columna de valor consolidado
        ind_cc_cols = [v for k, v in er_conf.get('CENTROS_COSTO_COLS',{}).items() if str(k).lower() not in ['total', 'sin centro de coste'] and v in df_er.columns]
        if ind_cc_cols:
            df_er['__temp_sum_kpi'] = df_er.loc[:, ind_cc_cols].sum(axis=1)
            val_col_kpi = '__temp_sum_kpi'
        else: # Fallback si no hay columnas de CC individuales
            scc_name = er_conf.get('CENTROS_COSTO_COLS',{}).get('Sin centro de coste', 'Total_Consolidado_ER')
            if scc_name and scc_name in df_er.columns:
                val_col_kpi = scc_name

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
    Genera un análisis financiero profundo utilizando el modelo Gemini de Google.
    """
    # Configuración de la API Key
    try:
        api_key = st.secrets["google_ai"]["api_key"]
        genai.configure(api_key=api_key)
    except Exception:
        return "🔴 **Error:** No se encontró la clave de API de Google AI en los secretos de Streamlit. Por favor, configúrala."

    # Extraer los 5 gastos operativos más significativos
    er_conf = COL_CONFIG['ESTADO_DE_RESULTADOS']
    gastos_df = _df_er_actual[_df_er_actual[er_conf['CUENTA']].str.startswith(('5', '7'))].copy()
    
    val_col = ''
    if nombre_cc != 'Todos' and nombre_cc in gastos_df.columns:
        val_col = nombre_cc
    else: # Consolidado
        ind_cc_cols = [v for k, v in er_conf.get('CENTROS_COSTO_COLS',{}).items() if str(k).lower() not in ['total', 'sin centro de coste'] and v in gastos_df.columns]
        if ind_cc_cols:
            gastos_df['__temp_sum_gastos'] = gastos_df[ind_cc_cols].sum(axis=1)
            val_col = '__temp_sum_gastos'
    
    top_5_gastos_str = "No se pudieron determinar los gastos principales."
    if val_col:
        gastos_df = gastos_df[[er_conf['NOMBRE_CUENTA'], val_col]].dropna()
        gastos_df[val_col] = pd.to_numeric(gastos_df[val_col], errors='coerce').abs()
        top_5_gastos = gastos_df.nlargest(5, val_col)
        top_5_gastos_str = "\n".join([f"- {row[er_conf['NOMBRE_CUENTA']]}: ${row[val_col]:,.0f}" for _, row in top_5_gastos.iterrows()])

    # --- INGENIERÍA DE PROMPT: La clave de la IA ---
    prompt = f"""
    Actúa como un Director Financiero (CFO) experto y un asesor de negocios conciso y directo.
    Estás analizando los resultados del centro de costo: "{nombre_cc}" para el periodo: "{periodo_actual}".

    Aquí están los datos financieros clave:
    - **Ingresos:** ${_kpis_actuales.get('ingresos', 0):,.0f}
    - **Utilidad Neta:** ${_kpis_actuales.get('utilidad_neta', 0):,.0f}
    - **Margen Neto:** {_kpis_actuales.get('margen_neto', 0):.2%}
    - **Rentabilidad sobre Patrimonio (ROE):** {_kpis_actuales.get('roe', 0):.2%}
    - **Razón Corriente (Liquidez):** {_kpis_actuales.get('razon_corriente', 0):.2f}
    - **Nivel de Endeudamiento (sobre Activo):** {_kpis_actuales.get('endeudamiento_activo', 0):.2%}
    - **Gastos Operativos Totales:** ${_kpis_actuales.get('gastos_operativos', 0):,.0f}

    Los 5 gastos operativos más significativos fueron:
    {top_5_gastos_str}

    Basado **exclusivamente** en estos datos, proporciona un análisis estructurado en 3 secciones con formato Markdown:

    ### 📈 Resumen Ejecutivo
    (Un párrafo corto que resuma la salud financiera general del centro de costo en este periodo: rentable, en problemas, estable, etc.)

    ### 🎯 Diagnóstico de Puntos Clave
    (Una lista de 3 a 4 puntos usando viñetas. Analiza la rentabilidad, la estructura de costos y la liquidez. Sé directo y menciona tanto los puntos fuertes como las áreas de preocupación. Por ejemplo: "El margen neto es saludable, pero...", "La liquidez es ajustada...", "Los gastos de personal representan una parte significativa de los costos...")

    ### 💡 Consejos y Pasos a Seguir
    (Una lista de 2 a 3 recomendaciones **accionables y específicas** basadas en el diagnóstico. Por ejemplo: "Revisar a fondo la cuenta 'Gastos de Personal' para identificar posibles optimizaciones.", "Implementar una estrategia para mejorar el margen bruto, negociando con proveedores o ajustando precios.", "Monitorear de cerca el flujo de caja debido a la ajustada razón corriente.")
    """

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"🔴 **Error al contactar la IA:** {e}. Revisa la API Key y la configuración."
