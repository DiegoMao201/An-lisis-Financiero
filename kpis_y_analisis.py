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
        total_col_name = er_conf.get('CENTROS_COSTO_COLS',{}).get('Total')
        if total_col_name and total_col_name in df_er.columns:
            val_col_kpi = total_col_name
        else:
            ind_cc_cols = [v for k, v in er_conf.get('CENTROS_COSTO_COLS',{}).items() if str(k).lower() not in ['total', 'sin centro de coste'] and v in df_er.columns]
            if ind_cc_cols:
                df_er['__temp_sum_kpi'] = df_er.loc[:, ind_cc_cols].sum(axis=1)
                val_col_kpi = '__temp_sum_kpi'
            else:
                scc_name = er_conf.get('CENTROS_COSTO_COLS',{}).get('Sin centro de coste')
                if scc_name and scc_name in df_er.columns:
                    val_col_kpi = scc_name
                elif 'Total_Consolidado_ER' in df_er.columns: # Añadido fallback
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
    Genera un análisis financiero profundo utilizando el modelo Gemini de Google.
    """
    try:
        api_key = st.secrets["google_ai"]["api_key"]
        genai.configure(api_key=api_key)
    except Exception:
        return "🔴 **Error:** No se encontró la clave de API de Google AI."

    er_conf = COL_CONFIG['ESTADO_DE_RESULTADOS']
    gastos_df = _df_er_actual[_df_er_actual.get(er_conf['CUENTA'], '').str.startswith(('5', '7'), na=False)].copy()

    val_col = ''
    if nombre_cc != 'Todos' and nombre_cc in gastos_df.columns:
        val_col = nombre_cc
    elif 'Total_Consolidado_ER' in gastos_df.columns:
        val_col = 'Total_Consolidado_ER'
    else:
        ind_cc_cols = [v for k, v in er_conf.get('CENTROS_COSTO_COLS',{}).items() if str(k).lower() not in ['total', 'sin centro de coste'] and v in gastos_df.columns]
        if ind_cc_cols:
            gastos_df['__temp_sum_gastos'] = gastos_df.loc[:, ind_cc_cols].sum(axis=1)
            val_col = '__temp_sum_gastos'

    top_5_gastos_str = "No se pudieron determinar los gastos principales."
    if val_col and er_conf['NOMBRE_CUENTA'] in gastos_df.columns and val_col in gastos_df.columns:
        gastos_df_filtered = gastos_df[[er_conf['NOMBRE_CUENTA'], val_col]].dropna()
        gastos_df_filtered.loc[:, val_col] = pd.to_numeric(gastos_df_filtered.loc[:, val_col], errors='coerce').abs()
        top_5_gastos = gastos_df_filtered.nlargest(5, val_col)
        top_5_gastos_str = "\n".join([f"- {row.iloc[:, 0]}: ${row.iloc[:, 1]:,.0f}" for _, row in top_5_gastos.iterrows()])

    prompt = f"""
    Actúa como un Director Financiero (CFO) experto. Analiza los resultados de "{nombre_cc}" para "{periodo_actual}".

    Datos clave:
    - Ingresos: ${_kpis_actuales.get('ingresos', 0):,.0f}
    - Utilidad Neta: ${_kpis_actuales.get('utilidad_neta', 0):,.0f}
    - Margen Neto: {_kpis_actuales.get('margen_neto', 0):.2%}
    - ROE: {_kpis_actuales.get('roe', 0):.2%}
    - Razón Corriente: {_kpis_actuales.get('razon_corriente', 0):.2f}
    - Endeudamiento (Activo): {_kpis_actuales.get('endeudamiento_activo', 0):.2%}
    - Gastos Operativos Totales: ${_kpis_actuales.get('gastos_operativos', 0):,.0f}
    Gastos principales:
    {top_5_gastos_str}

    Genera un análisis CFO en 3 secciones (Resumen, Diagnóstico, Consejos) con formato Markdown, basado **solo** en estos datos.
    """

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"🔴 **Error al contactar la IA:** {e}."

def generar_lista_cuentas(df: pd.DataFrame, nivel_col: str = 'Grupo', nivel: int = 1) -> list:
    """Genera una lista de cuentas filtradas por nivel."""
    if nivel_col in df.columns:
        return sorted(df.loc[(df.iloc[:, df.columns.get_loc(nivel_col)]) <= nivel, COL_CONFIG['ESTADO_DE_RESULTADOS']['NOMBRE_CUENTA']].unique())
    return []
