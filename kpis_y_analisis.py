# kpis_y_analisis.py
import pandas as pd
from mi_logica_original import get_principal_account_value, COL_CONFIG

def calcular_kpis_periodo(df_er: pd.DataFrame, df_bg: pd.DataFrame, cc_filter: str = 'Total') -> dict:
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

    if not val_col_kpi or val_col_kpi not in df_er.columns:
        return kpis # Retorna un diccionario vacío si no hay columna de valor

    cuenta_er = er_conf['CUENTA']
    ingresos = get_principal_account_value(df_er, '4', val_col_kpi, cuenta_er)
    costo_ventas = get_principal_account_value(df_er, '6', val_col_kpi, cuenta_er)
    gastos_admin = get_principal_account_value(df_er, '51', val_col_kpi, cuenta_er)
    gastos_ventas = get_principal_account_value(df_er, '52', val_col_kpi, cuenta_er)
    costos_prod = get_principal_account_value(df_er, '7', val_col_kpi, cuenta_er)
    gastos_no_op = get_principal_account_value(df_er, '53', val_col_kpi, cuenta_er)
    impuestos = get_principal_account_value(df_er, '54', val_col_kpi, cuenta_er)
    
    utilidad_bruta = ingresos + costo_ventas
    utilidad_operacional = utilidad_bruta + gastos_admin + gastos_ventas + costos_prod
    utilidad_neta = utilidad_operacional + gastos_no_op + impuestos
    
    kpis['ingresos'] = ingresos
    kpis['utilidad_bruta'] = utilidad_bruta
    kpis['utilidad_operacional'] = utilidad_operacional
    kpis['utilidad_neta'] = utilidad_neta
    
    cuenta_bg = bg_conf['CUENTA']
    saldo_final_col = bg_conf['SALDO_FINAL']
    
    activo = get_principal_account_value(df_bg, '1', saldo_final_col, cuenta_bg)
    pasivo = get_principal_account_value(df_bg, '2', saldo_final_col, cuenta_bg)
    patrimonio = get_principal_account_value(df_bg, '3', saldo_final_col, cuenta_bg)
    
    activo_corriente = sum([
        get_principal_account_value(df_bg, '11', saldo_final_col, cuenta_bg),
        get_principal_account_value(df_bg, '12', saldo_final_col, cuenta_bg),
        get_principal_account_value(df_bg, '13', saldo_final_col, cuenta_bg),
        get_principal_account_value(df_bg, '14', saldo_final_col, cuenta_bg)
    ])
    inventarios = get_principal_account_value(df_bg, '14', saldo_final_col, cuenta_bg)

    pasivo_corriente = sum([
        get_principal_account_value(df_bg, '21', saldo_final_col, cuenta_bg),
        get_principal_account_value(df_bg, '22', saldo_final_col, cuenta_bg),
        get_principal_account_value(df_bg, '23', saldo_final_col, cuenta_bg)
    ])
    
    kpis['razon_corriente'] = activo_corriente / pasivo_corriente if pasivo_corriente != 0 else 0
    kpis['prueba_acida'] = (activo_corriente - inventarios) / pasivo_corriente if pasivo_corriente != 0 else 0
    kpis['endeudamiento_activo'] = pasivo / activo if activo != 0 else 0
    kpis['apalancamiento'] = pasivo / patrimonio if patrimonio != 0 else 0
    kpis['margen_bruto'] = utilidad_bruta / ingresos if ingresos != 0 else 0
    kpis['margen_operacional'] = utilidad_operacional / ingresos if ingresos != 0 else 0
    kpis['margen_neto'] = utilidad_neta / ingresos if ingresos != 0 else 0
    kpis['roa'] = utilidad_neta / activo if activo != 0 else 0
    kpis['roe'] = utilidad_neta / patrimonio if patrimonio != 0 else 0
    
    return kpis

def preparar_datos_tendencia(datos_historicos: dict) -> pd.DataFrame:
    """Convierte el diccionario de datos históricos en un DataFrame para graficar tendencias."""
    lista_periodos = []
    for periodo, data in datos_historicos.items():
        kpis = data.get('kpis', {})
        if kpis:
            kpis['periodo'] = periodo
            lista_periodos.append(kpis)
    
    if not lista_periodos:
        return pd.DataFrame()
        
    df_tendencia = pd.DataFrame(lista_periodos)
    df_tendencia['periodo'] = pd.to_datetime(df_tendencia['periodo'], format='%Y-%m')
    df_tendencia = df_tendencia.sort_values(by='periodo').reset_index(drop=True)
    return df_tendencia

def generar_comentario_kpi(kpi_nombre: str, valor: float) -> str:
    """Genera un comentario básico para un KPI."""
    if pd.isna(valor):
        return f"No hay datos disponibles para {kpi_nombre}."
    elif kpi_nombre == 'margen_neto':
        return f"El margen neto actual es del {valor:.2%}, lo que indica la rentabilidad después de todos los costos e impuestos."
    elif kpi_nombre == 'razon_corriente':
        if valor > 1.5:
            return f"La razón corriente es {valor:.2f}, sugiriendo una buena capacidad para cubrir las obligaciones a corto plazo."
        elif valor < 1:
            return f"La razón corriente es {valor:.2f}, lo que podría indicar desafíos para cubrir las obligaciones a corto plazo."
        else:
            return f"La razón corriente es {valor:.2f}, un nivel moderado de liquidez a corto plazo."
    elif kpi_nombre == 'roe':
        return f"El retorno sobre el patrimonio (ROE) es del {valor:.2%}, reflejando la eficiencia con la que se utiliza el capital de los accionistas para generar ganancias."
    elif kpi_nombre == 'endeudamiento_activo':
        return f"El endeudamiento sobre el activo total es del {valor:.2%}, mostrando la proporción de activos financiados por deuda."
    else:
        return f"El valor de {kpi_nombre} es {valor:.2f}."
