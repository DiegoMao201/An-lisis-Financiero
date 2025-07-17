# analisis_adicional.py
import pandas as pd
from mi_logica_original import get_principal_account_value

def calcular_analisis_vertical(df_estado_financiero: pd.DataFrame, valor_col: str, cuenta_col: str, base_cuenta: str):
    """Calcula el análisis vertical para un estado financiero."""
    if df_estado_financiero.empty or valor_col not in df_estado_financiero.columns:
        return df_estado_financiero

    df_analisis = df_estado_financiero.copy()
    
    # Asegurarse que la columna de valor sea numérica
    df_analisis[valor_col] = pd.to_numeric(df_analisis[valor_col], errors='coerce').fillna(0)

    # Obtener el valor total de la base (ej. Total Activos o Ingresos)
    valor_base = get_principal_account_value(df_analisis, base_cuenta, valor_col, cuenta_col)
    
    if valor_base == 0:
        df_analisis['Análisis Vertical (%)'] = 0.0
    else:
        # CORRECCIÓN: Calcular el porcentaje usando valores absolutos para un resultado estándar y positivo.
        df_analisis['Análisis Vertical (%)'] = (df_analisis[valor_col].abs() / abs(valor_base)) * 100
    
    return df_analisis

def calcular_analisis_horizontal(df_actual: pd.DataFrame, df_anterior: pd.DataFrame, valor_col: str, cuenta_col: str):
    """Calcula el análisis horizontal comparando dos periodos."""
    if df_actual.empty or df_anterior.empty:
        return pd.DataFrame(columns=[cuenta_col, 'Valor Actual', 'Valor Anterior', 'Variación Absoluta', 'Variación Relativa (%)'])

    # Fusionar los dataframes en la columna de la cuenta
    df_comparativo = pd.merge(
        df_actual[[cuenta_col, valor_col]],
        df_anterior[[cuenta_col, valor_col]],
        on=cuenta_col,
        suffixes=('_actual', '_anterior'),
        how='outer'
    ).fillna(0)

    # Renombrar columnas para claridad
    df_comparativo.rename(columns={
        f'{valor_col}_actual': 'Valor Actual',
        f'{valor_col}_anterior': 'Valor Anterior'
    }, inplace=True)

    # Calcular variaciones
    df_comparativo['Variación Absoluta'] = df_comparativo['Valor Actual'] - df_comparativo['Valor Anterior']
    
    # Evitar división por cero
    # El uso de .abs() en el denominador es correcto para manejar la notación.
    df_comparativo['Variación Relativa (%)'] = (
        (df_comparativo['Variación Absoluta'] / df_comparativo['Valor Anterior'].abs()) * 100
    ).replace([float('inf'), -float('inf')], 100).fillna(0)

    return df_comparativo

def construir_flujo_de_caja(df_er: pd.DataFrame, df_bg_actual: pd.DataFrame, df_bg_anterior: pd.DataFrame, val_col_er: str, cuenta_er: str, saldo_final_bg: str, cuenta_bg: str) -> pd.DataFrame:
    """Construye un estado de flujo de caja simplificado (Método Indirecto)."""
    
    # 1. Utilidad Neta (Punto de partida)
    # Asumiendo que la utilidad ya viene con el signo correcto (negativo si es ganancia)
    utilidad_neta = get_principal_account_value(df_er, '59', val_col_er, cuenta_er) # Asumiendo 59 como utilidad neta
    if utilidad_neta == 0: # Si no está la 59, la calculamos
        ingresos = get_principal_account_value(df_er, '4', val_col_er, cuenta_er)
        costos = get_principal_account_value(df_er, '6', val_col_er, cuenta_er)
        gastos_op = get_principal_account_value(df_er, '5', val_col_er, cuenta_er)
        utilidad_neta = ingresos + costos + gastos_op

    # 2. Ajustes de partidas que no son efectivo (Depreciación)
    # La depreciación es un gasto (positivo), se debe sumar para revertir el efecto.
    # En la fórmula contable estándar se suma, y aquí nuestra utilidad es negativa, por lo que el cálculo es: -Utilidad + Depreciacion
    # Por lo tanto, debemos cambiar el signo de la utilidad para el cálculo del flujo de caja.
    utilidad_para_flujo = -utilidad_neta 
    depreciacion = get_principal_account_value(df_er, '5160', val_col_er, cuenta_er) # Típicamente en gastos adm.

    # 3. Cambios en Capital de Trabajo
    def get_variacion(cuenta, df_act, df_ant):
        val_act = get_principal_account_value(df_act, cuenta, saldo_final_bg, cuenta_bg)
        val_ant = get_principal_account_value(df_ant, cuenta, saldo_final_bg, cuenta_bg)
        return val_act - val_ant # (Actual - Anterior)

    # Aumento de activo (Cuentas por cobrar) disminuye el efectivo -> Restar la variación
    var_cuentas_cobrar = get_variacion('13', df_bg_actual, df_bg_anterior)
    # Aumento de activo (Inventarios) disminuye el efectivo -> Restar la variación
    var_inventarios = get_variacion('14', df_bg_actual, df_bg_anterior)
    # Aumento de pasivo (Proveedores) aumenta el efectivo -> Sumar la variación
    var_proveedores = get_variacion('22', df_bg_actual, df_bg_anterior)

    # Flujo de Efectivo de Operaciones
    fco = utilidad_para_flujo + depreciacion - var_cuentas_cobrar - var_inventarios + var_proveedores

    # 4. Flujo de Efectivo de Inversión (simplificado)
    # Aumento de Activos Fijos es una salida de efectivo
    var_activos_fijos = get_variacion('15', df_bg_actual, df_bg_anterior) # Compra/Venta de Activos Fijos
    fci = -var_activos_fijos

    # 5. Flujo de Efectivo de Financiación (simplificado)
    # Aumento de deuda es una entrada de efectivo
    var_obligaciones = get_variacion('21', df_bg_actual, df_bg_anterior)
    # Aumento de capital es una entrada de efectivo
    var_capital_social = get_variacion('31', df_bg_actual, df_bg_anterior)
    fcf = var_obligaciones + var_capital_social

    # Flujo de caja total
    flujo_neto = fco + fci + fcf
    saldo_inicial_caja = get_principal_account_value(df_bg_anterior, '11', saldo_final_bg, cuenta_bg)
    saldo_final_caja = saldo_inicial_caja + flujo_neto

    # Crear DataFrame
    data = {
        'Concepto': [
            'Utilidad Neta del Periodo', ' (+) Depreciación y Amortización', '(-) Aumento en Cuentas por Cobrar', '(-) Aumento en Inventarios',
            '(+) Aumento en Proveedores', '**Flujo de Efectivo de Operación (FCO)**',
            '(-) Compra Neta de Activos Fijos (Inversión)', '**Flujo de Efectivo de Inversión (FCI)**',
            '(+) Aumento en Obligaciones Financieras', '(+) Aumento en Capital Social', '**Flujo de Efectivo de Financiación (FCF)**',
            '**Flujo Neto de Efectivo del Periodo**', 'Saldo Inicial de Efectivo', '**Saldo Final de Efectivo**'
        ],
        'Valor': [
            utilidad_para_flujo, depreciacion, -var_cuentas_cobrar, -var_inventarios, var_proveedores, fco,
            -var_activos_fijos, fci, var_obligaciones, var_capital_social, fcf,
            flujo_neto, saldo_inicial_caja, saldo_final_caja
        ]
    }
    df_flujo = pd.DataFrame(data)
    return df_flujo
