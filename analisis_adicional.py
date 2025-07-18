# analisis_adicional.py
import pandas as pd
from mi_logica_original import get_principal_account_value

def calcular_analisis_vertical(df_estado_financiero: pd.DataFrame, valor_col: str, cuenta_col: str, base_cuenta: str):
    """Calcula el análisis vertical para un estado financiero."""
    if df_estado_financiero.empty or valor_col not in df_estado_financiero.columns:
        return df_estado_financiero

    df_analisis = df_estado_financiero.copy()
    
    df_analisis[valor_col] = pd.to_numeric(df_analisis[valor_col], errors='coerce').fillna(0)

    valor_base = get_principal_account_value(df_analisis, base_cuenta, valor_col, cuenta_col)
    
    if valor_base == 0:
        df_analisis['Análisis Vertical (%)'] = 0.0
    else:
        # CORRECCIÓN: Usar el valor absoluto de ambos para obtener un % estándar y positivo.
        # Esto evita confusiones en los reportes tabulares.
        df_analisis['Análisis Vertical (%)'] = (df_analisis[valor_col].abs() / abs(valor_base)) * 100
    
    return df_analisis

def calcular_analisis_horizontal(df_actual: pd.DataFrame, df_anterior: pd.DataFrame, valor_col: str, cuenta_col: str):
    """Calcula el análisis horizontal comparando dos periodos."""
    if df_actual.empty or df_anterior.empty:
        return pd.DataFrame(columns=[cuenta_col, 'Valor Actual', 'Valor Anterior', 'Variación Absoluta', 'Variación Relativa (%)'])

    df_comparativo = pd.merge(
        df_actual[[cuenta_col, valor_col]],
        df_anterior[[cuenta_col, valor_col]],
        on=cuenta_col,
        suffixes=('_actual', '_anterior'),
        how='outer'
    ).fillna(0)

    df_comparativo.rename(columns={
        f'{valor_col}_actual': 'Valor Actual',
        f'{valor_col}_anterior': 'Valor Anterior'
    }, inplace=True)

    # El cálculo de la variación absoluta es correcto.
    # Ej: -239M (actual) - (-71M) (anterior) = -168M (una mejora/crecimiento)
    df_comparativo['Variación Absoluta'] = df_comparativo['Valor Actual'] - df_comparativo['Valor Anterior']
    
    # El cálculo de la variación relativa también es matemáticamente consistente.
    # El signo negativo (-263%) indica una mejora/crecimiento según la lógica. La IA ahora está instruida para interpretarlo así.
    df_comparativo['Variación Relativa (%)'] = (
        (df_comparativo['Variación Absoluta'] / df_comparativo['Valor Anterior'].abs()) * 100
    ).replace([float('inf'), -float('inf')], 100).fillna(0)

    return df_comparativo

def construir_flujo_de_caja(df_er: pd.DataFrame, df_bg_actual: pd.DataFrame, df_bg_anterior: pd.DataFrame, val_col_er: str, cuenta_er: str, saldo_final_bg: str, cuenta_bg: str) -> pd.DataFrame:
    """Construye un estado de flujo de caja simplificado (Método Indirecto)."""
    
    utilidad_neta = get_principal_account_value(df_er, '59', val_col_er, cuenta_er) 
    if utilidad_neta == 0:
        ingresos = get_principal_account_value(df_er, '4', val_col_er, cuenta_er)
        costos = get_principal_account_value(df_er, '6', val_col_er, cuenta_er)
        gastos_op = get_principal_account_value(df_er, '5', val_col_er, cuenta_er)
        utilidad_neta = ingresos + costos + gastos_op

    # Para el Flujo de Caja (método indirecto), se parte de la utilidad y se le cambia el signo para trabajar con valores positivos.
    utilidad_para_flujo = -utilidad_neta 
    depreciacion = get_principal_account_value(df_er, '5160', val_col_er, cuenta_er) 

    def get_variacion(cuenta, df_act, df_ant):
        val_act = get_principal_account_value(df_act, cuenta, saldo_final_bg, cuenta_bg)
        val_ant = get_principal_account_value(df_ant, cuenta, saldo_final_bg, cuenta_bg)
        return val_act - val_ant

    var_cuentas_cobrar = get_variacion('13', df_bg_actual, df_bg_anterior)
    var_inventarios = get_variacion('14', df_bg_actual, df_bg_anterior)
    var_proveedores = get_variacion('22', df_bg_actual, df_bg_anterior)

    fco = utilidad_para_flujo + depreciacion - var_cuentas_cobrar - var_inventarios + var_proveedores

    var_activos_fijos = get_variacion('15', df_bg_actual, df_bg_anterior)
    fci = -var_activos_fijos

    var_obligaciones = get_variacion('21', df_bg_actual, df_bg_anterior)
    var_capital_social = get_variacion('31', df_bg_actual, df_bg_anterior)
    fcf = var_obligaciones + var_capital_social

    flujo_neto = fco + fci + fcf
    saldo_inicial_caja = get_principal_account_value(df_bg_anterior, '11', saldo_final_bg, cuenta_bg)
    saldo_final_caja = saldo_inicial_caja + flujo_neto

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
