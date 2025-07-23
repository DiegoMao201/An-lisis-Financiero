# analisis_adicional.py
# Archivo completo con correcciones lógicas en el Flujo de Caja.

import pandas as pd
from mi_logica_original import get_principal_account_value

def calcular_analisis_vertical(df_estado_financiero: pd.DataFrame, valor_col: str, cuenta_col: str, base_cuenta: str):
    """
    Calcula el análisis vertical para un estado financiero.
    Esta función es LÓGICAMENTE CORRECTA.
    """
    if df_estado_financiero.empty or valor_col not in df_estado_financiero.columns:
        return df_estado_financiero

    df_analisis = df_estado_financiero.copy()
    
    df_analisis[valor_col] = pd.to_numeric(df_analisis[valor_col], errors='coerce').fillna(0)

    # El valor base puede ser positivo (Activo, Ingreso) o negativo (Pasivo, Patrimonio)
    valor_base = get_principal_account_value(df_analisis, base_cuenta, valor_col, cuenta_col)
    
    if valor_base == 0:
        df_analisis['Análisis Vertical (%)'] = 0.0
    else:
        # Se usa el valor absoluto de ambos para obtener un % de magnitud estándar y positivo.
        # Esto es correcto para la presentación de reportes.
        df_analisis['Análisis Vertical (%)'] = (df_analisis[valor_col].abs() / abs(valor_base)) * 100
    
    return df_analisis

def calcular_analisis_horizontal(df_actual: pd.DataFrame, df_anterior: pd.DataFrame, valor_col: str, cuenta_col: str):
    """
    Calcula el análisis horizontal comparando dos periodos.
    Esta función es LÓGICAMENTE CORRECTA.
    """
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

    # La variación absoluta respeta los signos, lo cual es correcto.
    # Ej: Un pasivo que aumenta de -100 a -120 tiene una variación de -20.
    df_comparativo['Variación Absoluta'] = df_comparativo['Valor Actual'] - df_comparativo['Valor Anterior']
    
    # La variación relativa usa el absoluto del valor anterior como base, lo cual es el método estándar.
    df_comparativo['Variación Relativa (%)'] = (
        (df_comparativo['Variación Absoluta'] / df_comparativo['Valor Anterior'].abs()) * 100
    ).replace([float('inf'), -float('inf')], 100).fillna(0)

    return df_comparativo

def construir_flujo_de_caja(df_er: pd.DataFrame, df_bg_actual: pd.DataFrame, df_bg_anterior: pd.DataFrame, val_col_er: str, cuenta_er: str, saldo_final_bg: str, cuenta_bg: str) -> pd.DataFrame:
    """
    Construye un estado de flujo de caja simplificado (Método Indirecto).
    *** FUNCIÓN COMPLETAMENTE CORREGIDA PARA LÓGICA MIXTA ***
    """
    
    # --- 1. Punto de Partida: Utilidad Neta (con su signo original) ---
    # El método indirecto PARTE de la utilidad neta. No se cambia el signo.
    # Si la utilidad es positiva (ganancia), se suma. Si es negativa (pérdida), se resta.
    utilidad_neta = get_principal_account_value(df_er, '59', val_col_er, cuenta_er)
    if utilidad_neta == 0: # Fallback por si la cuenta 59 no existe
        ingresos = get_principal_account_value(df_er, '4', val_col_er, cuenta_er)
        costos = get_principal_account_value(df_er, '6', val_col_er, cuenta_er)
        gastos_op = get_principal_account_value(df_er, '5', val_col_er, cuenta_er)
        utilidad_neta = ingresos + costos + gastos_op

    # --- 2. Ajustes para llegar al Flujo de Operación (FCO) ---
    
    # La depreciación es un gasto no monetario. Se suma de vuelta.
    # Como viene con signo negativo del P&L, usamos abs() para sumarla.
    depreciacion = get_principal_account_value(df_er, '5160', val_col_er, cuenta_er)
    ajuste_depreciacion = abs(depreciacion)

    # Función auxiliar para calcular variaciones del Balance General
    def get_variacion(cuenta, df_act, df_ant):
        val_act = get_principal_account_value(df_act, cuenta, saldo_final_bg, cuenta_bg)
        val_ant = get_principal_account_value(df_ant, cuenta, saldo_final_bg, cuenta_bg)
        return val_act - val_ant

    # Variaciones en Activos de Operación:
    # Un aumento en un activo (ej. Cuentas por Cobrar) es un USO de efectivo. Se resta.
    var_cuentas_cobrar = get_variacion('13', df_bg_actual, df_bg_anterior)
    var_inventarios = get_variacion('14', df_bg_actual, df_bg_anterior)

    # Variaciones en Pasivos de Operación:
    # Un aumento en un pasivo (ej. Proveedores) es una FUENTE de efectivo. Se suma.
    # CORRECCIÓN: Como la variación es negativa (ej. -20), debemos RESTARLA para que se sume. (- (-20) = +20)
    var_proveedores = get_variacion('22', df_bg_actual, df_bg_anterior)

    fco = (utilidad_neta + 
           ajuste_depreciacion - 
           var_cuentas_cobrar - 
           var_inventarios - # Restamos la variación de pasivos
           var_proveedores)

    # --- 3. Flujo de Inversión (FCI) ---
    # Un aumento en Activos Fijos es una compra (USO de efectivo). Se resta.
    var_activos_fijos = get_variacion('15', df_bg_actual, df_bg_anterior)
    fci = -var_activos_fijos

    # --- 4. Flujo de Financiación (FCF) ---
    # Un aumento en Deuda o Capital es una FUENTE de efectivo. Se suma.
    # CORRECCIÓN: Misma lógica que con proveedores. Se resta la variación negativa.
    var_obligaciones = get_variacion('21', df_bg_actual, df_bg_anterior)
    var_capital_social = get_variacion('31', df_bg_actual, df_bg_anterior)
    fcf = -var_obligaciones - var_capital_social

    # --- 5. Cálculo Final del Flujo y Saldos de Caja ---
    flujo_neto = fco + fci + fcf
    saldo_inicial_caja = get_principal_account_value(df_bg_anterior, '11', saldo_final_bg, cuenta_bg)
    saldo_final_caja_calculado = saldo_inicial_caja + flujo_neto
    saldo_final_caja_real = get_principal_account_value(df_bg_actual, '11', saldo_final_bg, cuenta_bg)
    diferencia_caja = saldo_final_caja_real - saldo_final_caja_calculado # Debería ser cercano a cero

    # --- 6. Construcción del DataFrame para Visualización ---
    data = {
        'Concepto': [
            'Utilidad Neta del Periodo',
            '(+) Depreciación y Amortización',
            '(-) Aumento en Cuentas por Cobrar',
            '(-) Aumento en Inventarios',
            '(+) Aumento en Proveedores',
            '**Flujo de Efectivo de Operación (FCO)**',
            '(-) Compra Neta de Activos Fijos (Inversión)',
            '**Flujo de Efectivo de Inversión (FCI)**',
            '(+) Aumento en Obligaciones Financieras',
            '(+) Aumento en Capital Social',
            '**Flujo de Efectivo de Financiación (FCF)**',
            '**Flujo Neto de Efectivo del Periodo**',
            'Saldo Inicial de Efectivo',
            '**Saldo Final de Efectivo (Calculado)**',
            'Saldo Final de Efectivo (Real en Balance)',
            '*Diferencia (Prueba de Cuadre)*'
        ],
        'Valor': [
            utilidad_neta,
            ajuste_depreciacion,
            -var_cuentas_cobrar,
            -var_inventarios,
            -var_proveedores, # Se muestra como positivo (fuente), el cálculo ya lo invirtió
            fco,
            -var_activos_fijos, # Se muestra como positivo (uso), el cálculo ya lo invirtió
            fci,
            -var_obligaciones, # Se muestra como positivo (fuente)
            -var_capital_social, # Se muestra como positivo (fuente)
            fcf,
            flujo_neto,
            saldo_inicial_caja,
            saldo_final_caja_calculado,
            saldo_final_caja_real,
            diferencia_caja
        ]
    }
    df_flujo = pd.DataFrame(data)
    return df_flujo
