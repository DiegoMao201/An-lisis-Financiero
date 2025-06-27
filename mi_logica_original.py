# mi_logica_original.py
import pandas as pd
import numpy as np
import io

# ==============================================================================
#                      CONFIGURACIÓN Y FUNCIONES BASE
# ==============================================================================

# Tu variable de configuración global
COL_CONFIG = {
    'ESTADO_DE_RESULTADOS': {
        'NIVEL_LINEA': 'Grupo', 'CUENTA': 'Cuenta', 'NOMBRE_CUENTA': 'Título',
        'CENTROS_COSTO_COLS': {
            'Sin centro de coste': 'Sin centro de coste', 156: 'Armenia', 157: 'San antonio',
            158: 'Opalo', 189: 'Olaya', 238: 'Laureles', 'Total': 'Total_Consolidado_ER'
        }
    },
    'BALANCE_GENERAL': {
        'NIVEL_LINEA': 'Grupo', 'CUENTA': 'Cuenta', 'NOMBRE_CUENTA': 'Título',
        'SALDO_INICIAL': 'Saldo inicial', 'DEBE': 'Debe', 'HABER': 'Haber', 'SALDO_FINAL': 'Saldo Final'
    }
}

# Tus funciones de utilidad y procesamiento (sin cambios)
def clean_numeric_value(value):
    """Limpia y convierte un valor a float, manejando comas/puntos como decimales."""
    if pd.isna(value) or value == '': return 0.0
    s_value = str(value).strip().replace('.', '').replace(',', '.')
    try: return float(s_value)
    except ValueError: return 0.0

def classify_account(cuenta_str: str) -> str:
    """Clasifica una cuenta contable en su tipo de estado financiero."""
    if not isinstance(cuenta_str, str): cuenta_str = str(cuenta_str)
    cuenta_str = cuenta_str.strip()
    if not cuenta_str: return 'No Clasificado'
    if cuenta_str.startswith('1'): return 'Balance General - Activos'
    elif cuenta_str.startswith('2'): return 'Balance General - Pasivos'
    elif cuenta_str.startswith('3'): return 'Balance General - Patrimonio'
    elif cuenta_str.startswith('4'): return 'Estado de Resultados - Ingresos'
    elif cuenta_str.startswith('5'): return 'Estado de Resultados - Gastos'
    elif cuenta_str.startswith('6'): return 'Estado de Resultados - Costos de Ventas'
    elif cuenta_str.startswith('7'): return 'Estado de Resultados - Costos de Produccion o de Operacion'
    else: return 'No Clasificado'

def get_principal_account_value(df: pd.DataFrame, principal_account_code: str, value_column: str, cuenta_column_name: str):
    """Obtiene el valor de una cuenta principal (ej., '4' para ingresos) de un DataFrame."""
    if cuenta_column_name not in df.columns or value_column not in df.columns: return 0.0
    principal_row = df[df[cuenta_column_name].astype(str) == str(principal_account_code)]
    if not principal_row.empty:
        raw_value = principal_row[value_column].iloc[0]
        numeric_value = pd.to_numeric(raw_value, errors='coerce')
        return 0.0 if pd.isna(numeric_value) else float(numeric_value)
    return 0.0

def get_top_level_accounts_for_display(df_raw: pd.DataFrame, value_col_name: str, statement_type: str) -> pd.DataFrame:
    """Filtra y prepara las cuentas de alto nivel para su visualización."""
    # ... (CÓDIGO EXACTO DE TU FUNCIÓN) ...
    # Esta función es larga, así que la represento aquí, pero debes pegar tu código completo.
    # Por brevedad, se omite el cuerpo completo aquí, pero en tu archivo debe estar completo.
    config_key = statement_type.replace(' ', '_').upper()
    if config_key not in COL_CONFIG: return pd.DataFrame()
    config_specific = COL_CONFIG[config_key]
    default_cols = {'CUENTA': 'Cuenta', 'NOMBRE_CUENTA': 'Título', 'NIVEL_LINEA': 'Grupo'}

    cuenta_col = config_specific.get('CUENTA', default_cols['CUENTA'])
    nombre_cuenta_col = config_specific.get('NOMBRE_CUENTA', default_cols['NOMBRE_CUENTA'])
    nivel_linea_col = config_specific.get('NIVEL_LINEA', default_cols['NIVEL_LINEA'])

    if value_col_name not in df_raw.columns: df_raw[value_col_name] = 0.0
    required_cols = [cuenta_col, nombre_cuenta_col, nivel_linea_col, value_col_name]
    if not all(col in df_raw.columns for col in required_cols): return pd.DataFrame()

    df_processed = df_raw.copy()
    df_processed[cuenta_col] = df_processed[cuenta_col].astype(str).str.strip()
    df_processed['Cuenta_Str'] = df_processed[cuenta_col]

    df_sorted = df_processed.sort_values(by='Cuenta_Str').reset_index(drop=True)
    df_sorted = df_sorted.dropna(subset=['Cuenta_Str', nombre_cuenta_col, value_col_name])
    df_sorted = df_sorted[df_sorted['Cuenta_Str'] != ''].reset_index(drop=True)
    df_sorted = df_sorted[df_sorted[nombre_cuenta_col].astype(str).str.strip() != ''].reset_index(drop=True)
    df_sorted[value_col_name] = pd.to_numeric(df_sorted[value_col_name], errors='coerce').fillna(0.0)

    df_significant_values = df_sorted[df_sorted[value_col_name].abs() > 0.001].copy()
    unique_values_for_filter = df_significant_values[value_col_name].unique() if not df_significant_values.empty else []
    selected_rows_list = []
    for val_filter in unique_values_for_filter:
        group = df_significant_values[df_significant_values[value_col_name] == val_filter]
        if not group.empty:
            if 'Cuenta_Str' in group.columns:
                selected_rows_list.append(group.loc[group['Cuenta_Str'].str.len().idxmin()])
            elif cuenta_col in group.columns:
                selected_rows_list.append(group.loc[group[cuenta_col].astype(str).str.len().idxmin()])

    df_result = pd.DataFrame(selected_rows_list).drop_duplicates(subset=['Cuenta_Str']).reset_index(drop=True) if selected_rows_list else pd.DataFrame(columns=df_sorted.columns)

    principal_levels_for_zero_values = [1]
    df_sorted[nivel_linea_col] = pd.to_numeric(df_sorted[nivel_linea_col], errors='coerce')
    df_zero_sig = df_sorted[
        (df_sorted[value_col_name].abs() < 0.001) &
        (df_sorted[nivel_linea_col].notna()) &
        (df_sorted[nivel_linea_col].isin(principal_levels_for_zero_values))
    ].copy()

    df_final = pd.concat([df_result, df_zero_sig]).drop_duplicates(subset=['Cuenta_Str']).reset_index(drop=True)
    if df_final.empty: return df_final
    df_final[nivel_linea_col] = pd.to_numeric(df_final[nivel_linea_col], errors='coerce').fillna(99)
    return df_final.sort_values(by=[nivel_linea_col, 'Cuenta_Str'])


def generate_financial_statement(df_full_data: pd.DataFrame, statement_type: str, selected_cc_filter: str = None, max_level: int = 999) -> pd.DataFrame:
    """Genera un estado financiero (ER o BG) procesado para visualización."""
    # ... (CÓDIGO EXACTO DE TU FUNCIÓN) ...
    # Esta función es larga, así que la represento aquí, pero debes pegar tu código completo.
    # Por brevedad, se omite el cuerpo completo aquí, pero en tu archivo debe estar completo.
    config_key = statement_type.replace(' ', '_').upper()
    if config_key not in COL_CONFIG: return pd.DataFrame()
    config = COL_CONFIG[config_key]
    default_col_names = {'CUENTA': 'Cuenta', 'NOMBRE_CUENTA': 'Título', 'NIVEL_LINEA': 'Grupo'}
    cuenta_col = config.get('CUENTA', default_col_names['CUENTA'])
    nombre_col = config.get('NOMBRE_CUENTA', default_col_names['NOMBRE_CUENTA'])
    nivel_col = config.get('NIVEL_LINEA', default_col_names['NIVEL_LINEA'])

    final_cols = [cuenta_col, nombre_col, 'Valor']
    base_check = [cuenta_col, nombre_col, nivel_col, 'Tipo_Estado']
    if not all(col in df_full_data.columns for col in base_check):
        return pd.DataFrame(columns=final_cols)

    if statement_type == 'Estado de Resultados':
        df_statement_orig = df_full_data[df_full_data['Tipo_Estado'].str.contains('Estado de Resultados', na=False)].copy()
        if df_statement_orig.empty: return pd.DataFrame(columns=final_cols)

        value_col_to_use = ''
        if selected_cc_filter and selected_cc_filter != 'Todos':
            if selected_cc_filter in df_statement_orig.columns: value_col_to_use = selected_cc_filter
            else: df_statement_orig['Valor_Final_Temp_CC'] = 0.0; value_col_to_use = 'Valor_Final_Temp_CC'
        else:
            total_er_cfg_col = config.get('CENTROS_COSTO_COLS',{}).get('Total')
            if total_er_cfg_col and total_er_cfg_col in df_statement_orig.columns: value_col_to_use = total_er_cfg_col
            else:
                cc_ind_cols_list = [ v for k, v in config.get('CENTROS_COSTO_COLS',{}).items() if str(k).lower() not in ['total', 'sin centro de coste'] and v in df_statement_orig.columns and v != total_er_cfg_col]
                if cc_ind_cols_list:
                    for c_col in cc_ind_cols_list: df_statement_orig[c_col] = pd.to_numeric(df_statement_orig[c_col], errors='coerce').fillna(0)
                    df_statement_orig['__temp_sum_for_gfs'] = df_statement_orig[cc_ind_cols_list].sum(axis=1)
                    value_col_to_use = '__temp_sum_for_gfs'
                else:
                    scc_cfg_col = config.get('CENTROS_COSTO_COLS',{}).get('Sin centro de coste')
                    if scc_cfg_col and scc_cfg_col in df_statement_orig.columns: value_col_to_use = scc_cfg_col
                    else: df_statement_orig['Valor_Final_Temp_Total'] = 0.0; value_col_to_use = 'Valor_Final_Temp_Total'

        if value_col_to_use not in df_statement_orig.columns: df_statement_orig[value_col_to_use] = 0.0
        df_statement_orig['Valor_Final'] = pd.to_numeric(df_statement_orig[value_col_to_use], errors='coerce').fillna(0)

        df_display = get_top_level_accounts_for_display(df_statement_orig, 'Valor_Final', statement_type)
        if df_display.empty: return pd.DataFrame(columns=final_cols)

        if nivel_col not in df_display.columns: df_display[nivel_col] = 1
        df_display[nivel_col] = pd.to_numeric(df_display[nivel_col], errors='coerce').fillna(9999)
        df_display = df_display[df_display[nivel_col] <= float(max_level)].copy()

        er_categories_ordered = ['Estado de Resultados - Ingresos', 'Estado de Resultados - Costos de Ventas', 'Estado de Resultados - Gastos', 'Estado de Resultados - Costos de Produccion o de Operacion']
        processed_final_df = pd.DataFrame(columns=final_cols)
        required_loop_cols = ['Tipo_Estado', cuenta_col, nombre_col, nivel_col, 'Valor_Final']
        if not all(col_loop in df_display.columns for col_loop in required_loop_cols): return processed_final_df

        for tipo_estado_categoria in er_categories_ordered:
            group = df_display[df_display['Tipo_Estado'] == tipo_estado_categoria].copy()
            if not group.empty:
                group = group.sort_values(by=cuenta_col)
                group[nivel_col] = pd.to_numeric(group[nivel_col], errors='coerce').fillna(1).astype(int)
                group['Nombre_Cuenta_Display'] = group.apply(lambda r: f"{' ' * (r[nivel_col] - 1)}{r[nombre_col]}", axis=1)
                processed_final_df = pd.concat([processed_final_df, group[[cuenta_col, 'Nombre_Cuenta_Display', 'Valor_Final']].rename(columns={'Nombre_Cuenta_Display': nombre_col, 'Valor_Final': 'Valor'})], ignore_index=True)

        df_statement_orig[cuenta_col] = df_statement_orig[cuenta_col].astype(str)

        _ing_calc_gfs = get_principal_account_value(df_statement_orig, '4', 'Valor_Final', cuenta_col)
        _cv_calc_gfs = get_principal_account_value(df_statement_orig, '6', 'Valor_Final', cuenta_col)
        _go_admin_calc_gfs = get_principal_account_value(df_statement_orig, '51', 'Valor_Final', cuenta_col)
        _go_ventas_calc_gfs = get_principal_account_value(df_statement_orig, '52', 'Valor_Final', cuenta_col)
        _cost_prod_op_calc_gfs = get_principal_account_value(df_statement_orig, '7', 'Valor_Final', cuenta_col)
        _go_total_calc_gfs = _go_admin_calc_gfs + _go_ventas_calc_gfs + _cost_prod_op_calc_gfs
        _gno_calc_gfs = get_principal_account_value(df_statement_orig, '53', 'Valor_Final', cuenta_col)
        _imp_calc_gfs = get_principal_account_value(df_statement_orig, '54', 'Valor_Final', cuenta_col)
        _uo_calc_tabla_gfs = _ing_calc_gfs + _cv_calc_gfs + _go_total_calc_gfs
        total_val_er_correct = _uo_calc_tabla_gfs + _gno_calc_gfs + _imp_calc_gfs

        total_row = pd.DataFrame([{cuenta_col: '', nombre_col: 'TOTAL ESTADO DE RESULTADOS', 'Valor': total_val_er_correct}])
        processed_final_df = pd.concat([processed_final_df, total_row], ignore_index=True)
        blank_row = pd.DataFrame([{cuenta_col: '', nombre_col: '', 'Valor': None}])
        processed_final_df = pd.concat([processed_final_df, blank_row], ignore_index=True)
        return processed_final_df

    elif statement_type == 'Balance General':
        df_statement_bg = df_full_data[df_full_data['Tipo_Estado'].str.contains('Balance General', na=False)].copy()
        if df_statement_bg.empty: return pd.DataFrame(columns=final_cols)
        saldo_col_bg = config.get('SALDO_FINAL', 'Saldo Final')
        if saldo_col_bg and saldo_col_bg in df_statement_bg.columns: df_statement_bg['Valor_Final'] = pd.to_numeric(df_statement_bg[saldo_col_bg], errors='coerce').fillna(0)
        else: df_statement_bg['Valor_Final'] = 0.0
        df_statement_bg[cuenta_col] = df_statement_bg[cuenta_col].astype(str)
        t_act_bg = get_principal_account_value(df_statement_bg, '1', 'Valor_Final', cuenta_col)
        t_pas_bg = get_principal_account_value(df_statement_bg, '2', 'Valor_Final', cuenta_col)
        t_pat_bg = get_principal_account_value(df_statement_bg, '3', 'Valor_Final', cuenta_col)
        df_display_bg = get_top_level_accounts_for_display(df_statement_bg, 'Valor_Final', statement_type)
        if df_display_bg.empty:
            rows_bg_totals_only = [
                {cuenta_col:'1', nombre_col:'TOTAL ACTIVOS', 'Valor':t_act_bg},
                {cuenta_col:'2', nombre_col:'TOTAL PASIVOS', 'Valor':t_pas_bg},
                {cuenta_col:'3', nombre_col:'TOTAL PATRIMONIO', 'Valor':t_pat_bg},
                {cuenta_col:'', nombre_col:'', 'Valor':None},
                {cuenta_col:'', nombre_col:'TOTAL PASIVO + PATRIMONIO', 'Valor':t_pas_bg + t_pat_bg},
                {cuenta_col:'', nombre_col:'VERIFICACIÓN (Activo = Pasivo+Patrimonio)', 'Valor':t_act_bg + t_pas_bg + t_pat_bg}
            ]
            return pd.DataFrame(rows_bg_totals_only)
        if nivel_col not in df_display_bg.columns: df_display_bg[nivel_col] = 1
        df_display_bg[nivel_col] = pd.to_numeric(df_display_bg[nivel_col], errors='coerce').fillna(9999)
        df_display_bg = df_display_bg[df_display_bg[nivel_col] <= float(max_level)].copy()
        order_categories_bg_list = ['Balance General - Activos', 'Balance General - Pasivos', 'Balance General - Patrimonio']
        final_df_bg_display = pd.DataFrame(columns=final_cols)
        required_loop_cols_bg_list = ['Tipo_Estado', cuenta_col, nombre_col, nivel_col, 'Valor_Final']
        if all(col_loop_bg_item in df_display_bg.columns for col_loop_bg_item in required_loop_cols_bg_list):
            for tipo_estado_cat_bg in order_categories_bg_list:
                group_bg_display = df_display_bg[df_display_bg['Tipo_Estado'] == tipo_estado_cat_bg].copy()
                if not group_bg_display.empty:
                    group_bg_display = group_bg_display.sort_values(by=cuenta_col)
                    group_bg_display[nivel_col] = pd.to_numeric(group_bg_display[nivel_col], errors='coerce').fillna(1).astype(int)
                    group_bg_display['Nombre_Cuenta_Display'] = group_bg_display.apply( lambda r: f"{' ' * (r[nivel_col]-1)}{r[nombre_col]}", axis=1 )
                    final_df_bg_display = pd.concat([final_df_bg_display, group_bg_display[[cuenta_col, 'Nombre_Cuenta_Display', 'Valor_Final']].rename( columns={'Nombre_Cuenta_Display': nombre_col, 'Valor_Final': 'Valor'} )], ignore_index=True)
        else:
            rows_bg_principales = [
                {cuenta_col:'1', nombre_col:COL_CONFIG['BALANCE_GENERAL'].get('NOMBRE_CUENTA', 'ACTIVOS'), 'Valor':t_act_bg},
                {cuenta_col:'2', nombre_col:COL_CONFIG['BALANCE_GENERAL'].get('NOMBRE_CUENTA', 'PASIVOS'), 'Valor':t_pas_bg},
                {cuenta_col:'3', nombre_col:COL_CONFIG['BALANCE_GENERAL'].get('NOMBRE_CUENTA', 'PATRIMONIO'), 'Valor':t_pat_bg},
            ]
            final_df_bg_display = pd.DataFrame(rows_bg_principales)
        rows_to_add_bg_final = [
            {cuenta_col:'', nombre_col:'', 'Valor':None},
            {cuenta_col:'', nombre_col:'TOTAL PASIVO + PATRIMONIO', 'Valor':t_pas_bg + t_pat_bg},
            {cuenta_col:'', nombre_col:'VERIFICACIÓN (Activo = Pasivo+Patrimonio)', 'Valor':t_act_bg + t_pas_bg + t_pat_bg}
        ]
        final_df_bg_display = pd.concat([final_df_bg_display, pd.DataFrame(rows_to_add_bg_final)], ignore_index=True)
        return final_df_bg_display
    return pd.DataFrame(columns=final_cols)


def to_excel_buffer(er_df: pd.DataFrame, bg_df: pd.DataFrame) -> io.BytesIO:
    """Exporta los DataFrames de ER y BG a un archivo Excel en memoria."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if er_df is not None and not er_df.empty:
            df_er_export = er_df.copy()
            if 'Valor' in df_er_export.columns:
                df_er_export['Valor'] = df_er_export['Valor'].astype(str).str.replace(r'[$,]', '', regex=True)
                df_er_export['Valor'] = pd.to_numeric(df_er_export['Valor'], errors='coerce')
            df_er_export.to_excel(writer, sheet_name='Estado de Resultados', index=False)
        if bg_df is not None and not bg_df.empty:
            df_bg_export = bg_df.copy()
            if 'Valor' in df_bg_export.columns:
                df_bg_export['Valor'] = df_bg_export['Valor'].astype(str).str.replace(r'[$,]', '', regex=True)
                df_bg_export['Valor'] = pd.to_numeric(df_bg_export['Valor'], errors='coerce')
            df_bg_export.to_excel(writer, sheet_name='Balance General', index=False)
    output.seek(0)
    return output

def procesar_archivo_excel(bytes_del_archivo: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Toma el contenido de un archivo Excel en bytes, lo procesa usando la lógica
    original y devuelve los dos dataframes maestros (ER y BG).
    """
    xls = pd.ExcelFile(bytes_del_archivo)
    if 'EDO RESULTADO' not in xls.sheet_names or 'BALANCE' not in xls.sheet_names:
        raise ValueError("El archivo Excel debe contener las hojas 'EDO RESULTADO' y 'BALANCE'.")

    df_er_raw, df_bg_raw = pd.read_excel(xls, 'EDO RESULTADO'), pd.read_excel(xls, 'BALANCE')
    
    # --- Procesamiento ER (Crea df_er_master) ---
    er_conf = COL_CONFIG['ESTADO_DE_RESULTADOS']
    cuenta_col_er_config = er_conf.get('CUENTA', 'Cuenta')
    er_map_config_keys_as_str = {str(k): v for k, v in er_conf.get('CENTROS_COSTO_COLS', {}).items()}
    df_er_raw.columns = [str(col).strip() for col in df_er_raw.columns]
    er_rename_mapping = {excel_col: logic_name for excel_col, logic_name in er_map_config_keys_as_str.items() if excel_col in df_er_raw.columns}
    current_df_er = df_er_raw.rename(columns=er_rename_mapping).copy()
    for logical_cc_name in er_conf.get('CENTROS_COSTO_COLS', {}).values():
        if logical_cc_name in current_df_er.columns: current_df_er[logical_cc_name] = current_df_er[logical_cc_name].apply(clean_numeric_value)
    for col_key in ['CUENTA', 'NOMBRE_CUENTA', 'NIVEL_LINEA']:
        col_config_name = er_conf.get(col_key, col_key)
        if col_config_name not in current_df_er.columns:
            # Lógica simplificada para encontrar la columna si no coincide directamente
            original_col_name = next((str(k) for k, v in er_conf.items() if v == col_config_name), None)
            if original_col_name and original_col_name in df_er_raw.columns:
                current_df_er[col_config_name] = df_er_raw[original_col_name]
            elif col_config_name in df_er_raw.columns:
                 current_df_er[col_config_name] = df_er_raw[col_config_name]
            else:
                raise ValueError(f"Columna requerida '{col_config_name}' para '{col_key}' (ER) no encontrada en el archivo.")
    
    current_df_er[er_conf.get('NIVEL_LINEA')] = pd.to_numeric(current_df_er[er_conf.get('NIVEL_LINEA')], errors='coerce').fillna(0).astype(int)
    current_df_er[er_conf.get('CUENTA')] = current_df_er[er_conf.get('CUENTA')].astype(str).str.strip()
    current_df_er[er_conf.get('NOMBRE_CUENTA')] = current_df_er[er_conf.get('NOMBRE_CUENTA')].astype(str).str.strip()
    
    current_df_er['Tipo_Estado'] = current_df_er[cuenta_col_er_config].apply(classify_account)
    
    for cc_log_name_sign in er_conf.get('CENTROS_COSTO_COLS', {}).values():
        if cc_log_name_sign in current_df_er.columns:
            current_df_er[cc_log_name_sign] = pd.to_numeric(current_df_er[cc_log_name_sign], errors='coerce').fillna(0)
            ingresos_m = current_df_er['Tipo_Estado'] == 'Estado de Resultados - Ingresos'
            current_df_er.loc[ingresos_m, cc_log_name_sign] = current_df_er.loc[ingresos_m, cc_log_name_sign].abs()
            egresos_m = current_df_er['Tipo_Estado'].str.contains('Estado de Resultados', na=False) & ~ingresos_m
            current_df_er.loc[egresos_m, cc_log_name_sign] = current_df_er.loc[egresos_m, cc_log_name_sign].abs() * -1
    df_er_master = current_df_er

    # --- Procesamiento BG (Crea df_bg_master) ---
    bg_conf = COL_CONFIG['BALANCE_GENERAL']
    current_df_bg = df_bg_raw.copy()
    cuenta_col_bg_config = bg_conf.get('CUENTA', 'Cuenta')
    
    for col_key_bg_base in ['SALDO_INICIAL', 'DEBE', 'HABER', 'SALDO_FINAL', 'CUENTA', 'NOMBRE_CUENTA', 'NIVEL_LINEA']:
        col_cfg_name_bg = bg_conf.get(col_key_bg_base, col_key_bg_base)
        if col_cfg_name_bg not in current_df_bg.columns: raise ValueError(f"Columna '{col_cfg_name_bg}' (BG) no encontrada.")
        if col_key_bg_base not in ['CUENTA', 'NOMBRE_CUENTA', 'NIVEL_LINEA']: current_df_bg[col_cfg_name_bg] = current_df_bg[col_cfg_name_bg].apply(clean_numeric_value)
        elif col_key_bg_base == 'NIVEL_LINEA': current_df_bg[col_cfg_name_bg] = pd.to_numeric(current_df_bg[col_cfg_name_bg], errors='coerce').fillna(0).astype(int)
        else: current_df_bg[col_cfg_name_bg] = current_df_bg[col_cfg_name_bg].astype(str).str.strip()

    current_df_bg[cuenta_col_bg_config] = current_df_bg[cuenta_col_bg_config].astype(str)
    current_df_bg['Tipo_Estado'] = current_df_bg[cuenta_col_bg_config].apply(classify_account)
    df_bg_master = current_df_bg
    
    return df_er_master, df_bg_master
