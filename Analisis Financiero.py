import pandas as pd
import streamlit as st
import io
import numpy as np

# --- Configuración de Columnas ---
COL_CONFIG = {
    'ESTADO_DE_RESULTADOS': {
        'NIVEL_LINEA': 'Grupo',
        'CUENTA': 'Cuenta',
        'NOMBRE_CUENTA': 'Título',
        'CENTROS_COSTO_COLS': { 
            # ***** CAMBIO: Claves de CC numéricas (ej. 156) ahora son enteros *****
            # La clave 'Sin centro de coste' y 'Total' siguen siendo texto.
            'Sin centro de coste': 'Sin centro de coste',
            156: 'Armenia', # Clave numérica
            157: 'San antonio',# Clave numérica
            158: 'Opalo',      # Clave numérica
            189: 'Olaya',      # Clave numérica
            238: 'Laureles',   # Clave numérica
            'Total': 'Total_Consolidado_ER' 
        }
    },
    'BALANCE_GENERAL': {
        'NIVEL_LINEA': 'Grupo',
        'CUENTA': 'Cuenta',
        'NOMBRE_CUENTA': 'Título',
        'SALDO_INICIAL': 'Saldo inicial',
        'DEBE': 'Debe',
        'HABER': 'Haber',
        'SALDO_FINAL': 'Saldo Final'
    }
}

# --- Funciones de Utilidad ---
def clean_numeric_value(value):
    if pd.isna(value) or value == '': return 0.0
    s_value = str(value).strip().replace('.', '').replace(',', '.')
    try: return float(s_value)
    except ValueError: return 0.0

def classify_account(cuenta_str: str) -> str:
    if not isinstance(cuenta_str, str): return 'No Clasificado'
    if cuenta_str.startswith('11'): return 'Balance General - Activo Corriente'
    elif cuenta_str.startswith('1'): return 'Balance General - Activo No Corriente' 
    elif cuenta_str.startswith('21'): return 'Balance General - Pasivo Corriente'
    elif cuenta_str.startswith('2'): return 'Balance General - Pasivo No Corriente' 
    elif cuenta_str.startswith('3'): return 'Balance General - Patrimonio'
    elif cuenta_str.startswith('4'): return 'Estado de Resultados - Ingresos'
    elif cuenta_str.startswith('5'): return 'Estado de Resultados - Ingresos'
    elif cuenta_str.startswith('6'): return 'Estado de Resultados - Costo de Ventas'
    elif cuenta_str.startswith('7'): return 'Estado de Resultados - Gastos Operacionales'
    elif cuenta_str.startswith('8'): return 'Estado de Resultados - Gastos no Operacionales'
    elif cuenta_str.startswith('9'): return 'Estado de Resultados - Impuestos'
    else: return 'No Clasificado'

def get_top_level_accounts_for_display(df_raw: pd.DataFrame, value_col_name: str, statement_type: str) -> pd.DataFrame:
    cuenta_col_key = statement_type.replace(' ', '_').upper()
    if cuenta_col_key not in COL_CONFIG: return pd.DataFrame()
    config_specific = COL_CONFIG[cuenta_col_key]
    if not all(k in config_specific for k in ['CUENTA', 'NOMBRE_CUENTA', 'NIVEL_LINEA']): return pd.DataFrame()
    cuenta_col, nombre_cuenta_col, nivel_linea_col = config_specific['CUENTA'], config_specific['NOMBRE_CUENTA'], config_specific['NIVEL_LINEA']
    required_cols = [cuenta_col, nombre_cuenta_col, nivel_linea_col, value_col_name]
    if not all(col in df_raw.columns for col in required_cols): return pd.DataFrame()
    df_processed = df_raw.copy()
    df_processed['Cuenta_Str'] = df_processed[cuenta_col].astype(str)
    df_sorted = df_processed.sort_values(by='Cuenta_Str').reset_index(drop=True)
    df_sorted = df_sorted.dropna(subset=['Cuenta_Str', nombre_cuenta_col])
    df_sorted = df_sorted[df_sorted['Cuenta_Str'] != ''].reset_index(drop=True)
    df_sorted = df_sorted[df_sorted[nombre_cuenta_col] != ''].reset_index(drop=True)
    df_significant_values = df_sorted[df_sorted[value_col_name].abs() > 0.001].copy() 
    unique_values = df_significant_values[value_col_name].unique() if not df_significant_values.empty else []
    selected_rows = []
    for val in unique_values:
        group = df_significant_values[df_significant_values[value_col_name] == val]
        if not group.empty: selected_rows.append(group.loc[group['Cuenta_Str'].str.len().idxmin()])
    df_result = pd.DataFrame(selected_rows).drop_duplicates(subset=['Cuenta_Str']).reset_index(drop=True) if selected_rows else pd.DataFrame(columns=df_sorted.columns)
    levels_map = {'Estado de Resultados': [1,2,3,4,6,8], 'Balance General': [1,2,3,4]}
    zero_levels = levels_map.get(statement_type, [])
    df_zero_sig = df_sorted[(df_sorted[value_col_name].abs() < 0.001) & (pd.to_numeric(df_sorted[nivel_linea_col], errors='coerce').isin(zero_levels))].copy()
    if df_result.empty and df_zero_sig.empty: return pd.DataFrame(columns=df_sorted.columns)
    df_final = pd.concat([df_result, df_zero_sig]).drop_duplicates(subset=['Cuenta_Str']).reset_index(drop=True)
    return df_final.sort_values(by=[nivel_linea_col, 'Cuenta_Str']) if not df_final.empty else df_final

def generate_financial_statement(df_full_data: pd.DataFrame, statement_type: str, selected_cc_filter: str = None, max_level: int = 999) -> pd.DataFrame:
    config_key = statement_type.replace(' ', '_').upper()
    if config_key not in COL_CONFIG: return pd.DataFrame()
    config = COL_CONFIG[config_key]
    cuenta_col, nombre_col, nivel_col = config['CUENTA'], config['NOMBRE_CUENTA'], config['NIVEL_LINEA']
    final_cols = [cuenta_col, nombre_col, 'Valor']
    base_check = [cuenta_col, nombre_col, nivel_col, 'Tipo_Estado']
    if not all(col in df_full_data.columns for col in base_check): return pd.DataFrame(columns=final_cols)

    if statement_type == 'Estado de Resultados':
        df_statement = df_full_data[df_full_data['Tipo_Estado'].str.contains('Estado de Resultados', na=False)].copy()
        if df_statement.empty: return pd.DataFrame(columns=final_cols)
        value_col_to_use_for_report = ''
        if selected_cc_filter and selected_cc_filter != 'Todos': 
            if selected_cc_filter in df_statement.columns: value_col_to_use_for_report = selected_cc_filter
            else: df_statement['Valor_Fallback_CC'] = 0.0; value_col_to_use_for_report = 'Valor_Fallback_CC'
        else: 
            total_er_col_name = config['CENTROS_COSTO_COLS'].get('Total') 
            if total_er_col_name and total_er_col_name in df_statement.columns: value_col_to_use_for_report = total_er_col_name
            else: 
                # Usar str(k) para la comparación por si las claves son mixtas (int/str)
                cc_individual_cols = [v for k,v in config['CENTROS_COSTO_COLS'].items() if str(k).lower() != 'total' and v in df_statement.columns and v != config['CENTROS_COSTO_COLS'].get('Total')]
                if cc_individual_cols:
                    df_statement['__temp_sum_all_cc_for_report'] = df_statement[cc_individual_cols].sum(axis=1)
                    value_col_to_use_for_report = '__temp_sum_all_cc_for_report'
                else: df_statement['Valor_Fallback_Total'] = 0.0; value_col_to_use_for_report = 'Valor_Fallback_Total'
        df_statement['Valor_Final'] = df_statement[value_col_to_use_for_report]
        df_display = get_top_level_accounts_for_display(df_statement, 'Valor_Final', statement_type)
        if df_display.empty: return pd.DataFrame(columns=final_cols)
        df_display = df_display[pd.to_numeric(df_display[nivel_col], errors='coerce').fillna(9999) <= float(max_level)].copy()
        order_categories = ['Ingresos', 'Costo de Ventas', 'Gastos Operacionales', 'Gastos no Operacionales', 'Impuestos']
        order = [f'Estado de Resultados - {i}' for i in order_categories]
        final_df = pd.DataFrame(columns=final_cols)
        if 'Tipo_Estado' not in df_display.columns: return final_df
        for tipo in order:
            group = df_display[df_display['Tipo_Estado'] == tipo].copy()
            if not group.empty:
                group = group.sort_values(by=cuenta_col)
                group['Nombre_Cuenta_Display'] = group.apply(lambda r: f"{'  '*(int(pd.to_numeric(r[nivel_col],errors='coerce')or 1)-1)}{r[nombre_col]}", axis=1)
                group['Valor_Final'] = group['Valor_Final'].fillna(0)
                final_df = pd.concat([final_df, group[[cuenta_col, 'Nombre_Cuenta_Display', 'Valor_Final']].rename(columns={'Nombre_Cuenta_Display':nombre_col, 'Valor_Final':'Valor'})], ignore_index=True)
        total_val = 0.0
        if 'Valor' in final_df.columns: total_val = pd.to_numeric(final_df['Valor'], errors='coerce').sum()
        final_df.loc[len(final_df)] = pd.Series({cuenta_col:'', nombre_col:'TOTAL ESTADO DE RESULTADOS', 'Valor':total_val})
        final_df.loc[len(final_df)] = pd.Series({cuenta_col:'', nombre_col:'', 'Valor':None}) 
        return final_df
    elif statement_type == 'Balance General':
        df_statement = df_full_data[df_full_data['Tipo_Estado'].str.contains('Balance General', na=False)].copy()
        if df_statement.empty: return pd.DataFrame(columns=final_cols)
        saldo_col = config.get('SALDO_FINAL')
        df_statement['Valor_Final'] = df_statement[saldo_col] if saldo_col and saldo_col in df_statement.columns else 0.0
        df_display = get_top_level_accounts_for_display(df_statement, 'Valor_Final', statement_type)
        if df_display.empty: return pd.DataFrame(columns=final_cols)
        df_display = df_display[pd.to_numeric(df_display[nivel_col], errors='coerce').fillna(9999) <= float(max_level)].copy()
        order_categories_bg = ['Activo Corriente', 'Activo No Corriente', 'Pasivo Corriente', 'Pasivo No Corriente', 'Patrimonio']
        order_bg = [f'Balance General - {i}' for i in order_categories_bg]
        final_df_bg = pd.DataFrame(columns=final_cols)
        t_act_c, t_act_nc, t_pas_c, t_pas_nc, t_pat = 0.0,0.0,0.0,0.0,0.0
        if 'Tipo_Estado' not in df_display.columns: return final_df_bg
        for tipo_raw_bg in order_categories_bg:
            tipo_bg = f'Balance General - {tipo_raw_bg}'
            group_bg = df_display[df_display['Tipo_Estado'] == tipo_bg].copy()
            current_s_bg = 0.0
            if not group_bg.empty:
                group_bg = group_bg.sort_values(by=cuenta_col)
                group_bg['Nombre_Cuenta_Display'] = group_bg.apply(lambda r: f"{'  '*(int(pd.to_numeric(r[nivel_col],errors='coerce')or 1)-1)}{r[nombre_col]}", axis=1)
                group_bg['Valor_Final'] = group_bg['Valor_Final'].fillna(0)
                final_df_bg = pd.concat([final_df_bg, group_bg[[cuenta_col, 'Nombre_Cuenta_Display', 'Valor_Final']].rename(columns={'Nombre_Cuenta_Display':nombre_col, 'Valor_Final':'Valor'})], ignore_index=True)
                current_s_bg = pd.to_numeric(group_bg['Valor_Final'], errors='coerce').sum()
            if not group_bg.empty and tipo_raw_bg != 'Patrimonio' :
                 final_df_bg.loc[len(final_df_bg)] = pd.Series({cuenta_col:'', nombre_col:f'SUBTOTAL {tipo_raw_bg.upper()}', 'Valor':current_s_bg})
            if tipo_raw_bg == 'Activo Corriente': t_act_c += current_s_bg
            elif tipo_raw_bg == 'Activo No Corriente': t_act_nc += current_s_bg
            elif tipo_raw_bg == 'Pasivo Corriente': t_pas_c += current_s_bg
            elif tipo_raw_bg == 'Pasivo No Corriente': t_pas_nc += current_s_bg
            elif tipo_raw_bg == 'Patrimonio': t_pat += current_s_bg
        t_act = t_act_c + t_act_nc
        t_pas = t_pas_c + t_pas_nc
        final_df_bg.loc[len(final_df_bg)] = pd.Series({cuenta_col:'', nombre_col:'TOTAL ACTIVOS', 'Valor':t_act})
        final_df_bg.loc[len(final_df_bg)] = pd.Series({cuenta_col:'', nombre_col:'', 'Valor':None}) 
        final_df_bg.loc[len(final_df_bg)] = pd.Series({cuenta_col:'', nombre_col:'TOTAL PASIVOS', 'Valor':t_pas})
        final_df_bg.loc[len(final_df_bg)] = pd.Series({cuenta_col:'', nombre_col:'TOTAL PATRIMONIO', 'Valor':t_pat})
        final_df_bg.loc[len(final_df_bg)] = pd.Series({cuenta_col:'', nombre_col:'TOTAL PASIVO + PATRIMONIO', 'Valor':t_pas + t_pat})
        final_df_bg.loc[len(final_df_bg)] = pd.Series({cuenta_col:'', nombre_col:'DIFERENCIA (A-(P+Pt))', 'Valor':t_act - (t_pas + t_pat)})
        return final_df_bg
    return pd.DataFrame(columns=final_cols)

def to_excel_buffer(er_df: pd.DataFrame, bg_df: pd.DataFrame) -> io.BytesIO:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if er_df is not None and not er_df.empty: er_df.to_excel(writer, sheet_name='Estado de Resultados', index=False)
        if bg_df is not None and not bg_df.empty: bg_df.to_excel(writer, sheet_name='Balance General', index=False)
    output.seek(0)
    return output

# --- Aplicación Streamlit ---
st.set_page_config(layout="wide", page_title="Análisis Financiero Avanzado")
st.title("💰 Análisis Financiero y Tablero Gerencial")

for key in ['df_er', 'df_bg', 'final_er_display', 'final_bg_display']:
    if key not in st.session_state: st.session_state[key] = pd.DataFrame()

uploaded_file = st.file_uploader("Sube tu archivo Excel (hojas 'EDO RESULTADO' y 'BALANCE')", type=["xlsx"])

if uploaded_file is not None:
    try:
        xls = pd.ExcelFile(uploaded_file)
        if 'EDO RESULTADO' not in xls.sheet_names or 'BALANCE' not in xls.sheet_names:
            st.error("Asegúrate que el archivo Excel contenga las hojas 'EDO RESULTADO' y 'BALANCE'.")
            st.stop()
        df_er_raw, df_bg_raw = pd.read_excel(xls, 'EDO RESULTADO'), pd.read_excel(xls, 'BALANCE')
        st.success("Archivo cargado correctamente.")

        # ***** LÍNEAS DE DEPURACIÓN TEMPORALES *****
        st.expander("DEPURACIÓN: Columnas de Excel y DataFrame ER").write({
            "Columnas leídas de 'EDO RESULTADO' (Excel)": df_er_raw.columns.tolist(),
        })
        # ***** FIN DE LÍNEAS DE DEPURACIÓN *****

        er_conf, bg_conf = COL_CONFIG['ESTADO_DE_RESULTADOS'], COL_CONFIG['BALANCE_GENERAL']
        
        # --- Procesamiento ER ---
        # 1. Mapear/Renombrar columnas de CC según COL_CONFIG
        # Las claves de CENTROS_COSTO_COLS (ej. 156) DEBEN coincidir con las columnas de df_er_raw
        # (sean string o int, deben coincidir con cómo pandas las lee).
        er_map = {k_excel: v_logico for k_excel, v_logico in er_conf['CENTROS_COSTO_COLS'].items() if k_excel in df_er_raw.columns}
        st.session_state.df_er = df_er_raw.rename(columns=er_map).copy()

        st.expander("DEPURACIÓN: Columnas de df_er y Mapeo de CC").write({
            "Columnas en df_er DESPUÉS del renombrado de CCs": st.session_state.df_er.columns.tolist(),
            "Mapeo de CC aplicado (er_map)": er_map
        })
        
        # 2. Aplicar limpieza numérica
        for logical_cc_name in er_conf['CENTROS_COSTO_COLS'].values():
            if logical_cc_name in st.session_state.df_er.columns:
                st.session_state.df_er[logical_cc_name] = st.session_state.df_er[logical_cc_name].apply(clean_numeric_value)
        
        # 3. Procesar columnas de identificación y clasificación
        for col_key in ['CUENTA', 'NOMBRE_CUENTA', 'NIVEL_LINEA']:
            col = er_conf[col_key]
            if col not in st.session_state.df_er.columns: 
                st.error(f"Columna base '{col}' para '{col_key}' (ER) no encontrada. Revisa tu Excel o COL_CONFIG.")
                st.stop()
            st.session_state.df_er[col] = st.session_state.df_er[col].astype(str).str.strip() if col_key != 'NIVEL_LINEA' else pd.to_numeric(st.session_state.df_er[col], errors='coerce').fillna(0).astype(int)
        st.session_state.df_er['Tipo_Estado'] = st.session_state.df_er[er_conf['CUENTA']].apply(classify_account)

        # 4. Corrección de Signos para ER
        for col_to_fix_sign in er_conf['CENTROS_COSTO_COLS'].values():
            if col_to_fix_sign in st.session_state.df_er.columns:
                ingresos_mask = st.session_state.df_er['Tipo_Estado'].str.contains('Ingresos', na=False)
                st.session_state.df_er[col_to_fix_sign] = np.where(ingresos_mask, st.session_state.df_er[col_to_fix_sign] * -1, st.session_state.df_er[col_to_fix_sign])
                egresos_mask = (st.session_state.df_er['Tipo_Estado'].str.contains('Costo de Ventas',na=False) | st.session_state.df_er['Tipo_Estado'].str.contains('Gastos Operacionales',na=False) | st.session_state.df_er['Tipo_Estado'].str.contains('Gastos no Operacionales',na=False) | st.session_state.df_er['Tipo_Estado'].str.contains('Impuestos',na=False))
                st.session_state.df_er[col_to_fix_sign] = np.where(egresos_mask, st.session_state.df_er[col_to_fix_sign] * -1, st.session_state.df_er[col_to_fix_sign])
        
        # --- Procesamiento BG ---
        st.session_state.df_bg = df_bg_raw.copy()
        for col_key in ['SALDO_INICIAL', 'DEBE', 'HABER', 'SALDO_FINAL', 'CUENTA', 'NOMBRE_CUENTA', 'NIVEL_LINEA']:
            col = bg_conf[col_key]
            if col not in st.session_state.df_bg.columns: st.error(f"Columna '{col}' (BG) no encontrada."); st.stop()
            if col_key not in ['CUENTA', 'NOMBRE_CUENTA', 'NIVEL_LINEA']: st.session_state.df_bg[col] = st.session_state.df_bg[col].apply(clean_numeric_value)
            else: st.session_state.df_bg[col] = st.session_state.df_bg[col].astype(str).str.strip() if col_key != 'NIVEL_LINEA' else pd.to_numeric(st.session_state.df_bg[col], errors='coerce').fillna(0).astype(int)
        st.session_state.df_bg['Tipo_Estado'] = st.session_state.df_bg[bg_conf['CUENTA']].apply(classify_account)

    except Exception as e: st.error(f"Error al procesar archivo: {e}"); st.exception(e)

# --- Interfaz de Usuario ---
st.sidebar.header("Opciones de Reporte")
report_type = st.sidebar.radio("Selecciona el reporte:", ["Estado de Resultados", "Balance General"],
                               disabled=st.session_state.df_er.empty and st.session_state.df_bg.empty)

selected_cc_report = "Todos" 
if report_type == "Estado de Resultados" and not st.session_state.df_er.empty:
    er_conf_sidebar_cc_filter = COL_CONFIG['ESTADO_DE_RESULTADOS']
    cc_options_list_for_filter = []
    logical_total_col_name = er_conf_sidebar_cc_filter['CENTROS_COSTO_COLS'].get('Total')

    for excel_col_key, logical_col_name in er_conf_sidebar_cc_filter['CENTROS_COSTO_COLS'].items():
        if str(excel_col_key).lower() != 'total': # Excluir la entrada 'Total' de COL_CONFIG
            if logical_col_name in st.session_state.df_er.columns: # Si el nombre lógico es una columna en df_er
                 if logical_col_name != logical_total_col_name: # Y no es el nombre del total consolidado
                    cc_options_list_for_filter.append(logical_col_name)
    
    cc_options_list_for_filter = sorted(list(set(cc_options_list_for_filter)))

    if cc_options_list_for_filter:
        selected_cc_report = st.sidebar.selectbox(
            "Filtrar por Centro de Costo (ER):", 
            ['Todos'] + cc_options_list_for_filter, 
            key="cc_filter_er_sidebar_main"
        )
    elif er_conf_sidebar_cc_filter['CENTROS_COSTO_COLS'].get('Total') in st.session_state.df_er.columns :
        # Si solo está el total y no hay CCs individuales, no mostrar el filtro, se asume "Todos".
        pass
    else:
        st.sidebar.info("No se encontraron Centros de Costo individuales para filtrar.")


st.sidebar.header("Buscar Cuenta Específica")
search_account_input = st.sidebar.text_input("Número de Cuenta a detallar:", key="search_account_input_main")

# --- Reportes Principales y Detalle de Cuenta Buscada ---
if report_type == "Estado de Resultados" and not st.session_state.df_er.empty:
    st.header(f"📈 Estado de Resultados ({selected_cc_report})")
    er_conf_display = COL_CONFIG['ESTADO_DE_RESULTADOS']
    df_kpi_er = st.session_state.df_er.copy()
    val_col_kpi_er_display = ''
    if selected_cc_report and selected_cc_report != 'Todos':
        if selected_cc_report in df_kpi_er.columns: val_col_kpi_er_display = selected_cc_report
    else: 
        val_col_kpi_er_display = er_conf_display['CENTROS_COSTO_COLS'].get('Total')
        if not (val_col_kpi_er_display and val_col_kpi_er_display in df_kpi_er.columns):
            cc_individual_cols_kpi_disp = [v for k,v in er_conf_display['CENTROS_COSTO_COLS'].items() if str(k).lower() != 'total' and v in df_kpi_er.columns and v != er_conf_display['CENTROS_COSTO_COLS'].get('Total')]
            if cc_individual_cols_kpi_disp:
                df_kpi_er['__temp_sum_for_kpi_display'] = df_kpi_er[cc_individual_cols_kpi_disp].sum(axis=1)
                val_col_kpi_er_display = '__temp_sum_for_kpi_display'
            else: val_col_kpi_er_display = None
    kpi_er_ingresos = 0.0
    kpi_er_cv = 0.0 
    kpi_er_go = 0.0    
    kpi_er_uo = 0.0
    kpi_er_un = 0.0
    if val_col_kpi_er_display:
        kpi_er_ingresos = df_kpi_er[df_kpi_er['Tipo_Estado'] == 'Estado de Resultados - Ingresos'][val_col_kpi_er_display].sum()
        kpi_er_cv = df_kpi_er[df_kpi_er['Tipo_Estado'] == 'Estado de Resultados - Costo de Ventas'][val_col_kpi_er_display].sum()
        kpi_er_go = df_kpi_er[df_kpi_er['Tipo_Estado'] == 'Estado de Resultados - Gastos Operacionales'][val_col_kpi_er_display].sum()
    else: st.warning(f"No se pudo determinar la columna de valor para KPIs del ER ({selected_cc_report}).")
    kpi_er_uo = kpi_er_ingresos + kpi_er_cv + kpi_er_go 
    kpi_er_margen_op = (kpi_er_uo / kpi_er_ingresos) * 100 if kpi_er_ingresos else 0.0
    
    kpi_cols_er_display_main = st.columns(2) # Ajustado a 2 columnas
    kpi_cols_er_display_main[0].metric("Utilidad Operativa", f"${kpi_er_uo:,.0f}", f"{kpi_er_margen_op:.1f}% Margen Op.")
    
    if val_col_kpi_er_display:
        if kpi_er_uo > 0: st.success(f"¡Felicidades! Con una Utilidad Operativa de ${kpi_er_uo:,.0f}, estás por encima del punto de equilibrio operativo para {selected_cc_report}.")
        elif kpi_er_uo == 0 and kpi_er_ingresos > 0 : st.info(f"Estás en el punto de equilibrio operativo para {selected_cc_report}.")
        elif kpi_er_uo < 0 :
            costos_variables_abs_calc = abs(kpi_er_cv) 
            gastos_fijos_abs_calc = abs(kpi_er_go)
            margen_contribucion_total_calc = kpi_er_ingresos - costos_variables_abs_calc
            if kpi_er_ingresos > 0 and margen_contribucion_total_calc > 0:
                porc_margen_contribucion_calc = margen_contribucion_total_calc / kpi_er_ingresos
                if porc_margen_contribucion_calc > 0:
                    ingresos_equilibrio_op_calc = gastos_fijos_abs_calc / porc_margen_contribucion_calc
                    ventas_adicionales_necesarias_calc = ingresos_equilibrio_op_calc - kpi_er_ingresos
                    if ventas_adicionales_necesarias_calc > 0 :
                        st.warning(f"Para Utilidad Op. = $0 en {selected_cc_report}, se requerirían ingresos adicionales por aprox. ${ventas_adicionales_necesarias_calc:,.0f} (total ingresos: ${ingresos_equilibrio_op_calc:,.0f}).\n(Supone Gastos Op. fijos y Costo de Ventas variable).")
                    else: st.info(f"Utilidad Operativa: ${kpi_er_uo:,.0f}. Ya cubre fijos con margen de contribución positivo.")
                else: st.error(f"Utilidad Operativa: ${kpi_er_uo:,.0f}. Margen de contribución no positivo. Revisar costos.")
            elif kpi_er_ingresos > 0: st.error(f"Utilidad Operativa: ${kpi_er_uo:,.0f}. Margen de contribución cero o negativo. No es posible alcanzar punto de equilibrio operativo sin reestructurar costos/precios.")
            else: st.error(f"Utilidad Operativa: ${kpi_er_uo:,.0f}. No hay ingresos para calcular punto de equilibrio.")

    er_niv_col_main_disp_final = er_conf_display['NIVEL_LINEA']
    if er_niv_col_main_disp_final in st.session_state.df_er.columns:
        lvls_er_main_disp_final = sorted(st.session_state.df_er[er_niv_col_main_disp_final].astype(int).unique().tolist())
        min_l_er_d_f, max_l_er_d_f = (min(lvls_er_main_disp_final), max(lvls_er_main_disp_final)) if lvls_er_main_disp_final else (1,1)
        max_lvl_er_main_disp_final = st.sidebar.slider("Nivel Detalle (ER):", min_value=min_l_er_d_f, max_value=max_l_er_d_f, value=min_l_er_d_f, key="slider_er_level_main_display_final")
        st.session_state.final_er_display = generate_financial_statement(st.session_state.df_er, 'Estado de Resultados', selected_cc_report, max_lvl_er_main_disp_final)
        if not st.session_state.final_er_display.empty:
            total_er_row_main_disp_final = st.session_state.final_er_display[st.session_state.final_er_display[er_conf_display['NOMBRE_CUENTA']] == 'TOTAL ESTADO DE RESULTADOS']
            if not total_er_row_main_disp_final.empty:
                kpi_er_un = total_er_row_main_disp_final['Valor'].iloc[0]
                kpi_er_margen_neto = (kpi_er_un / kpi_er_ingresos) * 100 if kpi_er_ingresos else 0.0
                kpi_cols_er_display_main[1].metric("Utilidad Neta", f"${kpi_er_un:,.0f}", f"{kpi_er_margen_neto:.1f}% Margen Neto")
        st.dataframe(st.session_state.final_er_display, use_container_width=True, hide_index=True)

elif report_type == "Balance General" and not st.session_state.df_bg.empty:
    # ... (Sección BG sin cambios, ya incluye Razón Corriente)
    st.header("⚖️ Balance General") 
    bg_conf_main = COL_CONFIG['BALANCE_GENERAL']
    df_bg_kpi_main = st.session_state.df_bg.copy() 
    saldo_final_col_bg_kpi_main = bg_conf_main['SALDO_FINAL']
    kpi_bg_ac, kpi_bg_pc = 0.0, 0.0
    kpi_bg_at, kpi_bg_pt, kpi_bg_pat = 0.0,0.0,0.0
    if saldo_final_col_bg_kpi_main in df_bg_kpi_main.columns and 'Tipo_Estado' in df_bg_kpi_main.columns:
        kpi_bg_ac = df_bg_kpi_main[df_bg_kpi_main['Tipo_Estado'] == 'Balance General - Activo Corriente'][saldo_final_col_bg_kpi_main].sum()
        kpi_bg_pc = df_bg_kpi_main[df_bg_kpi_main['Tipo_Estado'] == 'Balance General - Pasivo Corriente'][saldo_final_col_bg_kpi_main].sum()
        kpi_bg_at = df_bg_kpi_main[df_bg_kpi_main['Tipo_Estado'].str.contains('Activo', na=False)][saldo_final_col_bg_kpi_main].sum()
        kpi_bg_pt = df_bg_kpi_main[df_bg_kpi_main['Tipo_Estado'].str.contains('Pasivo', na=False)][saldo_final_col_bg_kpi_main].sum()
        kpi_bg_pat = df_bg_kpi_main[df_bg_kpi_main['Tipo_Estado'] == 'Balance General - Patrimonio'][saldo_final_col_bg_kpi_main].sum()
    kpi_bg_rc = (kpi_bg_ac / kpi_bg_pc) if kpi_bg_pc and kpi_bg_pc != 0 else 0.0
    kpi_bg_end = (kpi_bg_pt / kpi_bg_at) * 100 if kpi_bg_at and kpi_bg_at != 0 else 0.0
    kpi_bg_pp = (kpi_bg_pt / kpi_bg_pat) if kpi_bg_pat and kpi_bg_pat != 0 else 0.0
    kpi_cols_bg_main = st.columns(4)
    kpi_cols_bg_main[0].metric("Total Activos", f"${kpi_bg_at:,.0f}")
    kpi_cols_bg_main[1].metric("Total Pasivos", f"${kpi_bg_pt:,.0f}", f"{kpi_bg_end:.1f}% Endeud.")
    kpi_cols_bg_main[2].metric("Total Patrimonio", f"${kpi_bg_pat:,.0f}", f"{kpi_bg_pp:.2f} Pas/Pat")
    kpi_cols_bg_main[3].metric("Razón Corriente", f"{kpi_bg_rc:.2f}")
    bg_niv_col_main = bg_conf_main['NIVEL_LINEA']
    if bg_niv_col_main in st.session_state.df_bg.columns:
        lvls_bg_main = sorted(st.session_state.df_bg[bg_niv_col_main].astype(int).unique().tolist())
        min_l_bg, max_l_bg = (min(lvls_bg_main), max(lvls_bg_main)) if lvls_bg_main else (1,1)
        max_lvl_bg_main = st.sidebar.slider("Nivel Detalle (BG):", min_value=min_l_bg, max_value=max_l_bg, value=min_l_bg, key="slider_bg_level_main")
        st.session_state.final_bg_display = generate_financial_statement(st.session_state.df_bg, 'Balance General', max_level=max_lvl_bg_main)
        st.dataframe(st.session_state.final_bg_display, use_container_width=True, hide_index=True)
        if kpi_bg_at or kpi_bg_pt or kpi_bg_pat :
            st.subheader("Composición del Balance")
            chart_bg_data_main = pd.DataFrame({'Categoría': ['Activos', 'Pasivos', 'Patrimonio'],'Valor': [kpi_bg_at, kpi_bg_pt, kpi_bg_pat]})
            st.bar_chart(chart_bg_data_main.set_index('Categoría'))

elif (st.session_state.df_er.empty or st.session_state.df_bg.empty) and uploaded_file:
    st.info("Procesando datos... o verifique errores previos.")
elif not uploaded_file:
    st.info("👆 Sube tu archivo Excel para comenzar el análisis.")

# --- Detalle de Cuenta Buscada en la página principal ---
if search_account_input: 
    with st.expander(f"Detalle y Subcuentas para la Cuenta '{search_account_input}'", expanded=True):
        df_for_search_detail = pd.DataFrame()
        config_for_search_detail = {}
        is_er_detail_search = False
        original_df_for_subaccounts = pd.DataFrame()
        if report_type == "Estado de Resultados" and not st.session_state.df_er.empty:
            original_df_for_subaccounts = st.session_state.df_er 
            config_for_search_detail = COL_CONFIG['ESTADO_DE_RESULTADOS']
            is_er_detail_search = True
        elif report_type == "Balance General" and not st.session_state.df_bg.empty:
            original_df_for_subaccounts = st.session_state.df_bg
            config_for_search_detail = COL_CONFIG['BALANCE_GENERAL']
            is_er_detail_search = False
        if not original_df_for_subaccounts.empty and config_for_search_detail:
            cuenta_col_s = config_for_search_detail['CUENTA']
            nombre_col_s = config_for_search_detail['NOMBRE_CUENTA']
            sub_accounts_df = original_df_for_subaccounts[original_df_for_subaccounts[cuenta_col_s].astype(str).str.startswith(search_account_input)].copy()
            if not sub_accounts_df.empty:
                st.write(f"Mostrando la cuenta '{search_account_input}' y sus subcuentas:")
                if is_er_detail_search:
                    cols_to_show_er = [cuenta_col_s, nombre_col_s] + list(config_for_search_detail['CENTROS_COSTO_COLS'].values())
                    cols_to_show_er_existing = [col for col in cols_to_show_er if col in sub_accounts_df.columns]
                    st.dataframe(sub_accounts_df[cols_to_show_er_existing], use_container_width=True, hide_index=True)
                else: 
                    cols_to_show_bg = [cuenta_col_s, nombre_col_s, config_for_search_detail['SALDO_INICIAL'], config_for_search_detail['DEBE'], config_for_search_detail['HABER'], config_for_search_detail['SALDO_FINAL']]
                    cols_to_show_bg_existing = [col for col in cols_to_show_bg if col in sub_accounts_df.columns]
                    st.dataframe(sub_accounts_df[cols_to_show_bg_existing], use_container_width=True, hide_index=True)
            elif search_account_input: st.info(f"No se encontró la cuenta '{search_account_input}' o subcuentas asociadas.")
        elif search_account_input: st.info(f"No hay datos cargados para el reporte de {report_type} para buscar la cuenta.")

# --- Descarga ---
st.sidebar.markdown("---")
st.sidebar.subheader("Exportar a Excel")
er_dl = st.session_state.final_er_display if not st.session_state.final_er_display.empty else st.session_state.df_er
bg_dl = st.session_state.final_bg_display if not st.session_state.final_bg_display.empty else st.session_state.df_bg
disable_dl = er_dl.empty and bg_dl.empty
excel_buffer = to_excel_buffer(er_dl, bg_dl)
st.sidebar.download_button(label="Descargar Reporte Seleccionado", data=excel_buffer, file_name=f"reporte_{report_type.lower().replace(' ', '_')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", disabled=disable_dl)
