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
        'CENTROS_COSTO_COLS': { # La CLAVE es como aparece en el Excel, el VALOR es como quieres que se llame en la app y en el DataFrame interno
            'Sin centro de coste': 'Sin centro de coste', # Asumiendo que esta columna existe en tu Excel ER
            '156': 'Armenia',
            '157': 'San antonio',
            '158': 'Opalo',
            '189': 'Olaya',
            '238': 'Laureles',
            'Total': 'Total_Consolidado_ER' # Nombre de la columna en Excel que tiene el total del ER
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
    # ... (Esta función se mantiene, su lógica de seleccionar cuentas principales es para la tabla agregada)
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

        # ***** CAMBIO: La columna de valor se determina por selected_cc_filter *****
        value_col_to_use_for_report = ''
        if selected_cc_filter and selected_cc_filter != 'Todos': 
            # selected_cc_filter es el nombre lógico del CC (ej. "Armenia")
            if selected_cc_filter in df_statement.columns:
                value_col_to_use_for_report = selected_cc_filter
            else: # Fallback si el CC seleccionado no es una columna (no debería pasar si la config es correcta)
                df_statement['Valor_Fallback_CC'] = 0.0
                value_col_to_use_for_report = 'Valor_Fallback_CC'
        else: # "Todos" o consolidado
            total_er_col_name = config['CENTROS_COSTO_COLS'].get('Total') # Ej: 'Total_Consolidado_ER'
            if total_er_col_name and total_er_col_name in df_statement.columns: 
                value_col_to_use_for_report = total_er_col_name
            else: 
                # Si no hay 'Total_Consolidado_ER', sumar los CCs individuales que no sean 'Total'
                cc_individual_cols = [v for k,v in config['CENTROS_COSTO_COLS'].items() if k != 'Total' and v in df_statement.columns]
                if cc_individual_cols:
                    df_statement['__temp_sum_all_cc_for_report'] = df_statement[cc_individual_cols].sum(axis=1)
                    value_col_to_use_for_report = '__temp_sum_all_cc_for_report'
                else: 
                    df_statement['Valor_Fallback_Total'] = 0.0
                    value_col_to_use_for_report = 'Valor_Fallback_Total'
        
        df_statement['Valor_Final'] = df_statement[value_col_to_use_for_report]
        # ----- FIN CAMBIO -----

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
        if 'Valor' in final_df.columns:
            total_val = pd.to_numeric(final_df['Valor'], errors='coerce').sum()
        
        final_df.loc[len(final_df)] = pd.Series({cuenta_col:'', nombre_col:'TOTAL ESTADO DE RESULTADOS', 'Valor':total_val})
        final_df.loc[len(final_df)] = pd.Series({cuenta_col:'', nombre_col:'', 'Valor':None}) 
        return final_df

    elif statement_type == 'Balance General':
        # ... (Lógica del BG sin cambios directos por el filtro de CC, ya que ese filtro es para ER)
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
            
            # Añadir subtotal para la categoría si hubo cuentas y no es Patrimonio (que va al final)
            if not group_bg.empty and tipo_raw_bg != 'Patrimonio':
                 final_df_bg.loc[len(final_df_bg)] = pd.Series({cuenta_col:'', nombre_col:f'SUBTOTAL {tipo_raw_bg.upper()}', 'Valor':current_s_bg})

            if tipo_raw_bg == 'Activo Corriente': t_act_c += current_s_bg
            elif tipo_raw_bg == 'Activo No Corriente': t_act_nc += current_s_bg
            elif tipo_raw_bg == 'Pasivo Corriente': t_pas_c += current_s_bg
            elif tipo_raw_bg == 'Pasivo No Corriente': t_pas_nc += current_s_bg
            elif tipo_raw_bg == 'Patrimonio': t_pat += current_s_bg # El subtotal de Patrimonio es el total mismo
        
        t_act = t_act_c + t_act_nc
        t_pas = t_pas_c + t_pas_nc
        
        # Si no hubo patrimonio individual pero sí otras categorías, la fila de subtotal de patrimonio se omite arriba.
        # Aseguramos que 'TOTAL PATRIMONIO' siempre aparezca si t_pat tiene valor o si es la última sección.
        # Esta lógica de totales ya estaba, solo revisando el flujo.

        final_df_bg.loc[len(final_df_bg)] = pd.Series({cuenta_col:'', nombre_col:'TOTAL ACTIVOS', 'Valor':t_act})
        final_df_bg.loc[len(final_df_bg)] = pd.Series({cuenta_col:'', nombre_col:'', 'Valor':None}) # Blank row
        final_df_bg.loc[len(final_df_bg)] = pd.Series({cuenta_col:'', nombre_col:'TOTAL PASIVOS', 'Valor':t_pas})
        final_df_bg.loc[len(final_df_bg)] = pd.Series({cuenta_col:'', nombre_col:'TOTAL PATRIMONIO', 'Valor':t_pat})
        final_df_bg.loc[len(final_df_bg)] = pd.Series({cuenta_col:'', nombre_col:'TOTAL PASIVO + PATRIMONIO', 'Valor':t_pas + t_pat})
        final_df_bg.loc[len(final_df_bg)] = pd.Series({cuenta_col:'', nombre_col:'DIFERENCIA (A-(P+Pt))', 'Valor':t_act - (t_pas + t_pat)})
        return final_df_bg
    return pd.DataFrame(columns=final_cols)

# ... (to_excel_buffer sin cambios) ...
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
        # ... (Carga de datos y procesamiento básico sin cambios) ...
        xls = pd.ExcelFile(uploaded_file)
        if 'EDO RESULTADO' not in xls.sheet_names or 'BALANCE' not in xls.sheet_names:
            st.error("Asegúrate que el archivo Excel contenga las hojas 'EDO RESULTADO' y 'BALANCE'.")
            st.stop()
        df_er_raw, df_bg_raw = pd.read_excel(xls, 'EDO RESULTADO'), pd.read_excel(xls, 'BALANCE')
        st.success("Archivo cargado correctamente.")

        er_conf, bg_conf = COL_CONFIG['ESTADO_DE_RESULTADOS'], COL_CONFIG['BALANCE_GENERAL']
        
        er_map = {k:v for k,v in er_conf['CENTROS_COSTO_COLS'].items() if k in df_er_raw.columns}
        st.session_state.df_er = df_er_raw.rename(columns=er_map).copy()
        
        for value_col_er in er_conf['CENTROS_COSTO_COLS'].values():
            if value_col_er in st.session_state.df_er.columns:
                st.session_state.df_er[value_col_er] = st.session_state.df_er[value_col_er].apply(clean_numeric_value)
        
        for col_key in ['CUENTA', 'NOMBRE_CUENTA', 'NIVEL_LINEA']:
            col = er_conf[col_key]
            if col not in st.session_state.df_er.columns: st.error(f"Columna '{col}' (ER) no encontrada."); st.stop()
            st.session_state.df_er[col] = st.session_state.df_er[col].astype(str).str.strip() if col_key != 'NIVEL_LINEA' else pd.to_numeric(st.session_state.df_er[col], errors='coerce').fillna(0).astype(int)
        st.session_state.df_er['Tipo_Estado'] = st.session_state.df_er[er_conf['CUENTA']].apply(classify_account)

        for col_to_fix_sign in er_conf['CENTROS_COSTO_COLS'].values():
            if col_to_fix_sign in st.session_state.df_er.columns:
                ingresos_mask = st.session_state.df_er['Tipo_Estado'].str.contains('Ingresos', na=False)
                st.session_state.df_er[col_to_fix_sign] = np.where(ingresos_mask, st.session_state.df_er[col_to_fix_sign] * -1, st.session_state.df_er[col_to_fix_sign])
                egresos_mask = (st.session_state.df_er['Tipo_Estado'].str.contains('Costo de Ventas', na=False) |
                                st.session_state.df_er['Tipo_Estado'].str.contains('Gastos Operacionales', na=False) |
                                st.session_state.df_er['Tipo_Estado'].str.contains('Gastos no Operacionales', na=False) |
                                st.session_state.df_er['Tipo_Estado'].str.contains('Impuestos', na=False))
                st.session_state.df_er[col_to_fix_sign] = np.where(egresos_mask, st.session_state.df_er[col_to_fix_sign] * -1, st.session_state.df_er[col_to_fix_sign])

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

# ***** CAMBIO: Filtro de Centro de Costo para ER ahora en el sidebar *****
selected_cc_report = "Todos" # Valor por defecto
if report_type == "Estado de Resultados" and not st.session_state.df_er.empty:
    er_conf_sidebar = COL_CONFIG['ESTADO_DE_RESULTADOS']
    # Los nombres lógicos de los CC (ej. "Armenia") para el selectbox. Excluir el 'Total' genérico de esta selección.
    cc_options_list = [v for k,v in er_conf_sidebar['CENTROS_COSTO_COLS'].items() if k != 'Total' and v in st.session_state.df_er.columns]
    selected_cc_report = st.sidebar.selectbox("Filtrar por Centro de Costo (ER):", 
                                              ['Todos'] + cc_options_list, 
                                              key="cc_filter_er")

st.sidebar.header("Buscar Cuenta Específica")
# Usamos una clave única para el text_input para que no se resetee al cambiar de reporte
search_account_input = st.sidebar.text_input("Número de Cuenta a detallar:", key="search_account_input_main")


# --- Reportes Principales y Detalle de Cuenta Buscada ---
# Moveremos el detalle de la cuenta buscada a la sección principal

# Sección Estado de Resultados
if report_type == "Estado de Resultados" and not st.session_state.df_er.empty:
    st.header(f"📈 Estado de Resultados ({selected_cc_report})") # Mostrar el CC seleccionado
    er_conf_main = COL_CONFIG['ESTADO_DE_RESULTADOS']
    
    # DataFrame para KPIs usa st.session_state.df_er que YA TIENE SIGNOS CORREGIDOS
    # Y debe reflejar el filtro de selected_cc_report
    df_for_kpi_er = st.session_state.df_er.copy()
    
    # Determinar la columna de valor para KPIs basado en selected_cc_report
    val_col_for_kpi_er = ''
    if selected_cc_report and selected_cc_report != 'Todos':
        if selected_cc_report in df_for_kpi_er.columns:
            val_col_for_kpi_er = selected_cc_report
    else: # Consolidado ("Todos")
        val_col_for_kpi_er = er_conf_main['CENTROS_COSTO_COLS'].get('Total') # Ej: 'Total_Consolidado_ER'
        if not (val_col_for_kpi_er and val_col_for_kpi_er in df_for_kpi_er.columns):
            cc_individual_cols_kpi = [v for k,v in er_conf_main['CENTROS_COSTO_COLS'].items() if k != 'Total' and v in df_for_kpi_er.columns]
            if cc_individual_cols_kpi:
                df_for_kpi_er['__temp_sum_for_kpi'] = df_for_kpi_er[cc_individual_cols_kpi].sum(axis=1)
                val_col_for_kpi_er = '__temp_sum_for_kpi'
            else:
                val_col_for_kpi_er = None # No hay cómo consolidar

    kpi_er_ingresos_val = 0.0
    kpi_er_costos_y_gastos_totales_val = 0.0 # Nuevo KPI
    kpi_er_costo_ventas_val = 0.0
    kpi_er_gastos_op_val = 0.0
    kpi_er_gastos_no_op_val = 0.0
    kpi_er_impuestos_val = 0.0
    kpi_er_utilidad_neta_val = 0.0

    if val_col_for_kpi_er and val_col_for_kpi_er in df_for_kpi_er.columns:
        kpi_er_ingresos_val = df_for_kpi_er[df_for_kpi_er['Tipo_Estado'] == 'Estado de Resultados - Ingresos'][val_col_for_kpi_er].sum()
        kpi_er_costo_ventas_val = df_for_kpi_er[df_for_kpi_er['Tipo_Estado'] == 'Estado de Resultados - Costo de Ventas'][val_col_for_kpi_er].sum() # será negativo
        kpi_er_gastos_op_val = df_for_kpi_er[df_for_kpi_er['Tipo_Estado'] == 'Estado de Resultados - Gastos Operacionales'][val_col_for_kpi_er].sum() # será negativo
        kpi_er_gastos_no_op_val = df_for_kpi_er[df_for_kpi_er['Tipo_Estado'] == 'Estado de Resultados - Gastos no Operacionales'][val_col_for_kpi_er].sum() # será negativo
        kpi_er_impuestos_val = df_for_kpi_er[df_for_kpi_er['Tipo_Estado'] == 'Estado de Resultados - Impuestos'][val_col_for_kpi_er].sum() # será negativo
        
        # Total Costos y Gastos: Suma de los valores absolutos de las cantidades negativas
        kpi_er_costos_y_gastos_totales_val = abs(kpi_er_costo_ventas_val) + abs(kpi_er_gastos_op_val) + abs(kpi_er_gastos_no_op_val) + abs(kpi_er_impuestos_val)
    else:
        st.warning(f"No se pudo determinar la columna de valor para calcular KPIs del ER ({selected_cc_report}).")

    kpi_er_utilidad_op_val = kpi_er_ingresos_val + kpi_er_costo_ventas_val + kpi_er_gastos_op_val
    kpi_er_margen_op_val = (kpi_er_utilidad_op_val / kpi_er_ingresos_val) * 100 if kpi_er_ingresos_val else 0.0
    
    # La Utilidad Neta se tomará del total de la tabla generada, que ya considera todos los egresos
    
    # ***** CAMBIO: KPIs para ER ajustados *****
    kpi_cols_er_main = st.columns(3) 
    # kpi_cols_er_main[0].metric("Ingresos Totales", f"${kpi_er_ingresos_val:,.0f}") # Eliminado
    kpi_cols_er_main[0].metric("Total Costos y Gastos", f"${kpi_er_costos_y_gastos_totales_val:,.0f}")
    kpi_cols_er_main[1].metric("Utilidad Operativa", f"${kpi_er_utilidad_op_val:,.0f}", f"{kpi_er_margen_op_val:.1f}% Margen Op.")
    # La Utilidad Neta se llenará después de generar la tabla.

    er_niv_col_main = er_conf_main['NIVEL_LINEA']
    if er_niv_col_main in st.session_state.df_er.columns:
        lvls_er_main = sorted(st.session_state.df_er[er_niv_col_main].astype(int).unique().tolist())
        min_l_er, max_l_er = (min(lvls_er_main), max(lvls_er_main)) if lvls_er_main else (1,1)
        # Usar una clave única para el slider para que no interfiera con otros sliders
        max_lvl_er_main = st.sidebar.slider("Nivel Detalle (ER):", min_value=min_l_er, max_value=max_l_er, value=min_l_er, key="slider_er_level_main")
        
        # Pasar el selected_cc_report a generate_financial_statement
        st.session_state.final_er_display = generate_financial_statement(st.session_state.df_er, 'Estado de Resultados', selected_cc_report, max_lvl_er_main)
        
        if not st.session_state.final_er_display.empty:
            total_er_row_main = st.session_state.final_er_display[st.session_state.final_er_display[er_conf_main['NOMBRE_CUENTA']] == 'TOTAL ESTADO DE RESULTADOS']
            if not total_er_row_main.empty:
                kpi_er_utilidad_neta_val = total_er_row_main['Valor'].iloc[0]
                kpi_er_margen_neto_val = (kpi_er_utilidad_neta_val / kpi_er_ingresos_val) * 100 if kpi_er_ingresos_val else 0.0
                kpi_cols_er_main[2].metric("Utilidad Neta", f"${kpi_er_utilidad_neta_val:,.0f}", f"{kpi_er_margen_neto_val:.1f}% Margen Neto")
        
        st.dataframe(st.session_state.final_er_display, use_container_width=True, hide_index=True)

# Sección Balance General
elif report_type == "Balance General" and not st.session_state.df_bg.empty:
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

    kpi_bg_rc = (kpi_bg_ac / kpi_bg_pc) if kpi_bg_pc else 0.0
    kpi_bg_end = (kpi_bg_pt / kpi_bg_at) * 100 if kpi_bg_at else 0.0
    kpi_bg_pp = (kpi_bg_pt / kpi_bg_pat) if kpi_bg_pat else 0.0

    kpi_cols_bg_main = st.columns(4)
    kpi_cols_bg_main[0].metric("Total Activos", f"${kpi_bg_at:,.0f}")
    kpi_cols_bg_main[1].metric("Total Pasivos", f"${kpi_bg_pt:,.0f}", f"{kpi_bg_end:.1f}% Endeud.")
    kpi_cols_bg_main[2].metric("Total Patrimonio", f"${kpi_bg_pat:,.0f}", f"{kpi_bg_pp:.2f} Pas/Pat")
    kpi_cols_bg_main[3].metric("Razón Corriente", f"{kpi_bg_rc:.2f}") # Indicador de Liquidez

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


# ***** NUEVO: Detalle de Cuenta Buscada en la página principal *****
if search_account_input: # Solo mostrar si se ha buscado algo
    with st.expander(f"Detalle y Subcuentas para la Cuenta '{search_account_input}'", expanded=False):
        df_for_search_detail = pd.DataFrame()
        config_for_search_detail = {}
        is_er_detail_search = False
        original_df_for_subaccounts = pd.DataFrame() # Usaremos el df original antes de get_top_level

        if report_type == "Estado de Resultados" and not st.session_state.df_er.empty:
            # Para subcuentas, usamos el df_er completo antes de la agregación de get_top_level_accounts_for_display
            original_df_for_subaccounts = st.session_state.df_er 
            config_for_search_detail = COL_CONFIG['ESTADO_DE_RESULTADOS']
            is_er_detail_search = True
        elif report_type == "Balance General" and not st.session_state.df_bg.empty:
            original_df_for_subaccounts = st.session_state.df_bg
            config_for_search_detail = COL_CONFIG['BALANCE_GENERAL']
            is_er_detail_search = False # Es BG

        if not original_df_for_subaccounts.empty and config_for_search_detail:
            cuenta_col_s = config_for_search_detail['CUENTA']
            nombre_col_s = config_for_search_detail['NOMBRE_CUENTA']
            
            # Filtrar todas las cuentas que comienzan con el número buscado (para incluir subcuentas)
            # Asegurarse de que la columna de cuenta sea string para .startswith()
            sub_accounts_df = original_df_for_subaccounts[
                original_df_for_subaccounts[cuenta_col_s].astype(str).str.startswith(search_account_input)
            ].copy()

            if not sub_accounts_df.empty:
                st.write(f"Mostrando la cuenta '{search_account_input}' y sus subcuentas:")
                
                if is_er_detail_search:
                    # Mostrar Cuenta, Título, y todos los centros de costo + Total Consolidado
                    cols_to_show_er = [cuenta_col_s, nombre_col_s] + \
                                      list(config_for_search_detail['CENTROS_COSTO_COLS'].values())
                    # Asegurarse que todas las columnas existan en sub_accounts_df
                    cols_to_show_er_existing = [col for col in cols_to_show_er if col in sub_accounts_df.columns]
                    st.dataframe(sub_accounts_df[cols_to_show_er_existing], use_container_width=True, hide_index=True)
                else: # Balance General
                    cols_to_show_bg = [
                        cuenta_col_s, nombre_col_s,
                        config_for_search_detail['SALDO_INICIAL'],
                        config_for_search_detail['DEBE'],
                        config_for_search_detail['HABER'],
                        config_for_search_detail['SALDO_FINAL']
                    ]
                    cols_to_show_bg_existing = [col for col in cols_to_show_bg if col in sub_accounts_df.columns]
                    st.dataframe(sub_accounts_df[cols_to_show_bg_existing], use_container_width=True, hide_index=True)
            else:
                st.info(f"No se encontró la cuenta '{search_account_input}' o subcuentas asociadas en el reporte de {report_type}.")
        elif search_account_input: # Si se buscó algo pero no hay datos o config
             st.info(f"No hay datos cargados para el reporte de {report_type} para buscar la cuenta.")


# --- Descarga ---
st.sidebar.markdown("---")
# ... (sin cambios) ...
st.sidebar.subheader("Exportar a Excel")
er_dl = st.session_state.final_er_display if not st.session_state.final_er_display.empty else st.session_state.df_er
bg_dl = st.session_state.final_bg_display if not st.session_state.final_bg_display.empty else st.session_state.df_bg
disable_dl = er_dl.empty and bg_dl.empty

excel_buffer = to_excel_buffer(er_dl, bg_dl)
st.sidebar.download_button(
    label="Descargar Reporte Seleccionado", data=excel_buffer,
    file_name=f"reporte_{report_type.lower().replace(' ', '_')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    disabled=disable_dl
)
