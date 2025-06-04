import pandas as pd
import streamlit as st
import io
import numpy as np # Necesario para np.where

# --- Configuración de Columnas ---
COL_CONFIG = {
    'ESTADO_DE_RESULTADOS': {
        'NIVEL_LINEA': 'Grupo',
        'CUENTA': 'Cuenta',
        'NOMBRE_CUENTA': 'Título',
        'CENTROS_COSTO_COLS': {
            'Sin centro de coste': 'Sin centro de coste',
            '156': 'Armenia',
            '157': 'San antonio',
            '158': 'Opalo',
            '189': 'Olaya',
            '238': 'Laureles',
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
    
    # Balance General - Modificado para liquidez
    if cuenta_str.startswith('11'): return 'Balance General - Activo Corriente'
    elif cuenta_str.startswith('1'): return 'Balance General - Activo No Corriente' 
    elif cuenta_str.startswith('21'): return 'Balance General - Pasivo Corriente'
    elif cuenta_str.startswith('2'): return 'Balance General - Pasivo No Corriente' 
    elif cuenta_str.startswith('3'): return 'Balance General - Patrimonio'
    
    # Estado de Resultados
    elif cuenta_str.startswith('4'): return 'Estado de Resultados - Ingresos'
    elif cuenta_str.startswith('5'): return 'Estado de Resultados - Ingresos' # O 'Otros Ingresos'
    elif cuenta_str.startswith('6'): return 'Estado de Resultados - Costo de Ventas'
    elif cuenta_str.startswith('7'): return 'Estado de Resultados - Gastos Operacionales'
    elif cuenta_str.startswith('8'): return 'Estado de Resultados - Gastos no Operacionales'
    elif cuenta_str.startswith('9'): return 'Estado de Resultados - Impuestos'
    else: return 'No Clasificado'

def get_top_level_accounts_for_display(df_raw: pd.DataFrame, value_col_name: str, statement_type: str) -> pd.DataFrame:
    # ... (sin cambios significativos, pero se beneficia de datos con signos correctos si es ER)
    cuenta_col_key = statement_type.replace(' ', '_').upper()
    if cuenta_col_key not in COL_CONFIG:
        return pd.DataFrame()
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

    levels_map = {'Estado de Resultados': [1,2,3,4,6,8], 'Balance General': [1,2,3,4]} # Estos niveles son para cuentas con valor CERO
    zero_levels = levels_map.get(statement_type, [])
    df_zero_sig = df_sorted[(df_sorted[value_col_name].abs() < 0.001) & (pd.to_numeric(df_sorted[nivel_linea_col], errors='coerce').isin(zero_levels))].copy()
    
    if df_result.empty and df_zero_sig.empty: return pd.DataFrame(columns=df_sorted.columns)
    df_final = pd.concat([df_result, df_zero_sig]).drop_duplicates(subset=['Cuenta_Str']).reset_index(drop=True)
    return df_final.sort_values(by=[nivel_linea_col, 'Cuenta_Str']) if not df_final.empty else df_final


def generate_financial_statement(df_full_data: pd.DataFrame, statement_type: str, filter_cc: str = None, max_level: int = 999) -> pd.DataFrame:
    # ... (sin cambios en la lógica de suma, asume que df_full_data ya tiene signos correctos para ER)
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

        value_col_name = ''
        if filter_cc and filter_cc != 'Todos':
            value_col_name = filter_cc if filter_cc in df_statement.columns else 'Valor_Final_Fallback_ER'
            if value_col_name == 'Valor_Final_Fallback_ER': df_statement[value_col_name] = 0.0
        else:
            total_er_col = config['CENTROS_COSTO_COLS'].get('Total')
            if total_er_col and total_er_col in df_statement.columns: value_col_name = total_er_col
            else:
                cc_sum_cols = [n for ex_n, n in config['CENTROS_COSTO_COLS'].items() if ex_n!='Total' and n in df_statement.columns]
                if cc_sum_cols:
                    df_statement['Temp_Sum_ER'] = df_statement[cc_sum_cols].sum(axis=1)
                    value_col_name = 'Temp_Sum_ER'
                else:
                    df_statement['Valor_Final_Fallback_ER'] = 0.0
                    value_col_name = 'Valor_Final_Fallback_ER'
        
        df_statement['Valor_Final'] = df_statement[value_col_name]
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
        final_df.loc[len(final_df)] = pd.Series({cuenta_col:'', nombre_col:'', 'Valor':None}) # Blank row
        return final_df

    elif statement_type == 'Balance General':
        # Esta función ahora también se beneficia de la clasificación más detallada (Corriente/No Corriente)
        # si se quiere mostrar esos subtotales en la tabla principal, pero la lógica actual no lo hace.
        # Solo agrupa por Activo, Pasivo, Patrimonio.
        df_statement = df_full_data[df_full_data['Tipo_Estado'].str.contains('Balance General', na=False)].copy()
        if df_statement.empty: return pd.DataFrame(columns=final_cols)
        
        saldo_col = config.get('SALDO_FINAL')
        df_statement['Valor_Final'] = df_statement[saldo_col] if saldo_col and saldo_col in df_statement.columns else 0.0
        
        df_display = get_top_level_accounts_for_display(df_statement, 'Valor_Final', statement_type)
        if df_display.empty: return pd.DataFrame(columns=final_cols)
        df_display = df_display[pd.to_numeric(df_display[nivel_col], errors='coerce').fillna(9999) <= float(max_level)].copy()

        # El orden actual solo considera Activo, Pasivo, Patrimonio.
        # Para mostrar Activo Corriente, etc., se necesitaría un orden más detallado.
        order_categories_bg = ['Activo Corriente', 'Activo No Corriente', 'Pasivo Corriente', 'Pasivo No Corriente', 'Patrimonio']
        order_bg = [f'Balance General - {i}' for i in order_categories_bg]
        
        final_df_bg = pd.DataFrame(columns=final_cols)
        t_act_c, t_act_nc, t_pas_c, t_pas_nc, t_pat = 0.0,0.0,0.0,0.0,0.0
        if 'Tipo_Estado' not in df_display.columns: return final_df_bg

        # Crear subtotales para la tabla principal
        subtotals_data = []

        for tipo_raw in order_categories_bg:
            tipo = f'Balance General - {tipo_raw}'
            group = df_display[df_display['Tipo_Estado'] == tipo].copy()
            if not group.empty:
                group = group.sort_values(by=cuenta_col)
                group['Nombre_Cuenta_Display'] = group.apply(lambda r: f"{'  '*(int(pd.to_numeric(r[nivel_col],errors='coerce')or 1)-1)}{r[nombre_col]}", axis=1)
                group['Valor_Final'] = group['Valor_Final'].fillna(0)
                final_df_bg = pd.concat([final_df_bg, group[[cuenta_col, 'Nombre_Cuenta_Display', 'Valor_Final']].rename(columns={'Nombre_Cuenta_Display':nombre_col, 'Valor_Final':'Valor'})], ignore_index=True)
                
                current_s = pd.to_numeric(group['Valor_Final'], errors='coerce').sum()
                subtotals_data.append({'Cuenta': '', 'Título': f'Subtotal {tipo_raw}', 'Valor': current_s})

                if tipo_raw == 'Activo Corriente': t_act_c += current_s
                elif tipo_raw == 'Activo No Corriente': t_act_nc += current_s
                elif tipo_raw == 'Pasivo Corriente': t_pas_c += current_s
                elif tipo_raw == 'Pasivo No Corriente': t_pas_nc += current_s
                elif tipo_raw == 'Patrimonio': t_pat += current_s
            
            # Añadir fila de subtotal si es relevante (ej. después de todas las cuentas de Activo Corriente)
            if not group.empty and ("Activo" in tipo_raw or "Pasivo" in tipo_raw) : # And es el fin de su sección
                 final_df_bg.loc[len(final_df_bg)] = pd.Series({cuenta_col:'', nombre_col:f'SUBTOTAL {tipo_raw.upper()}', 'Valor':current_s})


        t_act = t_act_c + t_act_nc
        t_pas = t_pas_c + t_pas_nc
        
        final_df_bg.loc[len(final_df_bg)] = pd.Series({cuenta_col:'', nombre_col:'TOTAL ACTIVOS', 'Valor':t_act})
        final_df_bg.loc[len(final_df_bg)] = pd.Series({cuenta_col:'', nombre_col:'', 'Valor':None})
        final_df_bg.loc[len(final_df_bg)] = pd.Series({cuenta_col:'', nombre_col:'TOTAL PASIVOS', 'Valor':t_pas})
        final_df_bg.loc[len(final_df_bg)] = pd.Series({cuenta_col:'', nombre_col:'TOTAL PATRIMONIO', 'Valor':t_pat})
        final_df_bg.loc[len(final_df_bg)] = pd.Series({cuenta_col:'', nombre_col:'TOTAL PASIVO + PATRIMONIO', 'Valor':t_pas + t_pat})
        final_df_bg.loc[len(final_df_bg)] = pd.Series({cuenta_col:'', nombre_col:'DIFERENCIA', 'Valor':t_act - (t_pas + t_pat)})
        return final_df_bg
    return pd.DataFrame(columns=final_cols)


def to_excel_buffer(er_df: pd.DataFrame, bg_df: pd.DataFrame) -> io.BytesIO:
    # ... (sin cambios)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if er_df is not None and not er_df.empty: er_df.to_excel(writer, sheet_name='Estado de Resultados', index=False)
        if bg_df is not None and not bg_df.empty: bg_df.to_excel(writer, sheet_name='Balance General', index=False)
    output.seek(0)
    return output

# --- Aplicación Streamlit ---
st.set_page_config(layout="wide", page_title="Análisis Financiero Avanzado")
st.title("💰 Análisis Financiero y Tablero Gerencial")

for key in ['df_er', 'df_bg', 'final_er_display', 'final_bg_display', 'er_kpis', 'bg_kpis']:
    if key not in st.session_state: st.session_state[key] = pd.DataFrame() if 'df' in key else {} if 'kpis' in key else pd.DataFrame()

uploaded_file = st.file_uploader("Sube tu archivo Excel (hojas 'EDO RESULTADO' y 'BALANCE')", type=["xlsx"])

if uploaded_file is not None:
    try:
        xls = pd.ExcelFile(uploaded_file)
        if 'EDO RESULTADO' not in xls.sheet_names or 'BALANCE' not in xls.sheet_names:
            st.error("Asegúrate que el archivo Excel contenga las hojas 'EDO RESULTADO' y 'BALANCE'.")
            st.stop()
        df_er_raw, df_bg_raw = pd.read_excel(xls, 'EDO RESULTADO'), pd.read_excel(xls, 'BALANCE')
        st.success("Archivo cargado correctamente.")

        er_conf, bg_conf = COL_CONFIG['ESTADO_DE_RESULTADOS'], COL_CONFIG['BALANCE_GENERAL']
        
        # Procesamiento ER
        er_map = {k:v for k,v in er_conf['CENTROS_COSTO_COLS'].items() if k in df_er_raw.columns}
        st.session_state.df_er = df_er_raw.rename(columns=er_map).copy()
        
        for cc_name_val_col in er_conf['CENTROS_COSTO_COLS'].values():
            if cc_name_val_col in st.session_state.df_er.columns:
                st.session_state.df_er[cc_name_val_col] = st.session_state.df_er[cc_name_val_col].apply(clean_numeric_value)
        
        for col_key in ['CUENTA', 'NOMBRE_CUENTA', 'NIVEL_LINEA']:
            col = er_conf[col_key]
            if col not in st.session_state.df_er.columns: st.error(f"Columna '{col}' (ER) no encontrada."); st.stop()
            st.session_state.df_er[col] = st.session_state.df_er[col].astype(str).str.strip() if col_key != 'NIVEL_LINEA' else pd.to_numeric(st.session_state.df_er[col], errors='coerce').fillna(0).astype(int)
        st.session_state.df_er['Tipo_Estado'] = st.session_state.df_er[er_conf['CUENTA']].apply(classify_account)

        # ***** CAMBIO: CORRECCIÓN DE SIGNOS PARA ESTADO DE RESULTADOS con np.where *****
        for col_to_fix in er_conf['CENTROS_COSTO_COLS'].values():
            if col_to_fix in st.session_state.df_er.columns:
                # Ingresos: de negativo en Excel a POSITIVO
                ingresos_mask = st.session_state.df_er['Tipo_Estado'].str.contains('Ingresos', na=False)
                st.session_state.df_er[col_to_fix] = np.where(ingresos_mask, 
                                                             st.session_state.df_er[col_to_fix] * -1, 
                                                             st.session_state.df_er[col_to_fix])

                # Egresos (Costos, Gastos, Impuestos): de positivo en Excel a NEGATIVO
                egresos_mask = (st.session_state.df_er['Tipo_Estado'].str.contains('Costo de Ventas', na=False) |
                                st.session_state.df_er['Tipo_Estado'].str.contains('Gastos Operacionales', na=False) |
                                st.session_state.df_er['Tipo_Estado'].str.contains('Gastos no Operacionales', na=False) |
                                st.session_state.df_er['Tipo_Estado'].str.contains('Impuestos', na=False))
                st.session_state.df_er[col_to_fix] = np.where(egresos_mask, 
                                                             st.session_state.df_er[col_to_fix] * -1, 
                                                             st.session_state.df_er[col_to_fix])
        # ***** FIN DE CORRECCIÓN DE SIGNOS *****

        # Procesamiento BG
        st.session_state.df_bg = df_bg_raw.copy()
        for col_key in ['SALDO_INICIAL', 'DEBE', 'HABER', 'SALDO_FINAL', 'CUENTA', 'NOMBRE_CUENTA', 'NIVEL_LINEA']:
            col = bg_conf[col_key]
            if col not in st.session_state.df_bg.columns: st.error(f"Columna '{col}' (BG) no encontrada."); st.stop()
            if col_key not in ['CUENTA', 'NOMBRE_CUENTA', 'NIVEL_LINEA']: st.session_state.df_bg[col] = st.session_state.df_bg[col].apply(clean_numeric_value)
            else: st.session_state.df_bg[col] = st.session_state.df_bg[col].astype(str).str.strip() if col_key != 'NIVEL_LINEA' else pd.to_numeric(st.session_state.df_bg[col], errors='coerce').fillna(0).astype(int)
        st.session_state.df_bg['Tipo_Estado'] = st.session_state.df_bg[bg_conf['CUENTA']].apply(classify_account)

    except Exception as e: st.error(f"Error al procesar archivo: {e}"); st.exception(e)

# --- Interfaz de Usuario ---
# ... (Sidebar sin cambios significativos, excepto que ahora hay más tipos de cuenta para BG)
st.sidebar.header("Opciones de Reporte")
report_type = st.sidebar.radio("Selecciona el reporte:", ["Estado de Resultados", "Balance General"],
                               disabled=st.session_state.df_er.empty and st.session_state.df_bg.empty)

st.sidebar.header("Buscar Cuenta Específica")
selected_account_number = st.sidebar.text_input("Buscar por Número de Cuenta:")
# ... (Lógica de búsqueda de cuenta sin cambios, pero usará df_er con signos corregidos)
if selected_account_number:
    cuenta_details_df = pd.DataFrame()
    df_to_search = pd.DataFrame()
    config_to_use = {}
    is_er = False

    if report_type == "Estado de Resultados" and not st.session_state.df_er.empty:
        df_to_search = st.session_state.df_er 
        config_to_use = COL_CONFIG['ESTADO_DE_RESULTADOS']
        is_er = True
    elif report_type == "Balance General" and not st.session_state.df_bg.empty:
        df_to_search = st.session_state.df_bg
        config_to_use = COL_CONFIG['BALANCE_GENERAL']

    if not df_to_search.empty:
        cuenta_col_name = config_to_use['CUENTA']
        nombre_cuenta_col_name = config_to_use['NOMBRE_CUENTA']
        
        account_data = df_to_search[df_to_search[cuenta_col_name] == selected_account_number]
        if account_data.empty:
             account_data = df_to_search[df_to_search[cuenta_col_name].str.contains(selected_account_number, case=False, na=False)]

        if not account_data.empty:
            st.sidebar.subheader(f"Detalle Cuenta: {selected_account_number}")
            for idx, row in account_data.iterrows():
                st.sidebar.write(f"**Nombre:** {row[nombre_cuenta_col_name]}")
                if is_er:
                    st.sidebar.write("**Valores por Centro de Costo (Signos corregidos):**")
                    cc_data = {}
                    for cc_excel_name, cc_display_name in config_to_use['CENTROS_COSTO_COLS'].items():
                        if cc_display_name in row and cc_excel_name != 'Total':
                             cc_data[cc_display_name] = row.get(cc_display_name, 0.0)
                    
                    total_consolidado_key = config_to_use['CENTROS_COSTO_COLS'].get('Total')
                    if total_consolidado_key and total_consolidado_key in row:
                        cc_data[f"**{total_consolidado_key.replace('_',' ')}**"] = row[total_consolidado_key]

                    st.sidebar.dataframe(pd.DataFrame(list(cc_data.items()), columns=['Centro de Costo/Total', 'Valor']))
                else: 
                    bg_details = { "Saldo Inicial": row.get(config_to_use['SALDO_INICIAL'], 0.0), "Debe": row.get(config_to_use['DEBE'], 0.0), "Haber": row.get(config_to_use['HABER'], 0.0), "Saldo Final": row.get(config_to_use['SALDO_FINAL'], 0.0), }
                    st.sidebar.dataframe(pd.DataFrame(list(bg_details.items()), columns=['Concepto', 'Valor']))
                st.sidebar.markdown("---")
        else:
            st.sidebar.info(f"No se encontró la cuenta '{selected_account_number}' en el reporte de {report_type}.")


# --- Reportes Principales ---
if report_type == "Estado de Resultados" and not st.session_state.df_er.empty:
    st.header("📈 Estado de Resultados")
    er_conf_ui = COL_CONFIG['ESTADO_DE_RESULTADOS']
    
    # ***** CAMBIO: Lógica de KPIs con signos corregidos y datos de st.session_state.df_er *****
    er_df_kpi = st.session_state.df_er.copy() # Esta es la df_er con signos ya corregidos
    total_col_er_kpi = er_conf_ui['CENTROS_COSTO_COLS'].get('Total')
    val_col_kpi_er = None
    
    if total_col_er_kpi and total_col_er_kpi in er_df_kpi.columns:
        val_col_kpi_er = total_col_er_kpi
    else: # Si no hay columna 'Total', sumar los CCs
        cc_sum_cols_kpi = [n for ex_n, n in er_conf_ui['CENTROS_COSTO_COLS'].items() if ex_n!='Total' and n in er_df_kpi.columns]
        if cc_sum_cols_kpi:
            er_df_kpi['__temp_sum_kpi_er'] = er_df_kpi[cc_sum_cols_kpi].sum(axis=1)
            val_col_kpi_er = '__temp_sum_kpi_er'

    kpi_er_ingresos = 0.0
    kpi_er_costo_ventas_val = 0.0 
    kpi_er_gastos_op_val = 0.0    
    kpi_er_utilidad_neta = 0.0 # Se actualizará después de generar la tabla

    if val_col_kpi_er and val_col_kpi_er in er_df_kpi.columns: # Asegurar que la columna de valor exista
        kpi_er_ingresos = er_df_kpi[er_df_kpi['Tipo_Estado'] == 'Estado de Resultados - Ingresos'][val_col_kpi_er].sum()
        kpi_er_costo_ventas_val = er_df_kpi[er_df_kpi['Tipo_Estado'] == 'Estado de Resultados - Costo de Ventas'][val_col_kpi_er].sum()
        kpi_er_gastos_op_val = er_df_kpi[er_df_kpi['Tipo_Estado'] == 'Estado de Resultados - Gastos Operacionales'][val_col_kpi_er].sum()
    else:
        st.warning("No se pudo determinar la columna consolidada para calcular KPIs del ER. Revise la configuración de 'Total' en CENTROS_COSTO_COLS.")


    kpi_er_utilidad_bruta = kpi_er_ingresos + kpi_er_costo_ventas_val # Ingresos(+) y Costos(-)
    kpi_er_margen_bruto = (kpi_er_utilidad_bruta / kpi_er_ingresos) * 100 if kpi_er_ingresos else 0.0
    
    kpi_er_utilidad_op = kpi_er_utilidad_bruta + kpi_er_gastos_op_val # U.Bruta(+) y GastosOp(-)
    kpi_er_margen_op = (kpi_er_utilidad_op / kpi_er_ingresos) * 100 if kpi_er_ingresos else 0.0
    
    kpi_cols_er = st.columns(4)
    kpi_cols_er[0].metric("Ingresos Totales", f"${kpi_er_ingresos:,.0f}")
    # Para el delta de st.metric, un margen positivo es bueno.
    kpi_cols_er[1].metric("Utilidad Bruta", f"${kpi_er_utilidad_bruta:,.0f}", f"{kpi_er_margen_bruto:.1f}% Margen")
    kpi_cols_er[2].metric("Utilidad Operacional", f"${kpi_er_utilidad_op:,.0f}", f"{kpi_er_margen_op:.1f}% Margen")

    selected_cc = st.sidebar.selectbox("Centro de Costo (ER):", ['Todos'] + [n for ex,n in er_conf_ui['CENTROS_COSTO_COLS'].items() if ex!='Total' and n in st.session_state.df_er.columns])
    er_niv_col = er_conf_ui['NIVEL_LINEA']
    if er_niv_col in st.session_state.df_er.columns:
        lvls = sorted(st.session_state.df_er[er_niv_col].astype(int).unique().tolist())
        min_l, max_l = (min(lvls), max(lvls)) if lvls else (1,1)
        max_lvl_er = st.sidebar.slider("Nivel Detalle (ER):", min_value=min_l, max_value=max_l, value=min_l)
        st.session_state.final_er_display = generate_financial_statement(st.session_state.df_er, 'Estado de Resultados', selected_cc, max_lvl_er)
        
        if not st.session_state.final_er_display.empty:
            total_er_row = st.session_state.final_er_display[st.session_state.final_er_display[er_conf_ui['NOMBRE_CUENTA']] == 'TOTAL ESTADO DE RESULTADOS']
            if not total_er_row.empty:
                kpi_er_utilidad_neta = total_er_row['Valor'].iloc[0]
                kpi_er_margen_neto = (kpi_er_utilidad_neta / kpi_er_ingresos) * 100 if kpi_er_ingresos else 0.0
                kpi_cols_er[3].metric("Utilidad Neta", f"${kpi_er_utilidad_neta:,.0f}", f"{kpi_er_margen_neto:.1f}% Margen")
        
        st.dataframe(st.session_state.final_er_display, use_container_width=True, hide_index=True)

elif report_type == "Balance General" and not st.session_state.df_bg.empty:
    st.header("⚖️ Balance General") 
    bg_conf_ui = COL_CONFIG['BALANCE_GENERAL']

    # ***** NUEVO: KPIs de Liquidez (Razón Corriente) y otros del BG *****
    df_bg_kpi = st.session_state.df_bg.copy() # Usar df_bg para cálculos antes de agregación de tabla
    saldo_final_col_bg_kpi = bg_conf_ui['SALDO_FINAL']
    
    kpi_bg_activo_c, kpi_bg_pasivo_c = 0.0, 0.0
    kpi_bg_activo_total, kpi_bg_pasivo_total, kpi_bg_patrimonio_total = 0.0,0.0,0.0

    if saldo_final_col_bg_kpi in df_bg_kpi.columns and 'Tipo_Estado' in df_bg_kpi.columns:
        kpi_bg_activo_c = df_bg_kpi[df_bg_kpi['Tipo_Estado'] == 'Balance General - Activo Corriente'][saldo_final_col_bg_kpi].sum()
        kpi_bg_pasivo_c = df_bg_kpi[df_bg_kpi['Tipo_Estado'] == 'Balance General - Pasivo Corriente'][saldo_final_col_bg_kpi].sum()
        
        # Totales para otros KPIs (se podrían tomar de final_bg_display también)
        kpi_bg_activo_total = df_bg_kpi[df_bg_kpi['Tipo_Estado'].str.contains('Activo', na=False)][saldo_final_col_bg_kpi].sum()
        kpi_bg_pasivo_total = df_bg_kpi[df_bg_kpi['Tipo_Estado'].str.contains('Pasivo', na=False)][saldo_final_col_bg_kpi].sum()
        kpi_bg_patrimonio_total = df_bg_kpi[df_bg_kpi['Tipo_Estado'] == 'Balance General - Patrimonio'][saldo_final_col_bg_kpi].sum()
    else:
        st.warning("No se pudieron calcular KPIs del BG. Columnas 'SALDO_FINAL' o 'Tipo_Estado' faltantes.")

    kpi_bg_razon_corriente = (kpi_bg_activo_c / kpi_bg_pasivo_c) if kpi_bg_pasivo_c else 0.0
    kpi_bg_endeudamiento = (kpi_bg_pasivo_total / kpi_bg_activo_total) * 100 if kpi_bg_activo_total else 0.0
    kpi_bg_pas_pat = (kpi_bg_pasivo_total / kpi_bg_patrimonio_total) if kpi_bg_patrimonio_total else 0.0

    kpi_cols_bg = st.columns(4) # Añadida una columna para Razón Corriente
    kpi_cols_bg[0].metric("Total Activos", f"${kpi_bg_activo_total:,.0f}")
    kpi_cols_bg[1].metric("Total Pasivos", f"${kpi_bg_pasivo_total:,.0f}", f"{kpi_bg_endeudamiento:.1f}% Endeud.")
    kpi_cols_bg[2].metric("Total Patrimonio", f"${kpi_bg_patrimonio_total:,.0f}", f"{kpi_bg_pas_pat:.2f} Pas/Pat")
    kpi_cols_bg[3].metric("Razón Corriente", f"{kpi_bg_razon_corriente:.2f}")


    bg_niv_col = bg_conf_ui['NIVEL_LINEA']
    if bg_niv_col in st.session_state.df_bg.columns:
        lvls_bg = sorted(st.session_state.df_bg[bg_niv_col].astype(int).unique().tolist())
        min_l_bg, max_l_bg = (min(lvls_bg), max(lvls_bg)) if lvls_bg else (1,1)
        max_lvl_bg = st.sidebar.slider("Nivel Detalle (BG):", min_value=min_l_bg, max_value=max_l_bg, value=min_l_bg)
        st.session_state.final_bg_display = generate_financial_statement(st.session_state.df_bg, 'Balance General', max_level=max_lvl_bg)
        st.dataframe(st.session_state.final_bg_display, use_container_width=True, hide_index=True)
        
        if kpi_bg_activo_total or kpi_bg_pasivo_total or kpi_bg_patrimonio_total : # Usar los totales ya calculados para el gráfico
            st.subheader("Composición del Balance")
            chart_bg_data = pd.DataFrame({'Categoría': ['Activos', 'Pasivos', 'Patrimonio'],'Valor': [kpi_bg_activo_total, kpi_bg_pasivo_total, kpi_bg_patrimonio_total]})
            st.bar_chart(chart_bg_data.set_index('Categoría'))

elif (st.session_state.df_er.empty or st.session_state.df_bg.empty) and uploaded_file:
    st.info("Procesando datos... o verifique errores previos.")
elif not uploaded_file:
    st.info("👆 Sube tu archivo Excel para comenzar el análisis.")

# --- Descarga ---
# ... (sin cambios)
st.sidebar.markdown("---")
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
