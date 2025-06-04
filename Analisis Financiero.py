import pandas as pd
import streamlit as st
import io

# --- Configuración de Columnas (¡AHORA MÁS PRECISA!) ---
# Esta configuración es CRUCIAL y debe coincidir EXACTAMENTE con los nombres
# de tus columnas en el archivo Excel.
COL_CONFIG = {
    'EDO_RESULTADO': {
        'NIVEL_LINEA': 'Grupo',
        'CUENTA': 'Cuenta',
        'NOMBRE_CUENTA': 'Título',
        'CENTROS_COSTO_COLS': {
            # Mapeo de nombres de columnas del Excel a nombres lógicos.
            # La clave es el nombre tal cual aparece en el Excel (ej. 'Sin centro de coste', '156')
            # El valor es el nombre que quieres mostrar en la app (ej. 'Sin centro de coste', 'Armenia')
            'Sin centro de coste': 'Sin centro de coste',
            '156': 'Armenia',
            '157': 'San antonio',
            '158': 'Opalo',
            '189': 'Olaya',
            '238': 'Laureles',
            'Total': 'Total_Consolidado_ER' # La columna 'Total' en EDO RESULTADO
        }
    },
    'BALANCE': {
        'NIVEL_LINEA': 'Grupo',
        'CUENTA': 'Cuenta',
        'NOMBRE_CUENTA': 'Título',
        'SALDO_INICIAL': 'Saldo inicial',
        'DEBE': 'Debe',
        'HABER': 'Haber',
        'SALDO_FINAL': 'Saldo Final' # Esta será la columna de valor para el balance
    }
}

# --- Funciones de Utilidad ---

def clean_numeric_value(value):
    """Limpia y convierte valores a float, manejando formatos con puntos/comas."""
    if pd.isna(value) or value == '':
        return 0.0
    s_value = str(value).strip()
    # Eliminar puntos de miles y reemplazar coma decimal por punto
    # CORRECCIÓN APLICADA: Se eliminó regex=False de los métodos .replace() de string
    s_value = s_value.replace('.', '').replace(',', '.')
    try:
        return float(s_value)
    except ValueError:
        # Considerar una lógica más robusta si los formatos son muy variados o inconsistentes.
        # Por ahora, si la conversión simple falla, retorna 0.0.
        return 0.0

def classify_account(cuenta_str: str) -> str:
    """
    Clasifica una cuenta en 'Estado de Resultados' o 'Balance General' y su tipo específico.
    ¡¡¡DEBES ADAPTAR ESTAS REGLAS A TU PLAN DE CUENTAS EN DETALLE SI LOS PREFIJOS VARÍAN!!!
    """
    if not isinstance(cuenta_str, str): # Añadir verificación de tipo
        return 'No Clasificado'
    if cuenta_str.startswith('1'):
        return 'Balance General - Activo'
    elif cuenta_str.startswith('2'):
        return 'Balance General - Pasivo'
    elif cuenta_str.startswith('3'):
        return 'Balance General - Patrimonio'
    elif cuenta_str.startswith('4'):
        return 'Estado de Resultados - Ingresos'
    elif cuenta_str.startswith('5'):
        return 'Estado de Resultados - Ingresos'
    elif cuenta_str.startswith('6'):
        return 'Estado de Resultados - Costo de Ventas'
    elif cuenta_str.startswith('7'):
        return 'Estado de Resultados - Gastos Operacionales'
    elif cuenta_str.startswith('8'):
        return 'Estado de Resultados - Gastos no Operacionales'
    elif cuenta_str.startswith('9'):
        return 'Estado de Resultados - Impuestos'
    else:
        return 'No Clasificado'

def get_top_level_accounts_for_display(df_raw: pd.DataFrame, value_col_name: str, statement_type: str) -> pd.DataFrame:
    """
    Identifica las cuentas relevantes para mostrar en un reporte financiero.
    """
    cuenta_col_key = statement_type.replace(' ', '_').upper()
    
    # Verificar si la clave existe en COL_CONFIG
    if cuenta_col_key not in COL_CONFIG:
        st.error(f"Error de Configuración: La clave '{cuenta_col_key}' no se encuentra en COL_CONFIG.")
        return pd.DataFrame() # Retornar DataFrame vacío para evitar más errores

    config_specific = COL_CONFIG[cuenta_col_key]
    
    # Verificar si las columnas existen en la configuración específica
    if not all(k in config_specific for k in ['CUENTA', 'NOMBRE_CUENTA', 'NIVEL_LINEA']):
        st.error(f"Error de Configuración: Faltan 'CUENTA', 'NOMBRE_CUENTA' o 'NIVEL_LINEA' en COL_CONFIG para '{cuenta_col_key}'.")
        return pd.DataFrame()

    cuenta_col = config_specific['CUENTA']
    nombre_cuenta_col = config_specific['NOMBRE_CUENTA']
    nivel_linea_col = config_specific['NIVEL_LINEA']

    # Verificar si las columnas existen en el DataFrame
    if not all(col in df_raw.columns for col in [cuenta_col, nombre_cuenta_col, nivel_linea_col, value_col_name]):
        missing_cols = [col for col in [cuenta_col, nombre_cuenta_col, nivel_linea_col, value_col_name] if col not in df_raw.columns]
        st.error(f"Error Interno en get_top_level_accounts_for_display: Faltan columnas en el DataFrame: {missing_cols}")
        return pd.DataFrame()

    df_processed = df_raw.copy() # Trabajar con una copia para evitar SettingWithCopyWarning
    df_processed['Cuenta_Str'] = df_processed[cuenta_col].astype(str)
    df_sorted = df_processed.sort_values(by='Cuenta_Str').reset_index(drop=True)
    
    df_sorted = df_sorted.dropna(subset=['Cuenta_Str', nombre_cuenta_col])
    df_sorted = df_sorted[df_sorted['Cuenta_Str'] != ''].reset_index(drop=True)
    df_sorted = df_sorted[df_sorted[nombre_cuenta_col] != ''].reset_index(drop=True)
    
    df_significant_values = df_sorted[df_sorted[value_col_name].abs() > 0.001].copy() # Usar .copy()
    
    if df_significant_values.empty: # Manejar caso donde no hay valores significativos
        unique_values = []
    else:
        unique_values = df_significant_values[value_col_name].unique()
    
    selected_rows_for_display = []
    
    for val in unique_values:
        group_of_accounts = df_significant_values[df_significant_values[value_col_name] == val].copy() # Usar .copy()
        if not group_of_accounts.empty:
            shortest_account_in_group = group_of_accounts.loc[
                group_of_accounts['Cuenta_Str'].str.len().idxmin()
            ]
            selected_rows_for_display.append(shortest_account_in_group)
    
    if selected_rows_for_display:
        df_result = pd.DataFrame(selected_rows_for_display).drop_duplicates(subset=['Cuenta_Str']).reset_index(drop=True)
    else: # Manejar caso donde no se seleccionaron filas
        df_result = pd.DataFrame(columns=df_sorted.columns)


    if statement_type == 'Estado de Resultados':
        zero_value_significant_levels = [1, 2, 3, 4, 6, 8]
    elif statement_type == 'Balance General':
        zero_value_significant_levels = [1, 2, 3, 4]
    else:
        zero_value_significant_levels = []

    df_zero_values_significant = df_sorted[
        (df_sorted[value_col_name].abs() < 0.001) & 
        (df_sorted[nivel_linea_col].isin(zero_value_significant_levels))
    ].copy() # Usar .copy()
    
    if df_result.empty and df_zero_values_significant.empty:
        return pd.DataFrame(columns=df_sorted.columns) # Retornar estructura si ambos están vacíos
    elif df_result.empty:
        df_final_display = df_zero_values_significant.drop_duplicates(subset=['Cuenta_Str']).reset_index(drop=True)
    elif df_zero_values_significant.empty:
        df_final_display = df_result.drop_duplicates(subset=['Cuenta_Str']).reset_index(drop=True)
    else:
        df_final_display = pd.concat([df_result, df_zero_values_significant]).drop_duplicates(subset=['Cuenta_Str']).reset_index(drop=True)

    df_final_display = df_final_display.sort_values(by=[nivel_linea_col, 'Cuenta_Str'])
    
    return df_final_display


def generate_financial_statement(df_full_data: pd.DataFrame, statement_type: str, filter_cc: str = None, max_level: int = 999) -> pd.DataFrame:
    config_section_key = 'EDO_RESULTADO' if statement_type == 'Estado de Resultados' else 'BALANCE'
    
    if config_section_key not in COL_CONFIG:
        st.error(f"Error de Configuración: '{config_section_key}' no encontrado en COL_CONFIG.")
        return pd.DataFrame()
        
    config_section = COL_CONFIG[config_section_key]

    if not all(k in config_section for k in ['CUENTA', 'NOMBRE_CUENTA', 'NIVEL_LINEA']):
        st.error(f"Error de Configuración: Faltan claves esenciales en COL_CONFIG para '{config_section_key}'.")
        return pd.DataFrame()

    cuenta_col = config_section['CUENTA']
    nombre_cuenta_col = config_section['NOMBRE_CUENTA']
    nivel_linea_col = config_section['NIVEL_LINEA']

    final_df_columns = [cuenta_col, nombre_cuenta_col, 'Valor']
    
    # Verificar si las columnas base existen en df_full_data
    if not all(col in df_full_data.columns for col in [cuenta_col, nombre_cuenta_col, nivel_linea_col, 'Tipo_Estado']):
        missing_cols = [col for col in [cuenta_col, nombre_cuenta_col, nivel_linea_col, 'Tipo_Estado'] if col not in df_full_data.columns]
        st.error(f"Error Interno en generate_financial_statement: Faltan columnas en df_full_data: {missing_cols}")
        return pd.DataFrame(columns=final_df_columns)

    if statement_type == 'Estado de Resultados':
        df_statement = df_full_data[df_full_data['Tipo_Estado'].str.contains('Estado de Resultados', na=False)].copy()
        
        if df_statement.empty:
            st.warning(f"No hay datos de 'Estado de Resultados' para procesar (verifique la columna 'Tipo_Estado').")
            return pd.DataFrame(columns=final_df_columns)

        value_column_to_use = ''
        if filter_cc and filter_cc != 'Todos':
            if filter_cc in df_statement.columns:
                value_column_to_use = filter_cc
            else:
                st.error(f"Error: El Centro de Costo '{filter_cc}' seleccionado no existe como columna en los datos del ER.")
                df_statement['Valor_Final'] = 0.0 # Fallback
                value_column_to_use = 'Valor_Final'
        else: # 'Todos' o ningún filtro CC
            total_col_er = COL_CONFIG['EDO_RESULTADO']['CENTROS_COSTO_COLS'].get('Total')
            if total_col_er and total_col_er in df_statement.columns:
                value_column_to_use = total_col_er
            else: # Intentar sumar los CCs si 'Total' no está o no está configurado como tal
                cc_cols_for_sum = [
                    name for excel_name, name in COL_CONFIG['EDO_RESULTADO']['CENTROS_COSTO_COLS'].items()
                    if excel_name != 'Total' and name in df_statement.columns
                ]
                if cc_cols_for_sum:
                    df_statement['Temp_Sum_CC'] = df_statement[cc_cols_for_sum].sum(axis=1)
                    value_column_to_use = 'Temp_Sum_CC'
                elif total_col_er: # Si 'Total' estaba configurado pero no existe la columna
                     st.warning(f"Advertencia: La columna '{total_col_er}' (Total Consolidado ER) no se encuentra. Se mostrarán ceros para el consolidado.")
                     df_statement['Valor_Final'] = 0.0
                     value_column_to_use = 'Valor_Final'
                else: # Ni 'Total' ni otros CCs para sumar
                    st.warning("Advertencia: No hay Centros de Costo configurados ni columna 'Total' para el ER consolidado. Se mostrarán ceros.")
                    df_statement['Valor_Final'] = 0.0
                    value_column_to_use = 'Valor_Final'
        
        if value_column_to_use and value_column_to_use in df_statement.columns:
            df_statement['Valor_Final'] = df_statement[value_column_to_use]
        elif not value_column_to_use and 'Valor_Final' not in df_statement.columns : # Si value_column_to_use quedó vacío y no hay fallback
            st.error("Error crítico: No se pudo determinar la columna de valor para el Estado de Resultados.")
            df_statement['Valor_Final'] = 0.0 # Fallback final


        df_display = get_top_level_accounts_for_display(df_statement, 'Valor_Final', statement_type)
        if df_display.empty:
            st.info("No hay cuentas para mostrar en el Estado de Resultados después del filtrado jerárquico.")
            return pd.DataFrame(columns=final_df_columns)
            
        df_display = df_display[df_display[nivel_linea_col].astype(float) <= float(max_level)].copy() # Asegurar tipos para comparación

        er_order = [
            'Estado de Resultados - Ingresos',
            'Estado de Resultados - Costo de Ventas',
            'Estado de Resultados - Gastos Operacionales',
            'Estado de Resultados - Gastos no Operacionales',
            'Estado de Resultados - Impuestos'
        ]
        
        final_er_df = pd.DataFrame(columns=final_df_columns)
        total_er = 0
        
        if 'Tipo_Estado' not in df_display.columns:
             st.error("Error interno: 'Tipo_Estado' no encontrado en df_display para ER.")
             return final_er_df


        for tipo_completo in er_order:
            # Asegurar que 'Tipo_Estado' exista en df_display
            if 'Tipo_Estado' not in df_display.columns:
                st.error(f"Error: La columna 'Tipo_Estado' no está en df_display al procesar {tipo_completo}.")
                continue # Saltar este grupo si la columna falta

            group_df = df_display[df_display['Tipo_Estado'] == tipo_completo].copy()
            if not group_df.empty:
                group_df = group_df.sort_values(by=cuenta_col)
                group_df['Nombre_Cuenta_Display'] = group_df.apply(
                    lambda row: f"{'  ' * (int(pd.to_numeric(row[nivel_linea_col], errors='coerce') or 1) - 1)}{row[nombre_cuenta_col]}",
                    axis=1
                )
                if 'Valor_Final' not in group_df.columns:
                    st.error(f"Error interno: 'Valor_Final' no encontrado en group_df para {tipo_completo}. Asignando 0.")
                    group_df['Valor_Final'] = 0.0 

                final_er_df = pd.concat([final_er_df, group_df[[cuenta_col, 'Nombre_Cuenta_Display', 'Valor_Final']].rename(columns={'Nombre_Cuenta_Display': nombre_cuenta_col, 'Valor_Final': 'Valor'})], ignore_index=True)
                total_er += group_df['Valor_Final'].sum()
        
        final_er_df.loc[len(final_er_df)] = pd.Series({cuenta_col:'', nombre_cuenta_col:'TOTAL ESTADO DE RESULTADOS', 'Valor':total_er})
        final_er_df.loc[len(final_er_df)] = pd.Series({cuenta_col:'', nombre_cuenta_col:'', 'Valor':None}) # pd.NA o None para valor
        return final_er_df

    elif statement_type == 'Balance General':
        df_statement = df_full_data[df_full_data['Tipo_Estado'].str.contains('Balance General', na=False)].copy()
        
        if df_statement.empty:
            st.warning(f"No hay datos de 'Balance General' para procesar (verifique la columna 'Tipo_Estado').")
            return pd.DataFrame(columns=final_df_columns)

        saldo_final_col = COL_CONFIG['BALANCE'].get('SALDO_FINAL')
        if not saldo_final_col or saldo_final_col not in df_statement.columns:
            st.error(f"Error: La columna '{saldo_final_col}' (Saldo Final) no se encuentra en los datos del Balance.")
            df_statement['Valor_Final'] = 0.0
        else:
            df_statement['Valor_Final'] = df_statement[saldo_final_col]

        df_display = get_top_level_accounts_for_display(df_statement, 'Valor_Final', statement_type)
        if df_display.empty:
            st.info("No hay cuentas para mostrar en el Balance General después del filtrado jerárquico.")
            return pd.DataFrame(columns=final_df_columns)
            
        df_display = df_display[df_display[nivel_linea_col].astype(float) <= float(max_level)].copy() # Asegurar tipos

        bg_order = [
            'Balance General - Activo',
            'Balance General - Pasivo',
            'Balance General - Patrimonio'
        ]
        
        final_bg_df = pd.DataFrame(columns=final_df_columns)
        total_activo = 0
        total_pasivo = 0
        total_patrimonio = 0

        if 'Tipo_Estado' not in df_display.columns:
             st.error("Error interno: 'Tipo_Estado' no encontrado en df_display para BG.")
             return final_bg_df

        for tipo_completo in bg_order:
            if 'Tipo_Estado' not in df_display.columns:
                st.error(f"Error: La columna 'Tipo_Estado' no está en df_display al procesar {tipo_completo}.")
                continue

            group_df = df_display[df_display['Tipo_Estado'] == tipo_completo].copy()
            if not group_df.empty:
                group_df = group_df.sort_values(by=cuenta_col)
                group_df['Nombre_Cuenta_Display'] = group_df.apply(
                    lambda row: f"{'  ' * (int(pd.to_numeric(row[nivel_linea_col], errors='coerce') or 1) - 1)}{row[nombre_cuenta_col]}",
                    axis=1
                )
                if 'Valor_Final' not in group_df.columns:
                    st.error(f"Error interno: 'Valor_Final' no encontrado en group_df para {tipo_completo}. Asignando 0.")
                    group_df['Valor_Final'] = 0.0

                final_bg_df = pd.concat([final_bg_df, group_df[[cuenta_col, 'Nombre_Cuenta_Display', 'Valor_Final']].rename(columns={'Nombre_Cuenta_Display': nombre_cuenta_col, 'Valor_Final': 'Valor'})], ignore_index=True)
                
                if 'Activo' in tipo_completo:
                    total_activo += group_df['Valor_Final'].sum()
                elif 'Pasivo' in tipo_completo:
                    total_pasivo += group_df['Valor_Final'].sum()
                elif 'Patrimonio' in tipo_completo:
                    total_patrimonio += group_df['Valor_Final'].sum()

        final_bg_df.loc[len(final_bg_df)] = pd.Series({cuenta_col:'', nombre_cuenta_col:'TOTAL ACTIVOS', 'Valor':total_activo})
        final_bg_df.loc[len(final_bg_df)] = pd.Series({cuenta_col:'', nombre_cuenta_col:'', 'Valor':None})
        final_bg_df.loc[len(final_bg_df)] = pd.Series({cuenta_col:'', nombre_cuenta_col:'TOTAL PASIVOS', 'Valor':total_pasivo})
        final_bg_df.loc[len(final_bg_df)] = pd.Series({cuenta_col:'', nombre_cuenta_col:'TOTAL PATRIMONIO', 'Valor':total_patrimonio})
        
        total_pasivo_mas_patrimonio = total_pasivo + total_patrimonio
        final_bg_df.loc[len(final_bg_df)] = pd.Series({cuenta_col:'', nombre_cuenta_col:'TOTAL PASIVO + PATRIMONIO', 'Valor':total_pasivo_mas_patrimonio})
        balance_check = total_activo - total_pasivo_mas_patrimonio
        final_bg_df.loc[len(final_bg_df)] = pd.Series({cuenta_col:'', nombre_cuenta_col:'DIFERENCIA (Activo - (Pasivo + Patrimonio))', 'Valor':balance_check})
        
        return final_bg_df
    return pd.DataFrame(columns=final_df_columns)


def to_excel_buffer(er_df: pd.DataFrame, bg_df: pd.DataFrame) -> io.BytesIO:
    """Exporta los DataFrames a un buffer de Excel."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if not er_df.empty:
            er_df.to_excel(writer, sheet_name='Estado de Resultados', index=False)
        if not bg_df.empty:
            bg_df.to_excel(writer, sheet_name='Balance General', index=False)
    output.seek(0)
    return output

# --- Aplicación Streamlit ---

st.set_page_config(layout="wide", page_title="Análisis Financiero Avanzado")

st.title("💰 Análisis Financiero y Tablero Gerencial")
st.write("Sube tu archivo Excel para generar Estados Financieros detallados y por Centro de Costo.")

# Inicializar DataFrames vacíos en session state para evitar errores si no se carga archivo
if 'df_er' not in st.session_state:
    st.session_state.df_er = pd.DataFrame()
if 'df_bg' not in st.session_state:
    st.session_state.df_bg = pd.DataFrame()
if 'final_er_display' not in st.session_state:
    st.session_state.final_er_display = pd.DataFrame()
if 'final_bg_display' not in st.session_state:
    st.session_state.final_bg_display = pd.DataFrame()


uploaded_file = st.file_uploader("Sube tu archivo Excel de cuentas (con hojas 'EDO RESULTADO' y 'BALANCE')", type=["xlsx"])

if uploaded_file is not None:
    try:
        xls = pd.ExcelFile(uploaded_file)
        if 'EDO RESULTADO' not in xls.sheet_names:
            st.error("Error: La hoja 'EDO RESULTADO' no se encontró en el archivo Excel.")
            st.stop()
        if 'BALANCE' not in xls.sheet_names:
            st.error("Error: La hoja 'BALANCE' no se encontró en el archivo Excel.")
            st.stop()

        df_er_raw = pd.read_excel(xls, sheet_name='EDO RESULTADO')
        df_bg_raw = pd.read_excel(xls, sheet_name='BALANCE')

        st.success("Archivo cargado y hojas 'EDO RESULTADO' y 'BALANCE' encontradas correctamente.")
        
        # --- Procesamiento EDO RESULTADO ---
        er_rename_map = {k: v for k, v in COL_CONFIG['EDO_RESULTADO']['CENTROS_COSTO_COLS'].items() if k in df_er_raw.columns}
        st.session_state.df_er = df_er_raw.rename(columns=er_rename_map).copy()

        for logical_cc_name in COL_CONFIG['EDO_RESULTADO']['CENTROS_COSTO_COLS'].values():
            if logical_cc_name in st.session_state.df_er.columns:
                st.session_state.df_er[logical_cc_name] = st.session_state.df_er[logical_cc_name].apply(clean_numeric_value)
        
        er_cuenta_col = COL_CONFIG['EDO_RESULTADO']['CUENTA']
        er_nombre_cuenta_col = COL_CONFIG['EDO_RESULTADO']['NOMBRE_CUENTA']
        er_nivel_linea_col = COL_CONFIG['EDO_RESULTADO']['NIVEL_LINEA']

        for col, name in [(er_cuenta_col, 'CUENTA'), (er_nombre_cuenta_col, 'NOMBRE_CUENTA'), (er_nivel_linea_col, 'NIVEL_LINEA (Grupo)')]:
            if col not in st.session_state.df_er.columns:
                st.error(f"Error de Configuración (ER): La columna '{col}' para '{name}' no existe. Verifica COL_CONFIG y tu Excel.")
                st.stop()
        
        st.session_state.df_er[er_cuenta_col] = st.session_state.df_er[er_cuenta_col].astype(str).str.strip()
        st.session_state.df_er[er_nombre_cuenta_col] = st.session_state.df_er[er_nombre_cuenta_col].astype(str).str.strip()
        st.session_state.df_er[er_nivel_linea_col] = pd.to_numeric(st.session_state.df_er[er_nivel_linea_col], errors='coerce').fillna(0).astype(int)
        st.session_state.df_er['Tipo_Estado'] = st.session_state.df_er[er_cuenta_col].apply(classify_account)

        # --- Procesamiento BALANCE ---
        st.session_state.df_bg = df_bg_raw.copy()
        balance_cols_to_clean = ['SALDO_INICIAL', 'DEBE', 'HABER', 'SALDO_FINAL']
        for col_key in balance_cols_to_clean:
            col_name = COL_CONFIG['BALANCE'][col_key]
            if col_name in st.session_state.df_bg.columns:
                st.session_state.df_bg[col_name] = st.session_state.df_bg[col_name].apply(clean_numeric_value)
            else:
                 st.error(f"Error de Configuración (Balance): La columna '{col_name}' para '{col_key}' no existe.")
                 st.stop()
        
        bg_cuenta_col = COL_CONFIG['BALANCE']['CUENTA']
        bg_nombre_cuenta_col = COL_CONFIG['BALANCE']['NOMBRE_CUENTA']
        bg_nivel_linea_col = COL_CONFIG['BALANCE']['NIVEL_LINEA']

        for col, name in [(bg_cuenta_col, 'CUENTA'), (bg_nombre_cuenta_col, 'NOMBRE_CUENTA'), (bg_nivel_linea_col, 'NIVEL_LINEA (Grupo)')]:
            if col not in st.session_state.df_bg.columns:
                st.error(f"Error de Configuración (Balance): La columna '{col}' para '{name}' no existe. Verifica COL_CONFIG y tu Excel.")
                st.stop()

        st.session_state.df_bg[bg_cuenta_col] = st.session_state.df_bg[bg_cuenta_col].astype(str).str.strip()
        st.session_state.df_bg[bg_nombre_cuenta_col] = st.session_state.df_bg[bg_nombre_cuenta_col].astype(str).str.strip()
        st.session_state.df_bg[bg_nivel_linea_col] = pd.to_numeric(st.session_state.df_bg[bg_nivel_linea_col], errors='coerce').fillna(0).astype(int)
        st.session_state.df_bg['Tipo_Estado'] = st.session_state.df_bg[bg_cuenta_col].apply(classify_account)
        

        st.subheader("Configuración de Columnas Utilizada:")
        st.json(COL_CONFIG)
        st.warning("Verifica que esta configuración de columnas refleje la estructura de tu Excel.")

        st.subheader("Datos Procesados (Primeras 5 filas del EDO RESULTADO - Renombradas):")
        st.dataframe(st.session_state.df_er.head())
        
        st.subheader("Datos Procesados (Primeras 5 filas del BALANCE):")
        st.dataframe(st.session_state.df_bg.head())

    except FileNotFoundError:
        st.error("Error: El archivo subido no se pudo encontrar o leer.")
    except KeyError as e:
        st.error(f"Error de Configuración o Dato Faltante: No se encontró la columna esperada: {e}. Revisa la configuración 'COL_CONFIG' y tu archivo Excel.")
    except ValueError as e:
        st.error(f"Error de Valor: Problema con la conversión de datos. {e}")
    except Exception as e:
        st.error(f"Ocurrió un error inesperado al procesar el archivo: {e}")
        st.exception(e) # Muestra el traceback completo para depuración

# --- Interfaz de Usuario para el Tablero (fuera del bloque if uploaded_file) ---
# Esto permite que el sidebar se muestre incluso sin archivo, pero los widgets estarán deshabilitados o con valores por defecto.

st.sidebar.header("Opciones de Reporte")
report_type = st.sidebar.radio(
    "Selecciona el tipo de reporte:", 
    ["Estado de Resultados", "Balance General"], 
    disabled=st.session_state.df_er.empty # Deshabilitar si no hay datos
)

if not st.session_state.df_er.empty and report_type == "Estado de Resultados":
    st.header("📈 Estado de Resultados por Centro de Costo")
    
    er_nivel_linea_col = COL_CONFIG['EDO_RESULTADO']['NIVEL_LINEA']
    actual_cc_logical_names = [
        name for excel_name, name in COL_CONFIG['EDO_RESULTADO']['CENTROS_COSTO_COLS'].items()
        if name in st.session_state.df_er.columns and excel_name != 'Total'
    ]
    selected_cc = st.sidebar.selectbox("Selecciona un Centro de Costo:", ['Todos'] + actual_cc_logical_names)
    
    level_options = sorted(st.session_state.df_er[er_nivel_linea_col].astype(int).unique().tolist())
    if not level_options: level_options = [1] # Default
    
    max_level_er_val = max(level_options) if level_options else 1
    min_level_er_val = min(level_options) if level_options else 1
    default_level_er = min_level_er_val

    max_level_er = st.sidebar.slider("Nivel de Detalle (Estado de Resultados):", 
                                  min_value=min_level_er_val, 
                                  max_value=max_level_er_val, 
                                  value=default_level_er)

    st.session_state.final_er_display = generate_financial_statement(st.session_state.df_er, 'Estado de Resultados', selected_cc, max_level_er)
    st.dataframe(st.session_state.final_er_display, use_container_width=True, hide_index=True)
    
    # Gráfico ER
    if not st.session_state.final_er_display.empty and 'Valor' in st.session_state.final_er_display.columns:
        er_nombre_cuenta_col_graph = COL_CONFIG['EDO_RESULTADO']['NOMBRE_CUENTA']
        if er_nombre_cuenta_col_graph in st.session_state.final_er_display.columns:
            st.subheader("Visualización del Estado de Resultados (Sumario)")
            
            # Para el gráfico sumario, usamos el df_er original que tiene Tipo_Estado
            temp_df_for_chart_er = st.session_state.df_er.copy()
            value_col_for_chart_er = ''

            if selected_cc and selected_cc != 'Todos' and selected_cc in temp_df_for_chart_er.columns:
                value_col_for_chart_er = selected_cc
            else: # 'Todos'
                total_col_config_er = COL_CONFIG['EDO_RESULTADO']['CENTROS_COSTO_COLS'].get('Total')
                if total_col_config_er and total_col_config_er in temp_df_for_chart_er.columns:
                    value_col_for_chart_er = total_col_config_er
                else:
                    cc_cols_sum_chart_er = [
                        name for excel_name, name in COL_CONFIG['EDO_RESULTADO']['CENTROS_COSTO_COLS'].items()
                        if excel_name != 'Total' and name in temp_df_for_chart_er.columns
                    ]
                    if cc_cols_sum_chart_er:
                        temp_df_for_chart_er['Temp_Sum_CC_Chart_ER'] = temp_df_for_chart_er[cc_cols_sum_chart_er].sum(axis=1)
                        value_col_for_chart_er = 'Temp_Sum_CC_Chart_ER'
            
            if value_col_for_chart_er and 'Tipo_Estado' in temp_df_for_chart_er.columns:
                summary_data = {
                    'Ingresos': temp_df_for_chart_er[temp_df_for_chart_er['Tipo_Estado'] == 'Estado de Resultados - Ingresos'][value_col_for_chart_er].sum(),
                    'Costo de Ventas': temp_df_for_chart_er[temp_df_for_chart_er['Tipo_Estado'] == 'Estado de Resultados - Costo de Ventas'][value_col_for_chart_er].sum(),
                    'Gastos Operacionales': temp_df_for_chart_er[temp_df_for_chart_er['Tipo_Estado'] == 'Estado de Resultados - Gastos Operacionales'][value_col_for_chart_er].sum(),
                    'Gastos no Operacionales': temp_df_for_chart_er[temp_df_for_chart_er['Tipo_Estado'] == 'Estado de Resultados - Gastos no Operacionales'][value_col_for_chart_er].sum(),
                    'Impuestos': temp_df_for_chart_er[temp_df_for_chart_er['Tipo_Estado'] == 'Estado de Resultados - Impuestos'][value_col_for_chart_er].sum(),
                }
                chart_summary_er = pd.DataFrame(list(summary_data.items()), columns=['Categoría', 'Valor'])
                st.bar_chart(chart_summary_er.set_index('Categoría'))
            else:
                st.write("No se pudieron generar datos para el gráfico sumario del ER.")


elif not st.session_state.df_bg.empty and report_type == "Balance General":
    st.header("💰 Balance General")
    bg_nivel_linea_col = COL_CONFIG['BALANCE']['NIVEL_LINEA']
    
    level_options_bg = sorted(st.session_state.df_bg[bg_nivel_linea_col].astype(int).unique().tolist())
    if not level_options_bg: level_options_bg = [1]

    max_level_bg_val = max(level_options_bg) if level_options_bg else 1
    min_level_bg_val = min(level_options_bg) if level_options_bg else 1
    default_level_bg = min_level_bg_val

    max_level_bg = st.sidebar.slider("Nivel de Detalle (Balance General):", 
                                     min_value=min_level_bg_val, 
                                     max_value=max_level_bg_val, 
                                     value=default_level_bg)
    
    st.session_state.final_bg_display = generate_financial_statement(st.session_state.df_bg, 'Balance General', max_level=max_level_bg)
    st.dataframe(st.session_state.final_bg_display, use_container_width=True, hide_index=True)
    
    if not st.session_state.final_bg_display.empty and 'Valor' in st.session_state.final_bg_display.columns:
        bg_nombre_cuenta_col_kpi = COL_CONFIG['BALANCE']['NOMBRE_CUENTA']
        if bg_nombre_cuenta_col_kpi in st.session_state.final_bg_display.columns:
            st.subheader("Indicadores Clave del Balance")
            
            total_act_series = st.session_state.final_bg_display[st.session_state.final_bg_display[bg_nombre_cuenta_col_kpi] == 'TOTAL ACTIVOS']['Valor']
            total_pas_series = st.session_state.final_bg_display[st.session_state.final_bg_display[bg_nombre_cuenta_col_kpi] == 'TOTAL PASIVOS']['Valor']
            total_pat_series = st.session_state.final_bg_display[st.session_state.final_bg_display[bg_nombre_cuenta_col_kpi] == 'TOTAL PATRIMONIO']['Valor']

            total_act = total_act_series.iloc[0] if not total_act_series.empty else 0
            total_pas = total_pas_series.iloc[0] if not total_pas_series.empty else 0
            total_pat = total_pat_series.iloc[0] if not total_pat_series.empty else 0

            st.metric(label="Total Activos", value=f"{total_act:,.2f}")
            st.metric(label="Total Pasivos", value=f"{total_pas:,.2f}")
            st.metric(label="Total Patrimonio", value=f"{total_pat:,.2f}")

            if total_act != 0:
                st.metric(label="Razón de Endeudamiento (Pasivo/Activo)", value=f"{total_pas / total_act:.2%}" if total_act !=0 else "N/A")
            if total_pat != 0:
                 st.metric(label="Razón Pasivo/Patrimonio", value=f"{total_pas / total_pat:.2%}" if total_pat !=0 else "N/A")
            
            balance_chart_data = pd.DataFrame({
                'Categoría': ['Activos', 'Pasivos', 'Patrimonio'],
                'Valor': [total_act, total_pas, total_pat]
            })
            st.bar_chart(balance_chart_data.set_index('Categoría'))

# Botón para descargar el Excel completo (fuera del if uploaded_file, pero deshabilitado si no hay datos)
st.sidebar.markdown("---")
st.sidebar.subheader("Exportar a Excel")

download_disabled = st.session_state.df_er.empty or st.session_state.df_bg.empty

if not download_disabled:
    # Preparar ER para exportación
    er_export_df_final = pd.DataFrame()
    if not st.session_state.df_er.empty:
        er_export_cols = [
            COL_CONFIG['EDO_RESULTADO']['NIVEL_LINEA'],
            COL_CONFIG['EDO_RESULTADO']['CUENTA'],
            COL_CONFIG['EDO_RESULTADO']['NOMBRE_CUENTA']
        ] + [name for name in COL_CONFIG['EDO_RESULTADO']['CENTROS_COSTO_COLS'].values() if name in st.session_state.df_er.columns]
        
        er_export_cols_exist = [col for col in er_export_cols if col in st.session_state.df_er.columns]
        er_export_df_final = st.session_state.df_er[er_export_cols_exist].copy()

        total_row_er_export = {col: '' for col in er_export_cols_exist} 
        if COL_CONFIG['EDO_RESULTADO']['NOMBRE_CUENTA'] in total_row_er_export:
            total_row_er_export[COL_CONFIG['EDO_RESULTADO']['NOMBRE_CUENTA']] = 'TOTALES'
        
        for logical_cc_name in COL_CONFIG['EDO_RESULTADO']['CENTROS_COSTO_COLS'].values():
            if logical_cc_name in er_export_df_final.columns: # Sumar solo columnas que existen y son numéricas
                 numeric_sum = pd.to_numeric(er_export_df_final[logical_cc_name], errors='coerce').sum()
                 total_row_er_export[logical_cc_name] = numeric_sum
        
        total_row_df = pd.DataFrame([total_row_er_export])
        er_export_df_final = pd.concat([er_export_df_final, total_row_df], ignore_index=True)

    # Preparar BG para exportación
    bg_export_df_final = pd.DataFrame()
    if not st.session_state.df_bg.empty:
        bg_nivel_linea_col_export = COL_CONFIG['BALANCE']['NIVEL_LINEA']
        max_level_for_export_bg_series = pd.to_numeric(st.session_state.df_bg[bg_nivel_linea_col_export], errors='coerce').dropna()
        max_level_for_export_bg = int(max_level_for_export_bg_series.max()) if not max_level_for_export_bg_series.empty else 999
        bg_export_df_final = generate_financial_statement(st.session_state.df_bg, 'Balance General', max_level=max_level_for_export_bg)

    excel_buffer = to_excel_buffer(er_export_df_final, bg_export_df_final)
    st.sidebar.download_button(
        label="Descargar Reporte Completo (Excel)",
        data=excel_buffer,
        file_name="reporte_financiero_completo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        disabled=download_disabled
    )
else:
    st.sidebar.download_button(
        label="Descargar Reporte Completo (Excel)",
        data=b"", # Datos vacíos
        file_name="reporte_financiero_completo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        disabled=True
    )