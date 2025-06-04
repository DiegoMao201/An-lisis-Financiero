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
    s_value = s_value.replace('.', '', regex=False).replace(',', '.', regex=False)
    try:
        return float(s_value)
    except ValueError:
        return 0.0 # O manejar como un error específico

def classify_account(cuenta_str: str) -> str:
    """
    Clasifica una cuenta en 'Estado de Resultados' o 'Balance General' y su tipo específico.
    ¡¡¡DEBES ADAPTAR ESTAS REGLAS A TU PLAN DE CUENTAS EN DETALLE SI LOS PREFIJOS VARÍAN!!!
    """
    if cuenta_str.startswith('1'):
        return 'Balance General - Activo'
    elif cuenta_str.startswith('2'):
        return 'Balance General - Pasivo'
    elif cuenta_str.startswith('3'):
        return 'Balance General - Patrimonio'
    elif cuenta_str.startswith('4'): # Ingresos Operacionales
        return 'Estado de Resultados - Ingresos'
    elif cuenta_str.startswith('5'): # Otros Ingresos / Ingresos No Operacionales
        return 'Estado de Resultados - Ingresos' # O podrías crear 'Otros Ingresos'
    elif cuenta_str.startswith('6'): # Costo de Ventas
        return 'Estado de Resultados - Costo de Ventas'
    elif cuenta_str.startswith('7'): # Gastos Operacionales (Administración y Ventas)
        return 'Estado de Resultados - Gastos Operacionales'
    # Agrega más clasificaciones si es necesario (ej. '8' para Gastos no Operacionales, '9' para Impuestos)
    elif cuenta_str.startswith('8'):
        return 'Estado de Resultados - Gastos no Operacionales'
    elif cuenta_str.startswith('9'):
        return 'Estado de Resultados - Impuestos'
    else:
        return 'No Clasificado'

def get_top_level_accounts_for_display(df_raw: pd.DataFrame, value_col_name: str, statement_type: str) -> pd.DataFrame:
    """
    Identifica las cuentas relevantes para mostrar en un reporte financiero.
    La lógica es: si un valor se repite en varias cuentas jerárquicas (ej. 6, 61, 6135 todos tienen el mismo -79.7M),
    solo se mantiene la cuenta más corta (la de nivel más alto) para ese valor.
    Las cuentas de detalle (hoja) con valores únicos o que son sumas finales también se mantienen.
    """
    # Usamos la columna 'Cuenta' para la jerarquía, aseguramos que sea string
    cuenta_col = COL_CONFIG[statement_type.replace(' ', '_').upper()]['CUENTA']
    nombre_cuenta_col = COL_CONFIG[statement_type.replace(' ', '_').upper()]['NOMBRE_CUENTA']
    nivel_linea_col = COL_CONFIG[statement_type.replace(' ', '_').upper()]['NIVEL_LINEA']

    df_raw['Cuenta_Str'] = df_raw[cuenta_col].astype(str)

    # Ordenar por cuenta para ayudar en el procesamiento de jerarquías
    df_sorted = df_raw.sort_values(by='Cuenta_Str').reset_index(drop=True)

    # Eliminar filas donde la Cuenta o Título es nulo o vacío, ya que no son cuentas válidas
    df_sorted = df_sorted.dropna(subset=['Cuenta_Str', nombre_cuenta_col])
    df_sorted = df_sorted[df_sorted['Cuenta_Str'] != ''].reset_index(drop=True)
    df_sorted = df_sorted[df_sorted[nombre_cuenta_col] != ''].reset_index(drop=True)
    
    # Filter out accounts with 0 value or no significant value, as they often clutter without adding info
    # Using .copy() to avoid SettingWithCopyWarning
    df_significant_values = df_sorted[df_sorted[value_col_name].abs() > 0.001].copy()

    unique_values = df_significant_values[value_col_name].unique()
    
    selected_rows_for_display = []
    
    for val in unique_values:
        # Get all rows that have this specific value
        group_of_accounts = df_significant_values[df_significant_values[value_col_name] == val].copy()
        
        if not group_of_accounts.empty:
            # From this group, pick the account with the shortest 'Cuenta_Str'
            shortest_account_in_group = group_of_accounts.loc[
                group_of_accounts['Cuenta_Str'].str.len().idxmin()
            ]
            selected_rows_for_display.append(shortest_account_in_group)
    
    df_result = pd.DataFrame(selected_rows_for_display).drop_duplicates(subset=['Cuenta_Str']).reset_index(drop=True)
    
    # Add back accounts with value 0 if they are 'Grupo' level (or other significant levels)
    # This handles cases like 'VENTA DE PINTURAS Y LACAS' which has 0 value but is a detail.
    # We should consider significant levels based on the statement type.
    # For EDO RESULTADO, perhaps levels 1, 2, 3, 4, 6, 8 are relevant.
    # For BALANCE, perhaps 1, 2, 3, 4 are relevant. Adjust as needed.
    
    if statement_type == 'Estado de Resultados':
        zero_value_significant_levels = [1, 2, 3, 4, 6, 8] # Ajusta estos niveles para ER
    elif statement_type == 'Balance General':
        zero_value_significant_levels = [1, 2, 3, 4] # Ajusta estos niveles para BG
    else:
        zero_value_significant_levels = [] # Default for other types

    df_zero_values_significant = df_sorted[
        (df_sorted[value_col_name].abs() < 0.001) & 
        (df_sorted[nivel_linea_col].isin(zero_value_significant_levels))
    ].copy()
    
    # Combine results, remove duplicates based on 'Cuenta_Str'
    df_final_display = pd.concat([df_result, df_zero_values_significant]).drop_duplicates(subset=['Cuenta_Str']).reset_index(drop=True)
    
    # Reorder by hierarchy and line level for better presentation
    df_final_display = df_final_display.sort_values(by=[nivel_linea_col, 'Cuenta_Str'])
    
    return df_final_display

def generate_financial_statement(df_full_data: pd.DataFrame, statement_type: str, filter_cc: str = None, max_level: int = 999) -> pd.DataFrame:
    """
    Genera el Estado de Resultados o Balance General consolidado/por CC, con filtrado de nivel.
    """
    # Determinar las columnas de configuración basándose en el tipo de estado
    config_section = 'EDO_RESULTADO' if statement_type == 'Estado de Resultados' else 'BALANCE'
    cuenta_col = COL_CONFIG[config_section]['CUENTA']
    nombre_cuenta_col = COL_CONFIG[config_section]['NOMBRE_CUENTA']
    nivel_linea_col = COL_CONFIG[config_section]['NIVEL_LINEA']

    final_df_columns = [cuenta_col, nombre_cuenta_col, 'Valor']
    
    if statement_type == 'Estado de Resultados':
        # Filtrar solo las cuentas de ER
        df_statement = df_full_data[df_full_data['Tipo_Estado'].str.contains('Estado de Resultados')].copy()
        
        # Determinar la columna de valor a usar
        if filter_cc and filter_cc != 'Todos':
            # Si se selecciona un CC específico, usamos esa columna
            value_column_to_use = filter_cc
        else:
            # Si es 'Todos', sumamos todas las columnas de CC para obtener un total consolidado
            # Excluir la columna 'Total_Consolidado_ER' de la suma si ya existe en los datos originales
            # y se usará si no hay otros CC para sumar.
            cc_cols_for_sum = [
                name for excel_name, name in COL_CONFIG['EDO_RESULTADO']['CENTROS_COSTO_COLS'].items()
                if excel_name != 'Total' and name in df_statement.columns
            ]
            
            # Si no hay CCs para sumar, o si la columna 'Total_Consolidado_ER' ya existe y se prefiere para el "Todos"
            if not cc_cols_for_sum or COL_CONFIG['EDO_RESULTADO']['CENTROS_COSTO_COLS']['Total'] in df_statement.columns:
                value_column_to_use = COL_CONFIG['EDO_RESULTADO']['CENTROS_COSTO_COLS']['Total']
            else:
                df_statement['Temp_Sum_CC'] = df_statement[cc_cols_for_sum].sum(axis=1)
                value_column_to_use = 'Temp_Sum_CC'
        
        df_statement['Valor_Final'] = df_statement[value_column_to_use]
        
        # Aplicar la lógica de mostrar solo las cuentas de nivel superior o detalle final
        df_display = get_top_level_accounts_for_display(df_statement, 'Valor_Final', statement_type)
        
        # Filtrar por el nivel de detalle máximo seleccionado por el usuario
        df_display = df_display[df_display[nivel_linea_col] <= max_level].copy()

        # Ordenar las categorías del Estado de Resultados para presentación gerencial
        # Asegúrate de que las categorías aquí coincidan con lo que devuelve `classify_account`
        er_order = [
            'Estado de Resultados - Ingresos',
            'Estado de Resultados - Costo de Ventas',
            'Estado de Resultados - Gastos Operacionales',
            'Estado de Resultados - Gastos no Operacionales',
            'Estado de Resultados - Impuestos'
        ]
        
        final_er_df = pd.DataFrame(columns=final_df_columns)
        total_er = 0
        
        for tipo_completo in er_order:
            # Filtrar por el tipo completo de estado (ej. 'Estado de Resultados - Ingresos')
            group_df = df_display[df_display['Tipo_Estado'] == tipo_completo].copy()
            if not group_df.empty:
                group_df = group_df.sort_values(by=cuenta_col)
                group_df['Nombre_Cuenta_Display'] = group_df.apply(
                    lambda row: f"{'  ' * (int(row[nivel_linea_col]) - 1)}{row[nombre_cuenta_col]}",
                    axis=1
                )
                final_er_df = pd.concat([final_er_df, group_df[[cuenta_col, 'Nombre_Cuenta_Display', 'Valor_Final']].rename(columns={'Nombre_Cuenta_Display': nombre_cuenta_col, 'Valor_Final': 'Valor'})], ignore_index=True)
                
                # Sumar para el total general
                total_er += group_df['Valor_Final'].sum()
        
        # Añadir el total final
        final_er_df.loc[len(final_er_df)] = ['', 'TOTAL ESTADO DE RESULTADOS', total_er]
        final_er_df.loc[len(final_er_df)] = ['', '', ''] # Línea en blanco para separación

        return final_er_df

    elif statement_type == 'Balance General':
        df_statement = df_full_data[df_full_data['Tipo_Estado'].str.contains('Balance General')].copy()
        df_statement['Valor_Final'] = df_statement[COL_CONFIG['BALANCE']['SALDO_FINAL']] # Tomamos el Saldo Final para el balance

        # Aplicar la lógica de mostrar solo las cuentas de nivel superior o detalle final
        df_display = get_top_level_accounts_for_display(df_statement, 'Valor_Final', statement_type)
        
        # Filtrar por el nivel de detalle máximo seleccionado por el usuario
        df_display = df_display[df_display[nivel_linea_col] <= max_level].copy()

        # Ordenar las categorías del Balance General
        bg_order = [
            'Balance General - Activo',
            'Balance General - Pasivo',
            'Balance General - Patrimonio'
        ]
        
        final_bg_df = pd.DataFrame(columns=final_df_columns)
        
        total_activo = 0
        total_pasivo = 0
        total_patrimonio = 0

        for tipo_completo in bg_order:
            group_df = df_display[df_display['Tipo_Estado'] == tipo_completo].copy()
            if not group_df.empty:
                group_df = group_df.sort_values(by=cuenta_col)
                group_df['Nombre_Cuenta_Display'] = group_df.apply(
                    lambda row: f"{'  ' * (int(row[nivel_linea_col]) - 1)}{row[nombre_cuenta_col]}",
                    axis=1
                )
                final_bg_df = pd.concat([final_bg_df, group_df[[cuenta_col, 'Nombre_Cuenta_Display', 'Valor_Final']].rename(columns={'Nombre_Cuenta_Display': nombre_cuenta_col, 'Valor_Final': 'Valor'})], ignore_index=True)
                
                # Sumar para los totales principales
                if 'Activo' in tipo_completo:
                    total_activo += group_df['Valor_Final'].sum()
                elif 'Pasivo' in tipo_completo:
                    total_pasivo += group_df['Valor_Final'].sum()
                elif 'Patrimonio' in tipo_completo:
                    total_patrimonio += group_df['Valor_Final'].sum()

        final_bg_df.loc[len(final_bg_df)] = ['', 'TOTAL ACTIVOS', total_activo]
        final_bg_df.loc[len(final_bg_df)] = ['', '', ''] # Línea en blanco
        final_bg_df.loc[len(final_bg_df)] = ['', 'TOTAL PASIVOS', total_pasivo]
        final_bg_df.loc[len(final_bg_df)] = ['', 'TOTAL PATRIMONIO', total_patrimonio]
        
        total_pasivo_mas_patrimonio = total_pasivo + total_patrimonio
        final_bg_df.loc[len(final_bg_df)] = ['', 'TOTAL PASIVO + PATRIMONIO', total_pasivo_mas_patrimonio]
        
        balance_check = total_activo - total_pasivo_mas_patrimonio
        final_bg_df.loc[len(final_bg_df)] = ['', 'DIFERENCIA (Activo - (Pasivo + Patrimonio))', balance_check]
        
        return final_bg_df
    return pd.DataFrame()


def to_excel_buffer(er_df: pd.DataFrame, bg_df: pd.DataFrame) -> io.BytesIO:
    """Exporta los DataFrames a un buffer de Excel."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        er_df.to_excel(writer, sheet_name='Estado de Resultados', index=False)
        bg_df.to_excel(writer, sheet_name='Balance General', index=False)
    output.seek(0)
    return output

# --- Aplicación Streamlit ---

st.set_page_config(layout="wide", page_title="Análisis Financiero Avanzado")

st.title("💰 Análisis Financiero y Tablero Gerencial")
st.write("Sube tu archivo Excel para generar Estados Financieros detallados y por Centro de Costo.")

uploaded_file = st.file_uploader("Sube tu archivo Excel de cuentas (con hojas 'EDO RESULTADO' y 'BALANCE')", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Cargar ambas hojas
        df_er_raw = pd.read_excel(uploaded_file, sheet_name='EDO RESULTADO')
        df_bg_raw = pd.read_excel(uploaded_file, sheet_name='BALANCE')

        st.success("Archivo cargado y hojas 'EDO RESULTADO' y 'BALANCE' encontradas correctamente.")
        
        # --- Configuración Dinámica de Nombres de Columnas ---
        # No es "dinámica" en el sentido de que el usuario la cambie,
        # sino que refleja la configuración que se usa internamente en el código.
        # Aquí no necesitas redefinirla si ya está definida globalmente arriba,
        # pero es bueno tener la sección para recordar su importancia.
        
        # Renombrar las columnas de Centro de Costo en el DataFrame del ER
        # para usar los nombres 'lógicos' (Armenia, San antonio, etc.)
        # Asegurarse de que el mapeo se aplica correctamente y que solo las columnas existentes se renombran
        er_rename_map = {k: v for k, v in COL_CONFIG['EDO_RESULTADO']['CENTROS_COSTO_COLS'].items() if k in df_er_raw.columns}
        df_er = df_er_raw.rename(columns=er_rename_map).copy()

        # Aplicar limpieza y conversión de tipos para EDO RESULTADO
        for logical_cc_name in COL_CONFIG['EDO_RESULTADO']['CENTROS_COSTO_COLS'].values():
            if logical_cc_name in df_er.columns: # Asegurarse de que la columna exista después del renombrado
                df_er[logical_cc_name] = df_er[logical_cc_name].apply(clean_numeric_value)
        
        df_er[COL_CONFIG['EDO_RESULTADO']['CUENTA']] = df_er[COL_CONFIG['EDO_RESULTADO']['CUENTA']].astype(str).str.strip()
        df_er[COL_CONFIG['EDO_RESULTADO']['NOMBRE_CUENTA']] = df_er[COL_CONFIG['EDO_RESULTADO']['NOMBRE_CUENTA']].astype(str).str.strip()

        # Aplicar limpieza y conversión de tipos para BALANCE
        df_bg = df_bg_raw.copy()
        df_bg[COL_CONFIG['BALANCE']['SALDO_INICIAL']] = df_bg[COL_CONFIG['BALANCE']['SALDO_INICIAL']].apply(clean_numeric_value)
        df_bg[COL_CONFIG['BALANCE']['DEBE']] = df_bg[COL_CONFIG['BALANCE']['DEBE']].apply(clean_numeric_value)
        df_bg[COL_CONFIG['BALANCE']['HABER']] = df_bg[COL_CONFIG['BALANCE']['HABER']].apply(clean_numeric_value)
        df_bg[COL_CONFIG['BALANCE']['SALDO_FINAL']] = df_bg[COL_CONFIG['BALANCE']['SALDO_FINAL']].apply(clean_numeric_value)
        df_bg[COL_CONFIG['BALANCE']['CUENTA']] = df_bg[COL_CONFIG['BALANCE']['CUENTA']].astype(str).str.strip()
        df_bg[COL_CONFIG['BALANCE']['NOMBRE_CUENTA']] = df_bg[COL_CONFIG['BALANCE']['NOMBRE_CUENTA']].astype(str).str.strip()
        
        # Clasificar cuentas en ambos DataFrames
        df_er['Tipo_Estado'] = df_er[COL_CONFIG['EDO_RESULTADO']['CUENTA']].apply(classify_account)
        df_bg['Tipo_Estado'] = df_bg[COL_CONFIG['BALANCE']['CUENTA']].apply(classify_account)

        st.subheader("Configuración de Columnas Utilizada:")
        st.json(COL_CONFIG)
        st.warning("Verifica que esta configuración de columnas refleje la estructura de tu Excel.")

        st.subheader("Datos Procesados (Primeras 5 filas del EDO RESULTADO - Renombradas):")
        st.dataframe(df_er.head())
        
        st.subheader("Datos Procesados (Primeras 5 filas del BALANCE):")
        st.dataframe(df_bg.head())

        # --- Interfaz de Usuario para el Tablero ---

        st.sidebar.header("Opciones de Reporte")
        report_type = st.sidebar.radio("Selecciona el tipo de reporte:", ["Estado de Resultados", "Balance General"])

        if report_type == "Estado de Resultados":
            st.header("📈 Estado de Resultados por Centro de Costo")
            
            # Opciones de filtro de centro de costo (usando los nombres lógicos)
            # Asegúrate de que solo se muestren los CC que realmente existen en las columnas del DF
            actual_cc_logical_names = [
                name for excel_name, name in COL_CONFIG['EDO_RESULTADO']['CENTROS_COSTO_COLS'].items()
                if name in df_er.columns and excel_name != 'Total' # Excluir 'Total' de la lista de selección individual
            ]
            selected_cc = st.sidebar.selectbox("Selecciona un Centro de Costo:", ['Todos'] + actual_cc_logical_names)
            
            # Filtro de nivel de detalle
            # Asegúrate de que el nivel de línea exista y sea numérico para min/max
            level_options = sorted(df_er[COL_CONFIG['EDO_RESULTADO']['NIVEL_LINEA']].dropna().astype(int).unique().tolist())
            if not level_options:
                level_options = [1] # Default if no levels found
            
            max_level = st.sidebar.slider("Nivel de Detalle (Estado de Resultados):", 
                                          min_value=min(level_options), 
                                          max_value=max(level_options), 
                                          value=min(level_options)) # Por defecto, el nivel más alto

            final_er_display = generate_financial_statement(df_er, 'Estado de Resultados', selected_cc, max_level)
            
            st.dataframe(final_er_display, use_container_width=True, hide_index=True)
            
            # Opcional: Gráficos de ER
            if not final_er_display.empty:
                st.subheader("Visualización del Estado de Resultados")
                # Excluir la fila de 'TOTAL ESTADO DE RESULTADOS' para el gráfico y líneas en blanco
                chart_data = final_er_display[
                    ~final_er_display[COL_CONFIG['EDO_RESULTADO']['NOMBRE_CUENTA']].str.contains('TOTAL', na=False) &
                    (final_er_display[COL_CONFIG['EDO_RESULTADO']['NOMBRE_CUENTA']] != '')
                ].copy()
                
                if not chart_data.empty:
                    # Las sumas deben basarse en las clasificaciones que hizo `classify_account`
                    ingresos_val = chart_data[chart_data['Tipo_Estado'].str.contains('Ingresos', na=False)]['Valor'].sum()
                    costos_val = chart_data[chart_data['Tipo_Estado'].str.contains('Costo de Ventas', na=False)]['Valor'].sum()
                    gastos_op_val = chart_data[chart_data['Tipo_Estado'].str.contains('Gastos Operacionales', na=False)]['Valor'].sum()
                    gastos_no_op_val = chart_data[chart_data['Tipo_Estado'].str.contains('Gastos no Operacionales', na=False)]['Valor'].sum()
                    
                    chart_summary = pd.DataFrame({
                        'Categoría': ['Ingresos', 'Costo de Ventas', 'Gastos Operacionales', 'Gastos no Operacionales'],
                        'Valor': [ingresos_val, costos_val, gastos_op_val, gastos_no_op_val]
                    })
                    st.bar_chart(chart_summary.set_index('Categoría'))


        elif report_type == "Balance General":
            st.header("💰 Balance General")
            
            level_options_bg = sorted(df_bg[COL_CONFIG['BALANCE']['NIVEL_LINEA']].dropna().astype(int).unique().tolist())
            if not level_options_bg:
                level_options_bg = [1] # Default if no levels found

            max_level_bg = st.sidebar.slider("Nivel de Detalle (Balance General):", 
                                             min_value=min(level_options_bg), 
                                             max_value=max(level_options_bg), 
                                             value=min(level_options_bg))
            
            final_bg_display = generate_financial_statement(df_bg, 'Balance General', max_level=max_level_bg)
            
            st.dataframe(final_bg_display, use_container_width=True, hide_index=True)
            
            # Opcional: KPIs o Gráficos de Balance
            if not final_bg_display.empty:
                st.subheader("Indicadores Clave del Balance")
                
                # Obtener los totales de las filas de resumen
                total_act = final_bg_display[final_bg_display[COL_CONFIG['BALANCE']['NOMBRE_CUENTA']] == 'TOTAL ACTIVOS']['Valor'].iloc[0] if 'TOTAL ACTIVOS' in final_bg_display[COL_CONFIG['BALANCE']['NOMBRE_CUENTA']].values else 0
                total_pas = final_bg_display[final_bg_display[COL_CONFIG['BALANCE']['NOMBRE_CUENTA']] == 'TOTAL PASIVOS']['Valor'].iloc[0] if 'TOTAL PASIVOS' in final_bg_display[COL_CONFIG['BALANCE']['NOMBRE_CUENTA']].values else 0
                total_pat = final_bg_display[final_bg_display[COL_CONFIG['BALANCE']['NOMBRE_CUENTA']] == 'TOTAL PATRIMONIO']['Valor'].iloc[0] if 'TOTAL PATRIMONIO' in final_bg_display[COL_CONFIG['BALANCE']['NOMBRE_CUENTA']].values else 0

                st.write(f"**Total Activos:** {total_act:,.2f}")
                st.write(f"**Total Pasivos:** {total_pas:,.2f}")
                st.write(f"**Total Patrimonio:** {total_pat:,.2f}")

                if total_pas != 0:
                    st.write(f"**Razón Corriente (Activo/Pasivo):** {total_act / total_pas:.2f}")
                else:
                    st.write("**Razón Corriente:** N/A (Pasivos en cero)")
                
                # Gráfico de activos, pasivos y patrimonio
                balance_chart_data = pd.DataFrame({
                    'Categoría': ['Activos', 'Pasivos', 'Patrimonio'],
                    'Valor': [total_act, total_pas, total_pat]
                })
                st.bar_chart(balance_chart_data.set_index('Categoría'))


        # Botón para descargar el Excel completo (ER por CC y BG)
        st.sidebar.markdown("---")
        st.sidebar.subheader("Exportar a Excel")
        
        # Para la exportación del ER completo, usaremos el DataFrame df_er que ya tiene los CCs renombrados lógicamente
        er_export_df = df_er[[
            COL_CONFIG['EDO_RESULTADO']['NIVEL_LINEA'],
            COL_CONFIG['EDO_RESULTADO']['CUENTA'],
            COL_CONFIG['EDO_RESULTADO']['NOMBRE_CUENTA']
        ] + list(COL_CONFIG['EDO_RESULTADO']['CENTROS_COSTO_COLS'].values())].copy() # Incluye todas las columnas de CC

        # Calcular los totales para cada centro de costo en la última fila del Excel exportado
        total_row_er_export = {
            COL_CONFIG['EDO_RESULTADO']['NIVEL_LINEA']: '',
            COL_CONFIG['EDO_RESULTADO']['CUENTA']: '',
            COL_CONFIG['EDO_RESULTADO']['NOMBRE_CUENTA']: 'TOTALES'
        }
        for logical_cc_name in COL_CONFIG['EDO_RESULTADO']['CENTROS_COSTO_COLS'].values():
            if logical_cc_name in er_export_df.columns:
                total_row_er_export[logical_cc_name] = er_export_df[logical_cc_name].sum()
            else:
                total_row_er_export[logical_cc_name] = 0 # En caso de que la columna no exista (raro)
        
        er_export_df.loc[len(er_export_df)] = total_row_er_export

        # Para el BG, simplemente la tabla final que ya tiene el total
        # Asegurarse de que `final_bg_display` se ha generado, si no, generarlo
        # para que siempre esté disponible para la descarga
        if 'final_bg_display' not in locals() or final_bg_display.empty:
            # Generamos un BG por defecto para la exportación si no se generó antes
            # Usamos el nivel máximo de detalle para la exportación completa.
            max_level_for_export_bg = max(df_bg[COL_CONFIG['BALANCE']['NIVEL_LINEA']].dropna().astype(int).unique().tolist()) if not df_bg[COL_CONFIG['BALANCE']['NIVEL_LINEA']].dropna().empty else 999
            final_bg_display = generate_financial_statement(df_bg, 'Balance General', max_level=max_level_for_export_bg)

        bg_export_df = final_bg_display.copy()

        excel_buffer = to_excel_buffer(er_export_df, bg_export_df)
        st.sidebar.download_button(
            label="Descargar Reporte Completo (Excel)",
            data=excel_buffer,
            file_name="reporte_financiero_completo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
    except FileNotFoundError:
        st.error("Error: Asegúrate de que el archivo Excel contiene las hojas 'EDO RESULTADO' y 'BALANCE' exactamente.")
    except KeyError as e:
        st.error(f"Error: No se encontró la columna esperada '{e}'. Por favor, verifica los nombres de las columnas en tu Excel y ajusta la sección 'COL_CONFIG' en el código si es necesario.")
        st.info("Asegúrate de que los nombres de las columnas en tu Excel coincidan con los especificados en COL_CONFIG, incluyendo los centros de costo y la columna 'Total'.")
    except Exception as e:
        st.error(f"Ocurrió un error inesperado al procesar el archivo: {e}")
        st.info(f"Detalle del error: {type(e).__name__} - {e}. Revisa el formato de los datos, especialmente los valores numéricos y la presencia de todas las columnas esperadas.")
