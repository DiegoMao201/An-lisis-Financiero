# kpis_y_analisis.py
import pandas as pd
import streamlit as st
import google.generativeai as genai
from mi_logica_original import get_principal_account_value, COL_CONFIG
import numpy as np # Importamos numpy para manejar divisiones por cero de forma segura

def calcular_kpis_periodo(df_er: pd.DataFrame, df_bg: pd.DataFrame, cc_filter: str = 'Todos') -> dict:
    """
    Calcula un set de KPIs para un único periodo.
    Esta función es la ÚNICA fuente de verdad para los cálculos numéricos.
    Se adapta para calcular el consolidado ('Todos') o un centro de costo específico.
    *** LÓGICA CORREGIDA ***
    """
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
        total_col_name = er_conf.get('CENTROS_COSTO_COLS', {}).get('Total')
        if total_col_name and total_col_name in df_er.columns:
            val_col_kpi = total_col_name
        else:
            ind_cc_cols = [v for k, v in er_conf.get('CENTROS_COSTO_COLS', {}).items() if str(k).lower() not in ['total', 'sin centro de coste'] and v in df_er.columns]
            if ind_cc_cols:
                df_er['__temp_sum_kpi'] = df_er.loc[:, ind_cc_cols].sum(axis=1)
                val_col_kpi = '__temp_sum_kpi'
            else:
                scc_name = er_conf.get('CENTROS_COSTO_COLS', {}).get('Sin centro de coste')
                if scc_name and scc_name in df_er.columns:
                    val_col_kpi = scc_name
                elif 'Total_Consolidado_ER' in df_er.columns:
                    val_col_kpi = 'Total_Consolidado_ER'

    if not val_col_kpi or val_col_kpi not in df_er.columns: return kpis

    # --- EXTRACCIÓN CON LÓGICA DE SIGNOS INVERTIDA (COMO EN EL ORIGEN) ---
    cuenta_er = er_conf['CUENTA']
    # Ingresos vienen como negativos
    ingresos_raw = get_principal_account_value(df_er, '4', val_col_kpi, cuenta_er)
    # Costos y Gastos vienen como positivos
    costo_ventas = get_principal_account_value(df_er, '6', val_col_kpi, cuenta_er)
    gastos_admin = get_principal_account_value(df_er, '51', val_col_kpi, cuenta_er)
    gastos_ventas = get_principal_account_value(df_er, '52', val_col_kpi, cuenta_er)
    costos_prod = get_principal_account_value(df_er, '7', val_col_kpi, cuenta_er)
    gastos_no_op = get_principal_account_value(df_er, '53', val_col_kpi, cuenta_er)
    impuestos = get_principal_account_value(df_er, '54', val_col_kpi, cuenta_er)
    
    # --- CÁLCULOS INTERMEDIOS (TODAVÍA CON SIGNOS INVERTIDOS) ---
    utilidad_bruta_raw = ingresos_raw + costo_ventas # (Ej: -1000 + 400 = -600)
    gastos_operativos = gastos_admin + gastos_ventas + costos_prod
    utilidad_operacional_raw = utilidad_bruta_raw + gastos_operativos # (Ej: -600 + 300 = -300)
    utilidad_neta_raw = utilidad_operacional_raw + gastos_no_op + impuestos # (Ej: -300 + 50 = -250)

    # --- AJUSTE Y ESTANDARIZACIÓN DE SIGNOS PARA KPIS FINALES ---
    # Convertimos valores a la lógica estándar: Ganancia > 0, Pérdida < 0
    # Multiplicamos por -1 para invertir el signo y estandarizar.
    ingresos = ingresos_raw * -1
    utilidad_bruta = utilidad_bruta_raw * -1
    utilidad_operacional = utilidad_operacional_raw * -1
    utilidad_neta = utilidad_neta_raw * -1

    kpis['ingresos'] = ingresos
    kpis['costo_ventas'] = costo_ventas
    kpis['gastos_operativos'] = gastos_operativos
    kpis['utilidad_bruta'] = utilidad_bruta
    kpis['utilidad_operacional'] = utilidad_operacional
    kpis['utilidad_neta'] = utilidad_neta
    
    # --- EXTRACCIÓN DE BALANCE GENERAL (VALORES YA POSITIVOS) ---
    cuenta_bg = bg_conf['CUENTA']
    saldo_final_col = bg_conf['SALDO_FINAL']

    activo = get_principal_account_value(df_bg, '1', saldo_final_col, cuenta_bg)
    pasivo = get_principal_account_value(df_bg, '2', saldo_final_col, cuenta_bg)
    patrimonio = get_principal_account_value(df_bg, '3', saldo_final_col, cuenta_bg)
    activo_corriente = sum([get_principal_account_value(df_bg, c, saldo_final_col, cuenta_bg) for c in ['11','12','13','14']])
    inventarios = get_principal_account_value(df_bg, '14', saldo_final_col, cuenta_bg)
    pasivo_corriente = sum([get_principal_account_value(df_bg, c, saldo_final_col, cuenta_bg) for c in ['21','22','23']])

    kpis['activo'] = activo
    kpis['pasivo'] = pasivo
    kpis['patrimonio'] = patrimonio
    kpis['activo_corriente'] = activo_corriente
    kpis['pasivo_corriente'] = pasivo_corriente
    kpis['inventarios'] = inventarios
    
    # --- CÁLCULO DE RATIOS CON VALORES ESTANDARIZADOS Y SEGUROS ---
    # Usamos np.divide para manejar divisiones por cero de forma segura y abs() para asegurar que los denominadores sean positivos.
    kpis['razon_corriente'] = np.divide(abs(activo_corriente), abs(pasivo_corriente)) if pasivo_corriente != 0 else 0
    kpis['endeudamiento_activo'] = np.divide(abs(pasivo), abs(activo)) if activo != 0 else 0
    
    # ROE: La utilidad neta ya tiene el signo correcto. Positivo=Ganancia, Negativo=Pérdida.
    kpis['roe'] = np.divide(utilidad_neta, abs(patrimonio)) if patrimonio != 0 else 0
    
    # Márgenes: Usan la utilidad (ya con signo correcto) y los ingresos (en valor absoluto).
    kpis['margen_neto'] = np.divide(utilidad_neta, abs(ingresos)) if ingresos != 0 else 0
    kpis['margen_operacional'] = np.divide(utilidad_operacional, abs(ingresos)) if ingresos != 0 else 0
    
    kpis['rotacion_activos'] = np.divide(abs(ingresos), abs(activo)) if activo != 0 else 0
    kpis['apalancamiento'] = np.divide(abs(activo), abs(patrimonio)) if patrimonio != 0 else 0

    if '__temp_sum_kpi' in df_er.columns:
        df_er.drop(columns=['__temp_sum_kpi'], inplace=True)
        
    return kpis

def preparar_datos_tendencia(datos_historicos: dict) -> pd.DataFrame:
    """Convierte el diccionario de datos históricos en un DataFrame para graficar tendencias."""
    lista_periodos = [
        dict(periodo=periodo, **data['kpis']['Todos'])
        for periodo, data in datos_historicos.items()
        if 'Todos' in data.get('kpis', {})
    ]
    if not lista_periodos: return pd.DataFrame()
    df_tendencia = pd.DataFrame(lista_periodos)
    df_tendencia['periodo'] = pd.to_datetime(df_tendencia['periodo'], format='%Y-%m')
    df_tendencia = df_tendencia.sort_values(by='periodo').reset_index(drop=True)
    return df_tendencia

@st.cache_data(show_spinner=False)
def generar_analisis_avanzado_ia(contexto_ia: dict):
    """Genera un análisis financiero profundo usando un contexto enriquecido y con instrucciones reforzadas."""
    try:
        api_key = st.secrets["google_ai"]["api_key"]
        genai.configure(api_key=api_key)
    except Exception:
        return "🔴 **Error:** No se encontró la clave de API de Google AI."

    kpis = contexto_ia.get("kpis", {})
    periodo = contexto_ia.get("periodo", "N/A")
    cc = contexto_ia.get("centro_costo", "N/A")
    convencion = contexto_ia.get("convencion_contable", "")
    favorables = contexto_ia.get("variaciones_favorables", [])
    desfavorables = contexto_ia.get("variaciones_desfavorables", [])

    fav_str = "\n".join([f"- {item['Descripción']}: ${item['Variacion_Absoluta']:,.0f}" for item in favorables]) if favorables else "No hubo variaciones favorables significativas."
    des_str = "\n".join([f"- {item['Descripción']}: ${item['Variacion_Absoluta']:,.0f}" for item in desfavorables]) if desfavorables else "No hubo variaciones desfavorables significativas."

    prompt = f"""
    **Rol:** Eres un Asesor Financiero Estratégico y CFO virtual. Tu análisis debe ser agudo, directo y accionable.

    **REGLA DE ORO Y LÓGICA DE NEGOCIO (¡INSTRUCCIONES CRÍTICAS!):**
    {convencion}

    **INSTRUCCIONES DE INTERPRETACIÓN Y LENGUAJE (¡LA PARTE MÁS IMPORTANTE!):**
    Tu tarea tiene dos partes lógicas que debes seguir rigurosamente:

    **1. Para Analizar las *Variaciones* y sus *Causas* (Impactos Positivos y Negativos):**
    - Cuando una cuenta de **Ingreso** (código '4') se vuelve **MÁS NEGATIVA** (ej: de -70M a -240M), descríbelo como un **"CRECIMIENTO", "AUMENTO" o "INCREMENTO FAVORABLE"**.
    - **NUNCA** uses "caída" o "reducción" para describir este evento. Es un **CRECIMIENTO POSITIVO** para el negocio.
    - Si la **'Variacion_Absoluta'** en la tabla de impactos es **NEGATIVA**, significa que hubo un **CRECIMIENTO o MEJORA**. Un número como -170,000,000 es un **FUERTE INCREMENTO**.
    - Para cuentas de **Gasto/Costo** (códigos '5', '6', '7'), un aumento en su valor (positivo) es un impacto desfavorable.

    **2. Para Analizar los *Indicadores Clave (KPIs) Finales*:**
    - Los KPIs que te proporciono ya están **ESTANDARIZADOS** a la lógica financiera universal.
    - **Utilidad Neta:** Un valor **POSITIVO es una GANANCIA ✅**. Un valor **NEGATIVO es una PÉRDIDA 🔻**.
    - **ROE, Razón Corriente, Márgenes:** Interpreta estos ratios de la forma estándar. Un ROE negativo indica una pérdida neta. Un margen negativo indica que los costos superaron los ingresos.

    **Contexto de Análisis:**
    - **Periodo:** {periodo}
    - **Unidad de Negocio:** "{cc}"

    **Indicadores Clave (KPIs) del Periodo (Lógica Estándar):**
    - **Utilidad Neta:** ${kpis.get('utilidad_neta', 0):,.0f}
    - **Margen Neto:** {kpis.get('margen_neto', 0):.2%}
    - **ROE (Rentabilidad sobre Patrimonio):** {kpis.get('roe', 0):.2%}
    - **Razón Corriente (Liquidez):** {kpis.get('razon_corriente', 0):.2f}
    - **Nivel de Endeudamiento:** {kpis.get('endeudamiento_activo', 0):.2%}

    **Análisis Comparativo vs. Periodo Anterior (Causas):**
    - **Principales Impactos Positivos (Crecimientos/Mejoras):**
    {fav_str}
    - **Principales Impactos Negativos (Deterioros):**
    {des_str}

    **Instrucciones de Respuesta:**
    Con base en TODA la información y respetando rigurosamente las reglas de interpretación, genera un informe ejecutivo. Usa emojis (📈, ⚠️, ✅, 💡, 🔻).

    ### Diagnóstico General 🎯
    (Ofrece un veredicto claro sobre la salud financiera. Indica si hubo ganancia o pérdida, y luego explica las causas usando la lógica de variaciones. Ejemplo: "El periodo cerró con una ganancia/pérdida neta de $X. Este resultado fue impulsado principalmente por un fuerte crecimiento en los ingresos de Y, aunque se vio contrarrestado por un aumento en los gastos Z.")

    ### Puntos Clave del Análisis 🔑
    (Detalla 2-3 observaciones importantes. Conecta los KPIs finales con las variaciones que los causaron.)

    ### Plan de Acción Recomendado 💡
    (Proporciona 2-3 recomendaciones específicas basadas en tu diagnóstico.)
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        cleaned_response = response.text.replace('•', '*')
        return cleaned_response
    except Exception as e:
        return f"🔴 **Error al contactar la IA:** {e}."

@st.cache_data(show_spinner=False)
def generar_analisis_tendencia_ia(_df_tendencia: pd.DataFrame):
    """Genera un análisis de EVOLUCIÓN y TENDENCIA con instrucciones reforzadas."""
    try:
        api_key = st.secrets["google_ai"]["api_key"]
        genai.configure(api_key=api_key)
    except Exception:
        return "🔴 **Error:** No se encontró la clave de API de Google AI."

    if _df_tendencia.empty or len(_df_tendencia) < 2:
        return "ℹ️ Se necesitan al menos dos periodos para un análisis de tendencia."

    primer_periodo = _df_tendencia.iloc[0]
    ultimo_periodo = _df_tendencia.iloc[-1]
    
    resumen_datos = f"""
    - **Periodo Analizado:** De {primer_periodo['periodo'].strftime('%Y-%m')} a {ultimo_periodo['periodo'].strftime('%Y-%m')}.
    - **Ingresos (Estandarizados a positivo):** Evolucionaron de ${primer_periodo['ingresos']:,.0f} a ${ultimo_periodo['ingresos']:,.0f}.
    - **Utilidad Neta (Positivo=Ganancia):** Evolucionó de ${primer_periodo['utilidad_neta']:,.0f} a ${ultimo_periodo['utilidad_neta']:,.0f}.
    - **Margen Neto:** Evolucionó de {primer_periodo['margen_neto']:.2%} a {ultimo_periodo['margen_neto']:.2%}.
    - **Razón Corriente (Liquidez):** Varió de {primer_periodo['razon_corriente']:.2f} a {ultimo_periodo['razon_corriente']:.2f}.
    - **ROE:** Se movió de {primer_periodo['roe']:.2%} a {ultimo_periodo['roe']:.2%}.
    """

    prompt = f"""
    **Rol:** Eres un Analista Financiero Senior y Asesor Estratégico.

    **¡REGLA DE ORO Y LÓGICA DE NEGOCIO (INSTRUCCIÓN CRÍTICA E INELUDIBLE)!**
    - Los datos que te proporciono en el resumen ya están **ESTANDARIZADOS** a la lógica financiera universal.
    - **Utilidad Neta POSITIVA es GANANCIA ✅.**
    - **Utilidad Neta NEGATIVA es PÉRDIDA 🔻.**

    **INSTRUCCIONES DE INTERPRETACIÓN Y LENGUAJE (¡LA PARTE MÁS IMPORTANTE!):**
    - Tu tarea es analizar la **TENDENCIA** de estos indicadores estandarizados.
    - Si la utilidad neta evoluciona de $500 a $600, es una **MEJORA y un CRECIMIENTO** en la rentabilidad.
    - Si la utilidad neta evoluciona de $500 a -$100, es un **DETERIORO SIGNIFICATIVO**, pasando de ganancia a pérdida.
    - Si la utilidad neta evoluciona de -$200 a -$100, es una **MEJORA**, ya que la pérdida se redujo.
    - Describe la evolución de los ingresos como crecimiento o decrecimiento de forma normal.

    **Contexto:** Has analizado la evolución financiera de la compañía. Aquí está el resumen de la trayectoria:
    {resumen_datos}
    
    **Instrucciones de Respuesta:**
    Con base en TODA la información y respetando rigurosamente las reglas de interpretación, genera un informe de evolución.

    ### Veredicto Estratégico 📜
    (En un párrafo, da un veredicto sobre la trayectoria. Ejemplo: "La compañía muestra una tendencia de crecimiento/decrecimiento en su rentabilidad. La utilidad neta ha mejorado/empeorado, pasando de X a Y en el periodo analizado...")

    ### Análisis de Evolución por Área 🔍
    - **Rentabilidad (Utilidad, Márgenes, ROE):** Analiza la tendencia de la rentabilidad. ¿La empresa es más o menos rentable que antes? ¿Por qué?
    - **Crecimiento y Eficiencia:** ¿El crecimiento de los ingresos se traduce en mayor rentabilidad? ¿O los costos están creciendo más rápido?
    - **Salud y Riesgo Financiero:** ¿La posición de liquidez (Razón Corriente) ha mejorado o empeorado?

    ### Prioridades para el Próximo Trimestre 🎯
    (Basado en la evolución, proporciona 2-3 prioridades estratégicas.)
    """

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        cleaned_response = response.text.replace('•', '*')
        return cleaned_response
    except Exception as e:
        return f"🔴 **Error al contactar la IA:** {e}."

@st.cache_data(show_spinner=False)
def generar_analisis_con_prompt_libre(prompt_personalizado: str):
    """
    Genera un análisis de IA a partir de un prompt libre y directo.
    Ideal para casos donde no se necesita un contexto estructurado.
    """
    try:
        api_key = st.secrets["google_ai"]["api_key"]
        genai.configure(api_key=api_key)
    except Exception:
        return "🔴 **Error:** No se encontró la clave de API de Google AI."

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt_personalizado)
        cleaned_response = response.text.replace('•', '*')
        return cleaned_response
    except Exception as e:
        return f"🔴 **Error al contactar la IA:** {e}."
