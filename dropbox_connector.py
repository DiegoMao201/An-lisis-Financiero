# dropbox_connector.py

import dropbox
import streamlit as st
import os # Necesario para manejar nombres de archivo

@st.cache_data(ttl=3600) # Cache por 1 hora
def get_dropbox_client():
    """Se conecta a Dropbox usando los secrets y devuelve un cliente."""
    try:
        app_key = st.secrets["dropbox"]["app_key"]
        app_secret = st.secrets["dropbox"]["app_secret"]
        refresh_token = st.secrets["dropbox"]["refresh_token"]

        dbx = dropbox.Dropbox(
            oauth2_refresh_token=refresh_token,
            app_key=app_key,
            app_secret=app_secret
        )
        dbx.users_get_current_account()
        return dbx
    except Exception as e:
        st.error(f"Error al conectar con Dropbox. Verifica tus secrets. Detalle: {e}")
        return None

# ==============================================================================
#                      FUNCIÓN CORREGIDA Y ACTUALIZADA
# ==============================================================================
@st.cache_data(ttl=3600)
def find_financial_files(_dbx, base_folder="/data"):
    """
    Busca archivos Excel directamente en la carpeta base (ej. /data/2025_04.xlsx).
    El nombre del archivo (sin la extensión) se usa como el periodo.
    """
    files_to_process = []
    if not _dbx:
        return files_to_process
    try:
        # Listamos todo el contenido directamente de la carpeta base (ej. /data)
        for entry in _dbx.files_list_folder(base_folder).entries:
            # Nos aseguramos de que sea un ARCHIVO y que termine en .xlsx o .xls
            if isinstance(entry, dropbox.files.FileMetadata) and \
               (entry.name.endswith('.xlsx') or entry.name.endswith('.xls')):
                
                # Extraemos el nombre del archivo SIN la extensión.
                # Ejemplo: '2025_04.xlsx' -> '2025_04'
                file_name_without_ext, _ = os.path.splitext(entry.name)
                
                files_to_process.append({
                    # Usamos el nombre del archivo como periodo, reemplazando '_' por '-' para consistencia
                    "periodo": file_name_without_ext.replace('_', '-'),
                    # Guardamos la ruta completa del archivo
                    "path": entry.path_lower
                })
    except Exception as e:
        st.error(f"Error al buscar archivos en Dropbox en la carpeta '{base_folder}': {e}")
    return files_to_process

def load_excel_from_dropbox(_dbx, file_path):
    """Descarga un archivo de Excel y devuelve su contenido en bytes."""
    try:
        _, res = _dbx.files_download(path=file_path)
        return res.content
    except Exception as e:
        st.error(f"No se pudo descargar el archivo {file_path}: {e}")
        return None
