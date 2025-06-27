# dropbox_connector.py

import dropbox
import streamlit as st

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

@st.cache_data(ttl=3600)
def find_financial_files(_dbx, base_folder="/data"):
    """Busca archivos Excel en subcarpetas de periodo (ej. /data/2025_01)."""
    files_to_process = []
    if not _dbx:
        return files_to_process
    try:
        for entry in _dbx.files_list_folder(base_folder).entries:
            if isinstance(entry, dropbox.files.FolderMetadata):
                period_folder_name = entry.name
                period_folder_path = entry.path_lower
                for sub_entry in _dbx.files_list_folder(period_folder_path).entries:
                    if isinstance(sub_entry, dropbox.files.FileMetadata) and (sub_entry.name.endswith('.xlsx') or sub_entry.name.endswith('.xls')):
                        files_to_process.append({
                            "periodo": period_folder_name.replace('_', '-'),
                            "path": sub_entry.path_lower
                        })
                        break
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
