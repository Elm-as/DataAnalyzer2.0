"""
Data loading module for DataAnalyzer 2.0
Supports CSV, Excel, JSON formats
"""
import pandas as pd
import os
from typing import Tuple, Dict, Optional

def load_data(file_path: str, separator: str = ',') -> Tuple[Optional[pd.DataFrame], str]:
    """
    Charge un fichier de données
    
    Args:
        file_path: Chemin vers le fichier
        separator: Séparateur pour CSV (, ou ;)
        
    Returns:
        (DataFrame, message d'erreur si échec)
    """
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension in ['.csv', '.txt']:
            # Essayer d'abord avec le séparateur spécifié
            try:
                df = pd.read_csv(file_path, sep=separator, encoding='utf-8')
            except:
                # Essayer avec l'autre séparateur
                other_sep = ';' if separator == ',' else ','
                try:
                    df = pd.read_csv(file_path, sep=other_sep, encoding='utf-8')
                except:
                    # Essayer avec utf-8-sig pour les fichiers avec BOM
                    try:
                        df = pd.read_csv(file_path, sep=separator, encoding='utf-8-sig')
                    except:
                        # Dernier recours: latin-1
                        df = pd.read_csv(file_path, sep=separator, encoding='latin-1')
                    
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, engine='openpyxl' if file_extension == '.xlsx' else None)
            
        elif file_extension == '.json':
            df = pd.read_json(file_path)
            
        else:
            return None, f"Format de fichier non supporté: {file_extension}"
        
        if df.empty:
            return None, "Le fichier est vide"
            
        return df, ""
        
    except Exception as e:
        return None, f"Erreur lors du chargement: {str(e)}"

def get_data_preview(df: pd.DataFrame, n_rows: int = 10) -> Dict:
    """
    Retourne un aperçu des données
    
    Args:
        df: DataFrame à prévisualiser
        n_rows: Nombre de lignes à afficher
        
    Returns:
        Dictionnaire avec les informations d'aperçu
    """
    return {
        'head': df.head(n_rows),
        'tail': df.tail(n_rows),
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict()
    }

def get_file_info(file_path: str) -> Dict:
    """
    Retourne les informations sur un fichier
    
    Args:
        file_path: Chemin vers le fichier
        
    Returns:
        Dictionnaire avec les informations du fichier
    """
    if not os.path.exists(file_path):
        return {}
    
    file_size = os.path.getsize(file_path)
    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_name)[1]
    
    # Convertir la taille en unité lisible
    size_units = ['B', 'KB', 'MB', 'GB']
    size_idx = 0
    size = file_size
    while size > 1024 and size_idx < len(size_units) - 1:
        size /= 1024
        size_idx += 1
    
    return {
        'name': file_name,
        'extension': file_ext,
        'size': f"{size:.2f} {size_units[size_idx]}",
        'size_bytes': file_size
    }

def validate_file_size(file_path: str, max_size_mb: int = 100) -> Tuple[bool, str]:
    """
    Valide la taille d'un fichier
    
    Args:
        file_path: Chemin vers le fichier
        max_size_mb: Taille maximale en MB
        
    Returns:
        (is_valid, message)
    """
    file_info = get_file_info(file_path)
    if not file_info:
        return False, "Fichier introuvable"
    
    size_mb = file_info['size_bytes'] / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"Fichier trop volumineux ({size_mb:.1f}MB > {max_size_mb}MB)"
    
    return True, ""

def save_uploaded_file(uploaded_file, upload_dir: str = 'data/uploads') -> Tuple[Optional[str], str]:
    """
    Sauvegarde un fichier uploadé
    
    Args:
        uploaded_file: Fichier uploadé (Streamlit)
        upload_dir: Répertoire de destination
        
    Returns:
        (chemin du fichier, message d'erreur)
    """
    try:
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, uploaded_file.name)
        
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path, ""
    except Exception as e:
        return None, f"Erreur lors de la sauvegarde: {str(e)}"
