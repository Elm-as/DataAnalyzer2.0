"""
Export module for DataAnalyzer 2.0
Exports: Reports (PDF/HTML), Python code, Models, Data
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
import pickle
from datetime import datetime
import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted

def generate_python_code(analysis_type: str, params: Dict) -> str:
    """
    Génère du code Python reproductible
    
    Args:
        analysis_type: Type d'analyse
        params: Paramètres de l'analyse
        
    Returns:
        Code Python sous forme de string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    code_template = f"""
# Code généré par DataAnalyzer 2.0
# Date: {timestamp}
# Analyse: {analysis_type}

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Charger les données
df = pd.read_csv('votre_fichier.csv')

# Paramètres utilisés
params = {params}

# Votre code d'analyse ici...
"""
    
    return code_template

def export_model(model, filepath: str) -> bool:
    """
    Exporte un modèle entraîné
    
    Args:
        model: Modèle scikit-learn
        filepath: Chemin de sauvegarde
        
    Returns:
        Succès de l'export
    """
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        return True
    except Exception as e:
        print(f"Erreur export modèle: {e}")
        return False

def export_data(df: pd.DataFrame, filepath: str, format: str = 'csv') -> bool:
    """
    Exporte les données transformées
    
    Args:
        df: DataFrame à exporter
        filepath: Chemin de sauvegarde
        format: Format (csv, excel, json)
        
    Returns:
        Succès de l'export
    """
    try:
        if format == 'csv':
            df.to_csv(filepath, index=False)
        elif format == 'excel':
            df.to_excel(filepath, index=False)
        elif format == 'json':
            df.to_json(filepath, orient='records')
        return True
    except Exception as e:
        print(f"Erreur export données: {e}")
        return False

def generate_html_report(results: Dict, title: str = "Rapport d'Analyse") -> str:
    """
    Génère un rapport HTML
    
    Args:
        results: Résultats de l'analyse
        title: Titre du rapport
        
    Returns:
        HTML sous forme de string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .section {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background-color: #ecf0f1;
            border-radius: 5px;
        }}
        .metric-label {{
            font-size: 12px;
            color: #7f8c8d;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #34495e;
            color: white;
        }}
        .footer {{
            text-align: center;
            color: #7f8c8d;
            margin-top: 40px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>Généré le {timestamp} par DataAnalyzer 2.0</p>
    </div>
    
    <div class="section">
        <h2>Résultats</h2>
        <pre>{json.dumps(results, indent=2, ensure_ascii=False)}</pre>
    </div>
    
    <div class="footer">
        <p>DataAnalyzer 2.0 - Plateforme d'Analyse de Données</p>
    </div>
</body>
</html>
"""
    
    return html


def generate_pdf_report(results: Dict, filepath: str, title: str = "Rapport d'Analyse") -> bool:
    """Génère un rapport PDF minimal et autonome.

    Args:
        results: Résultats à inclure dans le PDF.
        filepath: Chemin de sortie (.pdf).
        title: Titre du rapport.

    Returns:
        True si le PDF est généré.
    """
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        styles = getSampleStyleSheet()

        story = []
        story.append(Paragraph(title, styles['Title']))
        story.append(Paragraph(f"Généré le {ts} par DataAnalyzer 2.0", styles['Normal']))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Résultats (JSON)", styles['Heading2']))
        pretty = json.dumps(results, indent=2, ensure_ascii=False)
        story.append(Preformatted(pretty, styles['Code']))

        doc.build(story)
        return True
    except Exception as e:
        print(f"Erreur export PDF: {e}")
        return False

def save_session(session_data: Dict, filepath: str) -> bool:
    """
    Sauvegarde une session complète
    
    Args:
        session_data: Données de la session
        filepath: Chemin de sauvegarde (JSON)
        
    Returns:
        Succès de la sauvegarde
    """
    try:
        # Convertir les objets non sérialisables
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            return obj
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False, default=convert)
        return True
    except Exception as e:
        print(f"Erreur sauvegarde session: {e}")
        return False

def load_session(filepath: str) -> Optional[Dict]:
    """
    Charge une session sauvegardée
    
    Args:
        filepath: Chemin du fichier JSON
        
    Returns:
        Données de la session ou None
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Erreur chargement session: {e}")
        return None
