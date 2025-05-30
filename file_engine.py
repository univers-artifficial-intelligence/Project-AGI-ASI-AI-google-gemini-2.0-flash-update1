import os
import uuid
import shutil
import mimetypes
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import sqlite3
from datetime import datetime, timedelta
import json

# Bibliothèques pour l'extraction de texte
import PyPDF2
# Correction de l'importation du module python-docx
try:
    import docx  # Le nom du module importé est 'docx', pas 'python_docx'
except ImportError:
    # Fallback si le module n'est pas disponible
    docx = None
import csv
try:
    import pandas as pd
except ImportError:
    # Fallback si pandas n'est pas disponible
    pd = None
import chardet

from database import DB_PATH

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {
    'pdf', 'doc', 'docx', 'txt', 'csv', 'xls', 'xlsx',
    'json', 'html', 'xml', 'md', 'rtf'
}
FILE_RETENTION_DAYS = 7  # Durée de conservation des fichiers

# Créer le dossier d'upload s'il n'existe pas
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def init_file_db():
    """Initialise la table de fichiers dans la base de données"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Création de la table de fichiers
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS uploaded_files (
        id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        original_filename TEXT NOT NULL,
        stored_filename TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_type TEXT NOT NULL,
        file_size INTEGER NOT NULL,
        content_summary TEXT,
        extracted_text TEXT,
        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        expiry_date TIMESTAMP,
        last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    conn.commit()
    conn.close()

def allowed_file(filename: str) -> bool:
    """Vérifie si l'extension du fichier est autorisée"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_safe_filename(original_filename: str) -> str:
    """Génère un nom de fichier sécurisé et unique"""
    # Prendre l'extension du fichier original
    if '.' in original_filename:
        ext = original_filename.rsplit('.', 1)[1].lower()
    else:
        ext = ''
    
    # Générer un identifiant unique
    unique_id = str(uuid.uuid4())
    
    # Créer le nouveau nom de fichier
    if ext:
        return f"{unique_id}.{ext}"
    else:
        return unique_id

def get_file_type(file_path: str) -> str:
    """Détermine le type de fichier en fonction de l'extension"""
    ext = os.path.splitext(file_path)[1].lower()
    
    file_types = {
        '.pdf': 'PDF',
        '.doc': 'Word',
        '.docx': 'Word',
        '.txt': 'Text',
        '.csv': 'CSV',
        '.xls': 'Excel',
        '.xlsx': 'Excel',
        '.json': 'JSON',
        '.html': 'HTML',
        '.xml': 'XML',
        '.md': 'Markdown',
        '.rtf': 'Rich Text'
    }
    
    return file_types.get(ext, 'Unknown')

def extract_text_from_pdf(file_path: str) -> str:
    """Extrait le texte d'un fichier PDF"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text() + "\n"
    except Exception as e:
        text = f"Erreur lors de l'extraction du texte du PDF: {str(e)}"
    
    return text

def extract_text_from_docx(file_path: str) -> str:
    """Extrait le texte d'un fichier Word (docx)"""
    text = ""
    try:
        if docx is not None:
            # Utiliser le module docx s'il est disponible
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            # Message alternatif si le module n'est pas disponible
            text = "Contenu du document Word (extraction indisponible). Pour analyser ce document, installez python-docx avec la commande 'pip install python-docx'."
    except Exception as e:
        text = f"Erreur lors de l'extraction du texte du document Word: {str(e)}"
    
    return text

def extract_text_from_txt(file_path: str) -> str:
    """Extrait le texte d'un fichier texte, avec détection de l'encodage"""
    try:
        # Détecter l'encodage du fichier
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'
        
        # Lire le fichier avec l'encodage détecté
        with open(file_path, 'r', encoding=encoding, errors='replace') as file:
            text = file.read()
        
        return text
    except Exception as e:
        return f"Erreur lors de l'extraction du texte: {str(e)}"

def extract_text_from_csv(file_path: str) -> str:
    """Extrait le texte d'un fichier CSV"""
    try:
        # Détecter l'encodage
        with open(file_path, 'rb') as file:
            raw_data = file.read(4096)  # Lire un échantillon pour la détection
            result = chardet.detect(raw_data)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'
        
        # Essayer de déterminer le délimiteur
        sniffer = csv.Sniffer()
        with open(file_path, 'r', encoding=encoding, errors='replace') as file:
            sample = file.read(4096)
            try:
                dialect = sniffer.sniff(sample)
                delimiter = dialect.delimiter
            except:
                delimiter = ','  # Par défaut, utiliser une virgule
        
        # Lire le CSV
        text = ""
        with open(file_path, 'r', encoding=encoding, errors='replace') as file:
            reader = csv.reader(file, delimiter=delimiter)
            for row in reader:
                text += " | ".join(row) + "\n"
        
        # Pour les fichiers volumineux, limiter le texte extrait
        if len(text) > 50000:
            text = text[:50000] + "...\n[Contenu tronqué pour raisons de taille]"
            
        return text
    except Exception as e:
        return f"Erreur lors de l'extraction du texte CSV: {str(e)}"

def extract_text_from_excel(file_path: str) -> str:
    """Extrait le texte d'un fichier Excel"""
    try:
        if pd is not None:
            # Utiliser pandas s'il est disponible
            df_dict = pd.read_excel(file_path, sheet_name=None)
            text = ""
            
            for sheet_name, df in df_dict.items():
                text += f"Feuille: {sheet_name}\n"
                sheet_text = df.to_string(index=False)
                
                # Pour les grands tableaux, limiter la taille
                if len(sheet_text) > 25000:
                    sheet_text = sheet_text[:25000] + "...\n[Contenu de la feuille tronqué]"
                    
                text += sheet_text + "\n\n"
                
            # Limiter la taille totale
            if len(text) > 50000:
                text = text[:50000] + "...\n[Contenu global tronqué]"
        else:
            text = "Contenu du fichier Excel (extraction indisponible). Pour analyser ce document, installez pandas et openpyxl avec la commande 'pip install pandas openpyxl'."
        
        return text
    except Exception as e:
        return f"Erreur lors de l'extraction du texte Excel: {str(e)}"

def extract_text_from_file(file_path: str) -> str:
    """
    Extrait le texte d'un fichier en fonction de son type
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # Dispatch vers la fonction d'extraction appropriée
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    elif file_extension == '.txt':
        return extract_text_from_txt(file_path)
    elif file_extension == '.csv':
        return extract_text_from_csv(file_path)
    elif file_extension in ['.xls', '.xlsx']:
        return extract_text_from_excel(file_path)
    else:
        return f"Type de fichier non pris en charge pour l'extraction de texte: {file_extension}"

def generate_content_summary(text: str, max_length: int = 500) -> str:
    """
    Génère un résumé du contenu du fichier
    """
    if not text or len(text.strip()) == 0:
        return "Aucun contenu textuel détecté."
    
    # Prendre les premières lignes non vides
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    summary = ""
    
    for line in lines:
        if len(summary) + len(line) <= max_length:
            summary += line + "\n"
        else:
            remaining = max_length - len(summary)
            if remaining > 3:  # Au moins assez pour "..."
                summary += line[:remaining-3] + "..."
            break
            
    return summary or "Contenu non analysable."

def store_file(user_id: int, file, original_filename: str) -> Dict:
    """
    Stocke un fichier téléchargé et l'enregistre en base de données
    """
    # Générer un nom de fichier sûr
    stored_filename = generate_safe_filename(original_filename)
    file_path = os.path.join(UPLOAD_FOLDER, stored_filename)
    
    # Sauvegarder le fichier
    file.save(file_path)
    
    # Obtenir la taille du fichier
    file_size = os.path.getsize(file_path)
    
    # Déterminer le type de fichier
    file_type = get_file_type(original_filename)
    
    # Extraire le texte
    extracted_text = extract_text_from_file(file_path)
    
    # Générer un résumé du contenu
    content_summary = generate_content_summary(extracted_text)
    
    # Calculer la date d'expiration
    expiry_date = (datetime.now() + timedelta(days=FILE_RETENTION_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    
    # Stocker les informations dans la base de données
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO uploaded_files 
    (user_id, original_filename, stored_filename, file_path, file_type, file_size, content_summary, extracted_text, expiry_date)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id,
        original_filename,
        stored_filename,
        file_path,
        file_type,
        file_size,
        content_summary,
        extracted_text,
        expiry_date
    ))
    
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    # Renvoyer les informations du fichier
    return {
        'id': file_id,
        'original_filename': original_filename,
        'file_type': file_type,
        'file_size': file_size,
        'content_summary': content_summary
    }

def get_file_info(file_id: int) -> Optional[Dict]:
    """
    Récupère les informations d'un fichier
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, user_id, original_filename, file_path, file_type, file_size, 
           content_summary, extracted_text, upload_date, expiry_date
    FROM uploaded_files 
    WHERE id = ?
    ''', (file_id,))
    
    file_info = cursor.fetchone()
    
    if file_info:
        # Mettre à jour la date de dernier accès
        cursor.execute('''
        UPDATE uploaded_files 
        SET last_accessed = CURRENT_TIMESTAMP
        WHERE id = ?
        ''', (file_id,))
        conn.commit()
        
        # Formater les informations
        info = {
            'id': file_info[0],
            'user_id': file_info[1],
            'original_filename': file_info[2],
            'file_path': file_info[3],
            'file_type': file_info[4],
            'file_size': file_info[5],
            'content_summary': file_info[6],
            'extracted_text': file_info[7],
            'upload_date': file_info[8],
            'expiry_date': file_info[9]
        }
        conn.close()
        return info
    
    conn.close()
    return None

def get_user_files(user_id: int) -> List[Dict]:
    """
    Récupère la liste des fichiers d'un utilisateur
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, original_filename, file_type, file_size, content_summary, upload_date
    FROM uploaded_files 
    WHERE user_id = ?
    ORDER BY upload_date DESC
    ''', (user_id,))
    
    files = cursor.fetchall()
    conn.close()
    
    file_list = []
    for file in files:
        file_list.append({
            'id': file[0],
            'filename': file[1],
            'file_type': file[2],
            'file_size': file[3],
            'content_summary': file[4],
            'upload_date': file[5]
        })
    
    return file_list

def cleanup_expired_files():
    """
    Nettoie les fichiers expirés
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Trouver les fichiers expirés
    cursor.execute('''
    SELECT id, file_path
    FROM uploaded_files 
    WHERE expiry_date < CURRENT_TIMESTAMP
    ''')
    
    expired_files = cursor.fetchall()
    
    for file_id, file_path in expired_files:
        # Supprimer le fichier
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Erreur lors de la suppression du fichier {file_path}: {e}")
        
        # Supprimer l'entrée de la base de données
        cursor.execute('''
        DELETE FROM uploaded_files 
        WHERE id = ?
        ''', (file_id,))
    
    conn.commit()
    conn.close()

def get_file_extension_icon(file_type: str) -> str:
    """
    Renvoie l'icône Font Awesome correspondant au type de fichier
    """
    icons = {
        'PDF': 'fa-file-pdf',
        'Word': 'fa-file-word',
        'Text': 'fa-file-alt',
        'CSV': 'fa-file-csv',
        'Excel': 'fa-file-excel',
        'JSON': 'fa-file-code',
        'HTML': 'fa-file-code',
        'XML': 'fa-file-code',
        'Markdown': 'fa-file-alt',
        'Rich Text': 'fa-file-alt'
    }
    
    return icons.get(file_type, 'fa-file')

def format_file_size(size_bytes: int) -> str:
    """
    Formate la taille d'un fichier en KB, MB, etc.
    """
    if size_bytes < 1024:
        return f"{size_bytes} octets"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} Ko"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} Mo"
