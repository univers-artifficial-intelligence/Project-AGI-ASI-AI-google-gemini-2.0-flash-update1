import os
import shutil
from direct_file_access import ensure_text_data_folder_exists, TEXT_DATA_FOLDER, get_all_text_files

def add_text_file(filename, content):
    """Ajoute un nouveau fichier texte au dossier de données"""
    ensure_text_data_folder_exists()
    file_path = os.path.join(TEXT_DATA_FOLDER, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"Fichier {filename} ajouté avec succès"

def delete_text_file(filename):
    """Supprime un fichier texte du dossier de données"""
    file_path = os.path.join(TEXT_DATA_FOLDER, filename)
    if os.path.exists(file_path) and filename.endswith('.txt'):
        os.remove(file_path)
        return f"Fichier {filename} supprimé avec succès"
    return f"Fichier {filename} non trouvé"

def get_folder_stats():
    """Obtient les statistiques du dossier de données textuelles"""
    files = get_all_text_files()
    total_size = 0
    for file in files:
        file_path = os.path.join(TEXT_DATA_FOLDER, file)
        total_size += os.path.getsize(file_path)
    
    return {
        "nombre_fichiers": len(files),
        "taille_totale_octets": total_size,
        "taille_totale_mb": total_size / (1024 * 1024),
        "liste_fichiers": files
    }

def backup_text_folder(backup_location):
    """Effectue une sauvegarde du dossier de données textuelles"""
    if not os.path.exists(backup_location):
        os.makedirs(backup_location)
    
    backup_folder = os.path.join(backup_location, 'gemini_text_data_backup')
    if os.path.exists(backup_folder):
        shutil.rmtree(backup_folder)
    
    shutil.copytree(TEXT_DATA_FOLDER, backup_folder)
    return f"Sauvegarde effectuée dans {backup_folder}"
