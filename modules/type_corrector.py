"""
Module de sécurité qui intercepte et corrige les données de type incorrect.
Réécrit complètement pour offrir une protection maximale contre les erreurs de type.
"""

import logging
import traceback
from typing import Any, Dict, Union

# Métadonnées du module
MODULE_METADATA = {
    "enabled": True,
    "priority": 1,  # Priorité MAXIMALE pour s'exécuter avant tous les autres modules
    "description": "Module de sécurité qui protège contre les erreurs de type",
    "version": "0.2.0",
    "dependencies": [],
    "hooks": ["process_request", "process_response"]
}

# Configuration du logger spécifique avec plus de détails
logger = logging.getLogger('type_security')
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    # Ajouter un handler pour fichier avec plus de détails
    file_handler = logging.FileHandler('type_security.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Ajouter un handler console pour les erreurs critiques
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

def safe_copy(obj: Any) -> Any:
    """
    Crée une copie sûre d'un objet en fonction de son type.
    """
    try:
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, dict):
            return {k: safe_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [safe_copy(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(safe_copy(item) for item in obj)
        else:
            # Pour les types non pris en charge, tenter de les convertir en chaîne
            return str(obj)
    except Exception as e:
        logger.error(f"Erreur lors de la copie: {str(e)}")
        return None

def process(data: Any, hook: str) -> Dict[str, Any]:
    """
    Fonction de sécurité qui garantit que les données sont toujours dans un format valide.
    """
    try:
        logger.debug(f"[{hook}] Entrée: type={type(data)}")
        
        # 1. Protection contre les données non-dictionnaire
        if not isinstance(data, dict):
            logger.warning(f"[{hook}] Données non-dictionnaire détectées: {type(data)}")
            
            if isinstance(data, str):
                logger.info(f"[{hook}] Conversion d'une chaîne en dictionnaire")
                return {"text": data, "_secured": True}
            else:
                try:
                    # Tenter de convertir en dictionnaire si possible
                    dict_data = dict(data) if hasattr(data, "__iter__") else {"value": str(data)}
                    dict_data["_secured"] = True
                    return dict_data
                except:
                    # En cas d'échec, créer un nouveau dictionnaire
                    return {"value": str(data) if data is not None else "", "_secured": True}
        
        # 2. Créer une copie sécurisée du dictionnaire
        result = {}
        for key, value in data.items():
            # Protection contre les clés non-str
            safe_key = str(key) if not isinstance(key, str) else key
            
            # Traitement spécial de la clé "text"
            if safe_key == "text":
                if not isinstance(value, str):
                    logger.warning(f"[{hook}] Clé 'text' contient un type incorrect: {type(value)}")
                    result["text"] = str(value) if value is not None else ""
                    result["_text_corrected"] = True
                else:
                    result["text"] = value
            else:
                # Copie sécurisée des autres valeurs
                result[safe_key] = safe_copy(value)
        
        # 3. Vérification de la présence de la clé "text" si nécessaire dans le hook de réponse
        if hook == "process_response" and "text" not in result:
            logger.warning(f"[{hook}] Clé 'text' manquante dans les données de réponse")
            result["text"] = data.get("value", "") if isinstance(data, dict) else ""
            result["_text_added"] = True
        
        # 4. Ajouter un marqueur de sécurité
        result["_secured"] = True
        
        return result
        
    except Exception as e:
        # En cas d'erreur critique, retourner un dictionnaire minimal
        logger.critical(f"Erreur critique dans le module de sécurité: {str(e)}")
        logger.critical(traceback.format_exc())
        return {"text": "Désolé, une erreur est survenue.", "_error": str(e), "_secured": True}
