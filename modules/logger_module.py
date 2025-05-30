"""
Module de journalisation pour identifier les problèmes dans le système de modules.
Ce module enregistre les types de données entrantes et sortantes pour chaque module.
"""

import logging
import inspect
from typing import Any, Dict

# Métadonnées du module
MODULE_METADATA = {
    "enabled": True,
    "priority": 10,  # Priorité très élevée pour s'exécuter avant les autres modules
    "description": "Module de journalisation des types de données",
    "version": "0.1.0",
    "dependencies": [],
    "hooks": ["process_request", "process_response"]
}

# Configuration du logger spécifique
logger = logging.getLogger('module_debugger')
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    file_handler = logging.FileHandler('module_debug.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

def process(data: Dict[str, Any], hook: str) -> Dict[str, Any]:
    """
    Fonction principale qui journalise les types de données.
    """
    # Journaliser le type des données d'entrée
    logger.debug(f"[{hook}] Entrée: Type={type(data)}")
    
    # Pour les dictionnaires, journaliser le type de chaque valeur
    if isinstance(data, dict):
        for key, value in data.items():
            logger.debug(f"[{hook}] Clé={key}, Type={type(value)}")
            
            # Si la valeur est un dictionnaire, journaliser ses types également
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    logger.debug(f"[{hook}] Sous-clé={key}.{sub_key}, Type={sub_value}")
    
    # Obtenir le nom du module appelant (qui sera exécuté après celui-ci)
    caller_frame = inspect.currentframe().f_back
    if caller_frame:
        caller_module = caller_frame.f_globals.get('__name__')
        logger.debug(f"[{hook}] Prochain module: {caller_module}")
    
    # Ne pas modifier les données, juste les journaliser
    return data
