"""
Module de conscience temporelle pour Gemini
Ce module permet à l'IA de répondre intelligemment aux requêtes liées au temps,
sans introduire des références temporelles dans chaque réponse.
"""

import logging
import re
from typing import Dict, Any, Optional

from time_engine import get_current_datetime, format_datetime, is_time_request

# Métadonnées du module
MODULE_METADATA = {
    "enabled": True,
    "priority": 40,  # Priorité moyenne
    "description": "Gère la conscience temporelle et les références à l'heure/date",
    "version": "1.0.0",
    "dependencies": [],
    "hooks": ["process_request", "process_response"]
}

# Configuration du logger
logger = logging.getLogger(__name__)

def generate_time_response(timezone: str = "Europe/Paris") -> str:
    """
    Génère une réponse concernant l'heure et la date actuelles
    
    Args:
        timezone: Le fuseau horaire à utiliser
        
    Returns:
        Une réponse contenant l'heure et la date
    """
    current_time = get_current_datetime(timezone)
    time_str = format_datetime(current_time, "heure")
    date_str = format_datetime(current_time, "complet")
    
    return f"Il est actuellement {time_str} le {date_str} dans le fuseau horaire {timezone}."

def extract_timezone(text: str, default_timezone: str = "Europe/Paris") -> str:
    """
    Extrait le fuseau horaire mentionné dans le texte, si présent
    
    Args:
        text: Le texte à analyser
        default_timezone: Le fuseau horaire par défaut
        
    Returns:
        Le fuseau horaire extrait ou celui par défaut
    """
    # Liste de fuseaux horaires courants et leurs synonymes
    timezone_patterns = {
        "Europe/Paris": ["france", "paris", "français", "française"],
        "America/New_York": ["new york", "états-unis", "etats-unis", "usa", "amérique"],
        "Asia/Tokyo": ["japon", "tokyo"],
        "Europe/London": ["londres", "angleterre", "royaume-uni", "uk"],
        # Ajouter d'autres fuseaux horaires selon les besoins
    }
    
    text_lower = text.lower()
    
    # Chercher des mentions de fuseaux horaires
    for timezone, patterns in timezone_patterns.items():
        if any(pattern in text_lower for pattern in patterns):
            return timezone
    
    return default_timezone

def process(data: Dict[str, Any], hook: str) -> Dict[str, Any]:
    """
    Traite les requêtes et réponses pour gérer les références temporelles
    
    Args:
        data: Les données à traiter
        hook: Le hook appelé (process_request ou process_response)
        
    Returns:
        Les données modifiées
    """
    try:
        # Traitement des requêtes
        if hook == "process_request" and "text" in data:
            user_input = data["text"]
            
            # Vérifier si c'est une demande de temps (et seulement une demande explicite)
            if is_time_request(user_input):
                # Marquer cette requête comme une demande de temps
                data["is_time_request"] = True
                
                # Extraire le fuseau horaire si mentionné
                timezone = extract_timezone(user_input)
                data["requested_timezone"] = timezone
                
                logger.info(f"Demande de temps détectée, fuseau horaire: {timezone}")
            else:
                # S'assurer qu'on ne traite pas cela comme une demande de temps
                data["is_time_request"] = False
            
            return data
        
        # Traitement des réponses
        elif hook == "process_response":
            # Uniquement si c'était une demande explicite de temps, générer une réponse appropriée
            if data.get("is_time_request", False) and "text" in data:
                timezone = data.get("requested_timezone", "Europe/Paris")
                time_response = generate_time_response(timezone)
                
                # Remplacer complètement la réponse pour les demandes directes d'heure/date
                data["text"] = time_response
                logger.info(f"Réponse temporelle générée pour le fuseau horaire: {timezone}")
            
            return data
        
        return data
    
    except Exception as e:
        logger.error(f"Erreur dans le module de conscience temporelle: {str(e)}")
        return data
