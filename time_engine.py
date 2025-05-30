"""
Module de gestion du temps pour Gemini
Ce module permet à l'IA d'accéder à l'heure et à la date en temps réel,
et de comprendre les concepts temporels.
"""

import datetime
import pytz
import re
from typing import Dict, Any, Optional, List, Tuple

# Configuration par défaut
DEFAULT_TIMEZONE = 'Europe/Paris'
AVAILABLE_TIMEZONES = pytz.common_timezones

# Constantes pour la rétention de mémoire
MEMORY_RETENTION = {
    'SHORT_TERM': datetime.timedelta(minutes=30),
    'MEDIUM_TERM': datetime.timedelta(days=1),
    'LONG_TERM': datetime.timedelta(days=7)
}

# Cache pour les fuseaux horaires fréquemment utilisés
timezone_cache = {}

def get_current_datetime(timezone_str: str = DEFAULT_TIMEZONE) -> datetime.datetime:
    """
    Obtient l'heure et la date actuelles dans le fuseau horaire spécifié
    
    Args:
        timezone_str: Le fuseau horaire (par défaut: Europe/Paris)
        
    Returns:
        L'objet datetime actuel dans le fuseau horaire spécifié
    """
    # Utiliser le cache si disponible
    if timezone_str in timezone_cache:
        tz = timezone_cache[timezone_str]
    else:
        # Vérifier si le fuseau horaire est valide
        if timezone_str not in AVAILABLE_TIMEZONES:
            # Si non valide, utiliser le fuseau horaire par défaut
            timezone_str = DEFAULT_TIMEZONE
        
        # Obtenir l'objet timezone et le mettre en cache
        tz = pytz.timezone(timezone_str)
        timezone_cache[timezone_str] = tz
    
    # Obtenir le datetime actuel avec le fuseau horaire
    return datetime.datetime.now(tz)

def format_datetime(dt: datetime.datetime, format_str: str = "complet") -> str:
    """
    Formate un objet datetime en chaîne de caractères lisible
    
    Args:
        dt: L'objet datetime à formater
        format_str: Le format souhaité ("complet", "date", "heure", "court")
        
    Returns:
        La chaîne formatée
    """
    if format_str == "complet":
        return dt.strftime("%A %d %B %Y, %H:%M:%S")
    elif format_str == "date":
        return dt.strftime("%d/%m/%Y")
    elif format_str == "heure":
        return dt.strftime("%H:%M")
    elif format_str == "court":
        return dt.strftime("%d/%m/%Y %H:%M")
    else:
        return dt.strftime(format_str)

def is_time_request(text: str) -> bool:
    """
    Détecte si un texte contient une demande explicite d'heure ou de date
    
    Args:
        text: Le texte à analyser
        
    Returns:
        True si le texte demande l'heure ou la date, False sinon
    """
    text_lower = text.lower()
    
    # Patterns pour détecter les demandes d'heure et de date
    time_patterns = [
        r"quelle\s+(?:heure|est-il)",
        r"l[''']heure\s+actuelle",
        r"heure\s+(?:est-il|actuelle)",
        r"l'heure\s+s'il\s+(?:te|vous)\s+pla[iî]t"
    ]
    
    date_patterns = [
        r"quelle\s+(?:date|jour)",
        r"(?:date|jour)\s+(?:sommes-nous|est-on)",
        r"(?:date|jour)\s+d'aujourd'hui",
        r"quel\s+jour\s+(?:sommes-nous|est-on)"
    ]
    
    # Combiner tous les patterns
    all_patterns = time_patterns + date_patterns
    
    # Vérifier si l'un des patterns correspond
    for pattern in all_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False

def should_remember_conversation(creation_time: datetime.datetime, current_time: Optional[datetime.datetime] = None) -> bool:
    """
    Détermine si une conversation devrait être mémorisée en fonction de son ancienneté
    
    Args:
        creation_time: Le moment où la conversation a été créée
        current_time: Le moment actuel (par défaut: maintenant)
        
    Returns:
        True si la conversation devrait être mémorisée, False sinon
    """
    if current_time is None:
        current_time = datetime.datetime.now(creation_time.tzinfo)
    
    # Calculer la différence de temps
    time_diff = current_time - creation_time
    
    # Si moins de 7 jours, mémoriser
    return time_diff.days < 7

def timestamp_to_readable_time_diff(timestamp: str) -> str:
    """
    Convertit un timestamp en une différence de temps lisible
    
    Args:
        timestamp: Le timestamp au format ISO
        
    Returns:
        Une chaîne décrivant la différence de temps (ex: "il y a 2 jours")
    """
    try:
        # Convertir le timestamp en objet datetime
        dt = datetime.datetime.fromisoformat(timestamp)
        
        # Obtenir le moment actuel avec le même fuseau horaire
        now = datetime.datetime.now(dt.tzinfo)
        
        # Calculer la différence
        diff = now - dt
        
        # Convertir en formulation lisible
        if diff.days > 365:
            years = diff.days // 365
            return f"il y a {years} an{'s' if years > 1 else ''}"
        elif diff.days > 30:
            months = diff.days // 30
            return f"il y a {months} mois"
        elif diff.days > 0:
            return f"il y a {diff.days} jour{'s' if diff.days > 1 else ''}"
        elif diff.seconds >= 3600:
            hours = diff.seconds // 3600
            return f"il y a {hours} heure{'s' if hours > 1 else ''}"
        elif diff.seconds >= 60:
            minutes = diff.seconds // 60
            return f"il y a {minutes} minute{'s' if minutes > 1 else ''}"
        else:
            return "à l'instant"
    except (ValueError, TypeError):
        return "à un moment inconnu"
