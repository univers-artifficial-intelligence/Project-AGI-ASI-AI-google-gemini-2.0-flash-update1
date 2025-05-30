"""
Module de conscience temporelle autonome pour l'IA
Ce module permet à l'IA de maintenir une awareness constante du temps
et d'accéder automatiquement aux informations temporelles.
"""

import datetime
import pytz
import logging
from typing import Dict, Any, Optional

# Configuration du logging
logger = logging.getLogger(__name__)

class AutonomousTimeAwareness:
    """
    Classe pour gérer la conscience temporelle autonome de l'IA.
    """
    
    def __init__(self, default_timezone: str = "Europe/Paris"):
        """
        Initialise le système de conscience temporelle autonome.
        
        Args:
            default_timezone: Fuseau horaire par défaut
        """
        self.default_timezone = default_timezone
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def get_current_awareness(self) -> Dict[str, Any]:
        """
        Obtient la conscience temporelle actuelle complète.
        
        Returns:
            Dictionnaire contenant toutes les informations temporelles
        """
        try:
            # Obtenir l'heure actuelle
            current_dt = datetime.datetime.now(pytz.timezone(self.default_timezone))
            
            # Construire la conscience temporelle complète
            awareness = {
                "moment_actuel": {
                    "heure": current_dt.strftime("%H:%M:%S"),
                    "date": current_dt.strftime("%A %d %B %Y"),
                    "timestamp": current_dt.timestamp(),
                    "iso_format": current_dt.isoformat()
                },
                "contexte_temporel": {
                    "jour_semaine": current_dt.strftime("%A"),
                    "jour_mois": current_dt.day,
                    "mois": current_dt.strftime("%B"),
                    "annee": current_dt.year,
                    "fuseau_horaire": self.default_timezone
                },
                "conscience_narrative": self._generate_temporal_narrative(current_dt),
                "meta_conscience": {
                    "type": "conscience_temporelle_autonome",
                    "version": "1.0.0",
                    "derniere_mise_a_jour": current_dt.isoformat()
                }
            }
            
            return awareness
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération de la conscience temporelle: {str(e)}")
            return self._get_fallback_awareness()
    
    def _generate_temporal_narrative(self, dt: datetime.datetime) -> str:
        """
        Génère une description narrative du moment actuel.
        
        Args:
            dt: L'objet datetime actuel
            
        Returns:
            Description narrative du temps
        """
        try:
            hour = dt.hour
            day_name = dt.strftime("%A")
            date_str = dt.strftime("%d %B %Y")
            time_str = dt.strftime("%H:%M")
            
            # Déterminer la période de la journée
            if 5 <= hour < 12:
                period = "matinée"
            elif 12 <= hour < 17:
                period = "après-midi"
            elif 17 <= hour < 21:
                period = "soirée"
            else:
                period = "nuit"
            
            narrative = f"Nous sommes {day_name} {date_str}, il est {time_str} en {period}."
            
            return narrative
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération narrative: {str(e)}")
            return "Conscience temporelle en cours de récupération."
    
    def _get_fallback_awareness(self) -> Dict[str, Any]:
        """
        Retourne une conscience temporelle de secours en cas d'erreur.
        
        Returns:
            Conscience temporelle basique
        """
        return {
            "moment_actuel": {
                "heure": "Indéterminé",
                "date": "Indéterminé",
                "timestamp": 0,
                "iso_format": "Indéterminé"
            },
            "contexte_temporel": {
                "jour_semaine": "Indéterminé",
                "jour_mois": 0,
                "mois": "Indéterminé",
                "annee": 0,
                "fuseau_horaire": self.default_timezone
            },
            "conscience_narrative": "Ma conscience temporelle est temporairement indisponible.",
            "meta_conscience": {
                "type": "conscience_temporelle_autonome_fallback",
                "version": "1.0.0",
                "derniere_mise_a_jour": "Erreur"
            }
        }
    
    def get_temporal_context_for_ai(self) -> str:
        """
        Obtient un contexte temporel formaté pour l'IA.
        
        Returns:
            Contexte temporel sous forme de chaîne
        """
        awareness = self.get_current_awareness()
        narrative = awareness["conscience_narrative"]
        
        return f"[Conscience temporelle autonome] {narrative}"

# Instance globale pour l'utilisation par l'IA
autonomous_time = AutonomousTimeAwareness()

def get_ai_temporal_context() -> str:
    """
    Fonction utilitaire pour obtenir le contexte temporel pour l'IA.
    
    Returns:
        Contexte temporel formaté
    """
    return autonomous_time.get_temporal_context_for_ai()

def get_full_temporal_awareness() -> Dict[str, Any]:
    """
    Fonction utilitaire pour obtenir la conscience temporelle complète.
    
    Returns:
        Conscience temporelle complète
    """
    return autonomous_time.get_current_awareness()
