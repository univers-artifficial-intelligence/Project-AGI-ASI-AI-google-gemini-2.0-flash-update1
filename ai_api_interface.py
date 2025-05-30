"""
Interface abstraite pour les différentes APIs d'intelligence artificielle.
Ce module définit une interface commune pour toutes les APIs d'IA supportées.
"""
import logging
import abc
from typing import Dict, Any, Optional, List, Union

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIApiInterface(abc.ABC):
    """Interface abstraite que toutes les implémentations d'API d'IA doivent suivre."""
    
    @abc.abstractmethod
    def get_response(self, 
                    prompt: str, 
                    image_data: Optional[str] = None,
                    context: Optional[str] = None,
                    emotional_state: Optional[Dict[str, Any]] = None,
                    user_id: int = 1,
                    session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Méthode abstraite pour obtenir une réponse de l'API d'IA.
        
        Args:
            prompt: Le texte de la requête
            image_data: Données d'image encodées en base64 (optionnel)
            context: Contexte de conversation précédent (optionnel)
            emotional_state: État émotionnel actuel de l'IA (optionnel)
            user_id: ID de l'utilisateur
            session_id: ID de la session (optionnel)
        
        Returns:
            Dictionnaire contenant la réponse et les métadonnées
        """
        pass
    
    @abc.abstractmethod
    def process_memory_request(self, prompt: str, user_id: int, session_id: str) -> Optional[str]:
        """
        Méthode abstraite pour traiter les demandes liées à la mémoire.
        
        Args:
            prompt: La question ou instruction de l'utilisateur
            user_id: ID de l'utilisateur
            session_id: ID de la session actuelle
            
        Returns:
            Un contexte enrichi si la demande est liée à la mémoire, sinon None
        """
        pass
    
    @abc.abstractmethod
    def get_conversation_history(self, user_id: int, session_id: str, max_messages: int = 10) -> str:
        """
        Méthode abstraite pour récupérer l'historique de conversation.
        
        Args:
            user_id: ID de l'utilisateur
            session_id: ID de la session
            max_messages: Nombre maximal de messages à inclure
            
        Returns:
            Un résumé de la conversation précédente
        """
        pass
