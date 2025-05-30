"""
Implémentation de l'interface AIApiInterface pour un LLM personnalisé.
Ce module permet d'intégrer un LLM personnalisé défini par l'utilisateur via une API.
"""
import requests
import json
import logging
import os
import pytz
import datetime
import re
from typing import Dict, List, Any, Optional, Union

from ai_api_interface import AIApiInterface
from modules.text_memory_manager import TextMemoryManager

# Import du module de conscience temporelle autonome
try:
    from autonomous_time_awareness import get_ai_temporal_context
except ImportError:
    def get_ai_temporal_context():
        return "[Conscience temporelle] Système en cours d'initialisation."
    logging.getLogger(__name__).warning("Module autonomous_time_awareness non trouvé, utilisation de la fonction de secours")

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import de notre module de formatage de texte
try:
    from response_formatter import format_response
except ImportError:
    # Fonction de secours si le module n'est pas disponible
    def format_response(text):
        return text
    logger.warning("Module response_formatter non trouvé, utilisation de la fonction de secours")

class CustomLLMAPI(AIApiInterface):
    """Implémentation de l'interface AIApiInterface pour un LLM personnalisé défini par l'utilisateur"""
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """
        Initialise un LLM personnalisé avec une clé API et une URL d'API.
        
        Args:
            api_key: Clé API pour le LLM personnalisé
            api_url: URL de l'API du LLM personnalisé (obligatoire)
        """
        self.api_key = api_key
        self.api_url = api_url
        
        if not self.api_url:
            logger.warning("Aucune URL d'API personnalisée fournie, l'API ne fonctionnera pas correctement")
        else:
            logger.info("API LLM personnalisée initialisée avec l'URL: %s", self.api_url)
    
    def process_memory_request(self, prompt: str, user_id: int, session_id: str) -> Optional[str]:
        """
        Traite spécifiquement les demandes liées à la mémoire ou aux conversations passées.
        
        Args:
            prompt: La question ou instruction de l'utilisateur
            user_id: ID de l'utilisateur
            session_id: ID de la session actuelle
            
        Returns:
            Un contexte enrichi si la demande est liée à la mémoire, sinon None
        """
        # Mots clés qui indiquent une demande de mémoire
        memory_keywords = [
            "souviens", "rappelles", "mémoire", "précédemment", "auparavant",
            "conversation précédente", "parlé de", "sujet précédent", "discuté de",
            "déjà dit", "dernière fois", "avant"
        ]
        
        # Vérifier si la demande concerne la mémoire
        is_memory_request = any(keyword in prompt.lower() for keyword in memory_keywords)
        
        if not is_memory_request:
            return None
        
        try:
            # Utiliser le TextMemoryManager pour rechercher du contenu pertinent
            text_memory = TextMemoryManager()
            memory_results = text_memory.search_memory(user_id, prompt, session_id=session_id)
            
            if memory_results and len(memory_results) > 0:
                memory_context = "Voici ce dont nous avons discuté précédemment qui pourrait être pertinent:\n\n"
                
                for memory in memory_results:
                    message_content = memory.get('content', '').strip()
                    if message_content:
                        memory_context += f"- {message_content}\n\n"
                
                return memory_context
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la demande de mémoire: {str(e)}")
        
        return None
    
    def get_response(self, 
                    prompt: str, 
                    image_data: Optional[str] = None,
                    context: Optional[str] = None,
                    emotional_state: Optional[Dict[str, Any]] = None,
                    user_id: int = 1,
                    session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtient une réponse du LLM personnalisé.
        
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
        if not self.api_url:
            return {
                "success": False,
                "response": "Erreur: URL API du LLM personnalisé non configurée. Veuillez configurer l'URL dans les paramètres d'API.",
                "error": "URL API non configurée"
            }
        
        try:
            # Préparation du contexte temporel automatique
            time_context = get_ai_temporal_context()
            
            # Ajout du contexte émotionnel s'il est fourni
            emotional_context = ""
            if emotional_state and "base_state" in emotional_state:
                emotional_context = f"[État émotionnel: {emotional_state['base_state']}] "
            
            # Assemblage du prompt avec les contextes
            enhanced_prompt = prompt
            if context:
                enhanced_prompt = context + "\n\n" + prompt
            
            # Ajout du contexte temporel et émotionnel
            if time_context or emotional_context:
                enhanced_prompt = f"{emotional_context}{time_context}\n\n{enhanced_prompt}"
            
            # Préparation des données pour l'appel API
            headers = {
                "Content-Type": "application/json"
            }
            
            # Ajouter la clé API aux en-têtes si disponible
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Construction du payload selon le format général
            payload = {
                "prompt": enhanced_prompt,
                "user_id": user_id
            }
            
            # Ajouter l'image si présente
            if image_data:
                payload["image"] = image_data
            
            # Appel API
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=120  # 2 minutes de timeout
            )
            
            response.raise_for_status()  # Lever une exception si réponse non 2xx
            
            # Traiter la réponse
            response_data = response.json()
            
            # Obtenir le texte de la réponse
            if "response" in response_data:
                response_text = response_data["response"]
            else:
                # Essayer d'autres structures de réponse possibles
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    response_text = response_data["choices"][0].get("text", "")
                elif "text" in response_data:
                    response_text = response_data["text"]
                elif "output" in response_data:
                    response_text = response_data["output"]
                else:
                    response_text = json.dumps(response_data)
            
            # Formater la réponse pour la rendre plus lisible
            formatted_response = format_response(response_text)
            
            # Retourner une réponse formatée
            return {
                "success": True,
                "response": formatted_response,
                "emotional_state": emotional_state  # Retransmettre l'état émotionnel
            }
            
        except requests.exceptions.RequestException as e:
            error_message = f"Erreur lors de la communication avec l'API LLM personnalisée: {str(e)}"
            logger.error(error_message)
            return {
                "success": False,
                "response": f"Désolé, une erreur est survenue lors de la communication avec le LLM personnalisé. Veuillez vérifier vos paramètres d'API ou réessayer plus tard.\n\nErreur technique: {str(e)}",
                "error": error_message
            }
        except Exception as e:
            error_message = f"Erreur inattendue lors du traitement de la demande: {str(e)}"
            logger.error(error_message)
            return {
                "success": False,
                "response": "Désolé, une erreur inattendue s'est produite. Veuillez réessayer ou contacter le support.",
                "error": error_message
            }
