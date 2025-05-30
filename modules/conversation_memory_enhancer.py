"""
Module d'amélioration de la mémoire des conversations pour Gemini.
Ce module améliore la capacité de l'IA à se souvenir des conversations précédentes
en utilisant un système de mémoire temporel avancé.
"""

import logging
import datetime
import json
from typing import Dict, Any, List, Optional

from time_engine import should_remember_conversation, timestamp_to_readable_time_diff
from memory_engine import MemoryEngine

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("conversation_memory_enhancer")

# Métadonnées du module
MODULE_METADATA = {
    "enabled": True,
    "priority": 60,  # Priorité élevée pour être exécuté parmi les premiers
    "description": "Améliore la mémoire des conversations de l'IA",
    "version": "1.0.0",
    "dependencies": [],
    "hooks": ["process_request", "process_response"]
}

# Instance globale du moteur de mémoire
memory_engine = MemoryEngine()

def get_user_id_from_data(data: Dict[str, Any]) -> Optional[int]:
    """
    Extrait l'ID utilisateur des données.
    """
    # Essayer différentes clés possibles
    for key in ['user_id', 'userId', 'user']:
        if key in data and data[key]:
            try:
                return int(data[key])
            except (ValueError, TypeError):
                pass
    
    # Chercher dans la session si disponible
    if 'session' in data and isinstance(data['session'], dict):
        for key in ['user_id', 'userId', 'user']:
            if key in data['session'] and data['session'][key]:
                try:
                    return int(data['session'][key])
                except (ValueError, TypeError):
                    pass
    
    logger.warning("Impossible de déterminer l'ID utilisateur dans les données")
    return None

def get_session_id_from_data(data: Dict[str, Any]) -> Optional[str]:
    """
    Extrait l'ID de session des données.
    """
    # Essayer différentes clés possibles
    for key in ['session_id', 'sessionId', 'session']:
        if key in data and isinstance(data[key], (str, int)):
            return str(data[key])
    
    # Chercher dans la session si disponible
    if 'session' in data and isinstance(data['session'], dict):
        for key in ['id', 'session_id', 'sessionId']:
            if key in data['session'] and data['session'][key]:
                return str(data['session'][key])
    
    logger.warning("Impossible de déterminer l'ID de session dans les données")
    return None

def extract_message_content(data: Dict[str, Any]) -> Optional[str]:
    """
    Extrait le contenu du message des données.
    """
    # Pour une requête utilisateur
    if 'text' in data:
        return data['text']
    if 'message' in data:
        if isinstance(data['message'], str):
            return data['message']
        elif isinstance(data['message'], dict) and 'content' in data['message']:
            return data['message']['content']
    
    # Pour une réponse Gemini
    if 'response' in data:
        return data['response']
    if 'content' in data:
        return data['content']
    
    logger.warning("Impossible d'extraire le contenu du message des données")
    return None

def add_conversation_context(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ajoute le contexte des conversations précédentes aux données de la requête.
    """
    user_id = get_user_id_from_data(data)
    session_id = get_session_id_from_data(data)
    
    if not user_id:
        logger.warning("Impossible d'ajouter le contexte de conversation sans ID utilisateur")
        return data
    
    # Vérifier s'il y a des messages récents dans cette session
    recent_conversations = memory_engine.get_recent_conversations(
        user_id=user_id,
        session_id=session_id,
        limit=3,
        include_time_context=True
    )
    
    # Déterminer si c'est une conversation en cours ou nouvelle
    is_ongoing_conversation = len(recent_conversations) > 0
    
    # Générer le contexte mémoire
    memory_context = memory_engine.get_memory_context(
        user_id=user_id,
        session_id=session_id,
        max_conversations=5,
        max_long_term=3,
        format_as_text=True
    )
    
    # Ajouter le contexte à la requête
    if 'context' not in data:
        data['context'] = {}
    
    if isinstance(data['context'], dict):
        data['context']['conversation_memory'] = memory_context
        data['context']['is_ongoing_conversation'] = is_ongoing_conversation
        
        # Si c'est une conversation en cours, ajouter un indicateur explicite
        # pour éviter que l'IA ne recommence avec des salutations
        if is_ongoing_conversation:
            last_conversation_time = recent_conversations[0].get('time_ago', 'récemment')
            
            if 'instructions' not in data['context']:
                data['context']['instructions'] = []
            
            if isinstance(data['context']['instructions'], list):
                data['context']['instructions'].append(
                    f"Cette conversation est en cours. Vous avez déjà échangé avec cet utilisateur {last_conversation_time}. "
                    f"Évitez de vous présenter à nouveau ou de répéter des salutations comme 'Je suis ravi de vous rencontrer'. "
                    f"Continuez simplement la conversation naturellement."
                )
    else:
        # Si context est une chaîne ou un autre type, la convertir en dictionnaire
        current_context = data['context']
        data['context'] = {
            'previous': current_context,
            'conversation_memory': memory_context,
            'is_ongoing_conversation': is_ongoing_conversation
        }
    
    logger.info(f"Contexte de conversation ajouté pour l'utilisateur {user_id}")
    return data

def store_conversation_entry(data: Dict[str, Any], is_response: bool = False) -> Dict[str, Any]:
    """
    Stocke l'entrée de conversation dans la mémoire.
    """
    user_id = get_user_id_from_data(data)
    session_id = get_session_id_from_data(data)
    content = extract_message_content(data)
    
    if not user_id or not session_id or not content:
        logger.warning("Données insuffisantes pour stocker l'entrée de conversation")
        return data
    
    # Préparer les métadonnées
    metadata = {
        'is_response': is_response,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    # Ajouter des métadonnées supplémentaires si disponibles
    if 'emotional_state' in data:
        metadata['emotional_state'] = data['emotional_state']
    
    # Déterminer le niveau de mémoire (court, moyen ou long terme)
    memory_level = "MEDIUM_TERM"  # Par défaut, mémoriser pour environ 1 heure
    
    # Si c'est marqué comme important ou une réponse émotionnelle, stocker plus longtemps
    if is_response and 'emotional_state' in data:
        emotional_state = data['emotional_state']
        if isinstance(emotional_state, dict) and 'intensity' in emotional_state:
            intensity = emotional_state.get('intensity', 0)
            if intensity > 7:  # Haute intensité émotionnelle
                memory_level = "LONG_TERM"
    
    # Stocker dans la mémoire
    memory_id = memory_engine.store_conversation(
        session_id=session_id,
        user_id=user_id,
        content=content,
        memory_level=memory_level,
        metadata=metadata
    )
    
    logger.info(f"Conversation stockée avec ID {memory_id}, niveau {memory_level}")
    
    # Ajouter l'ID mémoire aux données pour référence
    if 'memory' not in data:
        data['memory'] = {}
    
    data['memory']['memory_id'] = memory_id
    data['memory']['memory_level'] = memory_level
    
    return data

def process(data: Dict[str, Any], hook: str) -> Dict[str, Any]:
    """
    Fonction principale de traitement pour le gestionnaire de modules.
    
    Args:
        data: Les données à traiter
        hook: Le hook appelé (process_request ou process_response)
        
    Returns:
        Les données modifiées
    """
    if not isinstance(data, dict):
        logger.warning(f"Les données ne sont pas un dictionnaire: {type(data)}")
        return data
    
    try:
        if hook == "process_request":
            # 1. Stocker la requête utilisateur dans la mémoire
            data = store_conversation_entry(data, is_response=False)
            
            # 2. Ajouter le contexte des conversations précédentes
            data = add_conversation_context(data)
            
            return data
            
        elif hook == "process_response":
            # Stocker la réponse de Gemini dans la mémoire
            data = store_conversation_entry(data, is_response=True)
            
            return data
        
        return data
    
    except Exception as e:
        logger.error(f"Erreur dans le module conversation_memory_enhancer: {str(e)}", exc_info=True)
        return data
