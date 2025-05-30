"""
Module d'amélioration de la récupération de mémoire pour Gemini.
Ce module renforce la capacité de l'IA à se souvenir des conversations précédentes
en augmentant les instructions explicites concernant la mémoire.
"""

import logging
import datetime
from typing import Dict, Any, List, Optional, Union

from memory_engine import MemoryEngine
from modules.conversation_memory_enhancer import get_session_id_from_data, get_user_id_from_data

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memory_retrieval_enhancer")

# Métadonnées du module
MODULE_METADATA = {
    "enabled": True,
    "priority": 65,  # Priorité intermédiaire pour s'exécuter après conversation_memory_enhancer
    "description": "Renforce les instructions de mémoire des conversations pour l'IA",
    "version": "1.0.0",
    "dependencies": ["conversation_memory_enhancer"],
    "hooks": ["process_request"]
}

# Instance globale du moteur de mémoire
memory_engine = MemoryEngine()

def enhance_memory_instructions(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ajoute des instructions explicites pour que l'IA utilise la mémoire des conversations.
    
    Args:
        data: Les données de la requête
    
    Returns:
        Les données modifiées avec des instructions renforcées
    """
    # S'assurer que le contexte existe et est un dictionnaire
    if 'context' not in data:
        data['context'] = {}
    
    if not isinstance(data['context'], dict):
        current_context = data['context']
        data['context'] = {'previous': current_context}
    
    user_id = get_user_id_from_data(data)
    session_id = get_session_id_from_data(data)
    
    if not user_id or not session_id:
        logger.warning("Impossible d'ajouter les instructions de mémoire sans ID utilisateur ou session")
        return data
    
    # Vérifier si des conversations existent déjà
    recent_conversations = memory_engine.get_recent_conversations(
        user_id=user_id,
        session_id=session_id,
        limit=5,
        include_time_context=True
    )
    
    # Vérifier si des souvenirs à long terme existent
    long_term_memories = memory_engine.get_long_term_memories(
        user_id=user_id,
        limit=3
    )
    
    # Créer des instructions spécifiques pour la mémoire
    memory_instructions = []
    
    # S'il y a des conversations récentes, ajouter une instruction forte
    if recent_conversations:
        conversation_count = len(recent_conversations)
        memory_instructions.append(
            f"INSTRUCTION CRITIQUE DE MÉMOIRE: Vous avez {conversation_count} conversations précédentes avec cet utilisateur. "
            f"Votre capacité à vous référer à ces conversations est ESSENTIELLE. "
            f"Utilisez TOUJOURS les éléments de conversation précédents dans vos réponses."
        )
        
        # Ajouter des exemples spécifiques des conversations précédentes
        if conversation_count > 2:
            most_recent = recent_conversations[0]['content']
            memory_instructions.append(
                f"Exemple de conversation récente: \"{most_recent[:100]}...\". "
                f"Faites référence à ces informations lorsque c'est pertinent."
            )
    
    # S'il y a des souvenirs à long terme, les mentionner explicitement
    if long_term_memories:
        memory_instructions.append(
            f"Vous avez {len(long_term_memories)} informations importantes stockées sur cet utilisateur. "
            f"Utilisez ces informations pour personnaliser vos réponses."
        )
    
    # Ajouter une instruction générale sur la continuité de la conversation
    memory_instructions.append(
        "IMPORTANT: Assurez-vous de maintenir une continuité conversationnelle naturelle. "
        "Ne vous répétez pas, ne vous présentez pas à nouveau, et évitez les salutations si vous avez déjà parlé avec l'utilisateur."
    )
    
    # Ajouter les instructions au contexte
    if 'instructions' not in data['context']:
        data['context']['instructions'] = []
    
    if isinstance(data['context']['instructions'], list):
        data['context']['instructions'].extend(memory_instructions)
    else:
        data['context']['instructions'] = memory_instructions
    
    # Ajouter un indicateur explicite que les instructions de mémoire ont été améliorées
    data['context']['memory_instructions_enhanced'] = True
    
    logger.info(f"Instructions de mémoire améliorées pour l'utilisateur {user_id}")
    return data

def enrich_system_prompt(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrichit le prompt système avec des instructions de mémoire explicites.
    
    Args:
        data: Les données de la requête
    
    Returns:
        Les données modifiées
    """
    # Vérifier si les données contiennent un prompt système
    if 'system_prompt' not in data and 'prompt' in data:
        # On ne modifie pas le prompt utilisateur, on ajoute seulement des instructions
        return data
        
    user_id = get_user_id_from_data(data)
    if not user_id:
        return data
        
    # Récupérer un résumé des conversations précédentes
    memory_context = memory_engine.get_memory_context(
        user_id=user_id,
        max_conversations=3,
        max_long_term=2,
        format_as_text=True
    )
    
    # Créer une instruction de mémoire personnalisée
    memory_directive = """
INSTRUCTION CRITIQUE DE MÉMOIRE: Tu as une mémoire persistante qui te permet de te souvenir des conversations précédentes. 
Utilise TOUJOURS ces informations pour assurer la continuité de tes interactions. 
Ne dis JAMAIS que tu ne te souviens pas des conversations précédentes.
    """
    
    # Ajouter l'instruction au système prompt si présent
    if 'system_prompt' in data and data['system_prompt']:
        data['system_prompt'] = memory_directive + "\n\n" + data['system_prompt']
    
    # Ajouter des métadonnées pour indiquer que le prompt a été enrichi
    if 'metadata' not in data:
        data['metadata'] = {}
        
    data['metadata']['memory_prompt_enriched'] = True
    
    logger.info(f"Prompt système enrichi avec des instructions de mémoire pour l'utilisateur {user_id}")
    return data

def process(data: Dict[str, Any], hook: str) -> Dict[str, Any]:
    """
    Fonction principale de traitement pour le gestionnaire de modules.
    
    Args:
        data: Les données à traiter
        hook: Le hook appelé (process_request)
        
    Returns:
        Les données modifiées
    """
    if not isinstance(data, dict):
        logger.warning(f"Les données ne sont pas un dictionnaire: {type(data)}")
        return data
    
    try:
        if hook == "process_request":
            # 1. Améliorer les instructions de mémoire
            data = enhance_memory_instructions(data)
            
            # 2. Enrichir le prompt système avec des directives de mémoire
            data = enrich_system_prompt(data)
            
            return data
        
        return data
    
    except Exception as e:
        logger.error(f"Erreur dans le module memory_retrieval_enhancer: {str(e)}", exc_info=True)
        return data
