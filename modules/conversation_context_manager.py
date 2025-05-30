"""
Module de gestion du contexte conversationnel pour Gemini.
Ce module améliore la continuité des conversations et équilibre les expressions émotionnelles.
"""

import logging
import re
import random
from typing import Dict, Any, List, Optional

from memory_engine import MemoryEngine

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("conversation_context_manager")

# Métadonnées du module
MODULE_METADATA = {
    "enabled": True,
    "priority": 70,  # Priorité élevée pour s'exécuter après la récupération du contexte
    "description": "Équilibre les expressions émotionnelles et améliore la continuité des conversations",
    "version": "1.0.0",
    "dependencies": [],
    "hooks": ["process_request", "process_response"]
}

# Instance globale du moteur de mémoire
memory_engine = MemoryEngine()

# Patterns pour détecter les salutations et expressions de reprise de conversation
GREETING_PATTERNS = [
    r"(?i)^(bonjour|salut|hello|coucou|hey|bonsoir)",
    r"(?i)^(je suis (ravi|content|heureux) de (vous|te) (voir|rencontrer|parler))",
    r"(?i)^(c'est un plaisir de vous (revoir|retrouver|parler))",
    r"(?i)^(comment (vas-tu|allez-vous|ça va))",
]

# Patterns pour détecter les expressions émotionnelles excessives
EXCESSIVE_EMOTION_PATTERNS = [
    r"(?i)(je suis (vraiment|extrêmement|incroyablement) (content|heureux|ravi|enthousiaste|excité))",
    r"(?i)(je ressens (beaucoup|énormément|tellement) d'(enthousiasme|excitation|joie))",
    r"(?i)(je ne peux pas (cacher|contenir) mon (enthousiasme|excitation|ravissement))",
    r"(?i)(je suis (totalement|complètement) (fasciné|émerveillé|captivé))",
]

# Patterns spécifiques pour les réponses d'analyse d'images
IMAGE_ANALYSIS_PATTERNS = [
    r"(?i)^(Absolument\s?!?\s?Je suis ravi de pouvoir t'aider\.?\s?Oui,?\s?je ressens des émotions en analysant cette image\s?Analyse de l'image)",
    r"(?i)^(Je suis (ravi|heureux|content) de pouvoir analyser cette image pour toi\.?\s?Analyse de l'image)",
    r"(?i)^(Analyse de l'image\s?:?\s?)"
]

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
    
    return None

def is_new_conversation(data: Dict[str, Any]) -> bool:
    """
    Détermine si la conversation est nouvelle ou en cours.
    
    Args:
        data: Les données de la requête
    
    Returns:
        True si c'est une nouvelle conversation, False sinon
    """
    user_id = get_user_id_from_data(data)
    session_id = get_session_id_from_data(data)
    
    if not user_id or not session_id:
        return True  # Par défaut, considérer comme une nouvelle conversation
    
    try:
        # Vérifier s'il y a des messages récents pour cette session
        recent_conversations = memory_engine.get_recent_conversations(
            user_id=user_id, 
            session_id=session_id, 
            limit=5
        )
        
        # Si pas de conversations récentes trouvées, c'est une nouvelle conversation
        return len(recent_conversations) == 0
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de l'état de la conversation: {str(e)}")
        return True  # En cas d'erreur, considérer comme une nouvelle conversation

def detect_image_analysis(response: str) -> bool:
    """
    Détecte si la réponse est une analyse d'image.
    
    Args:
        response: La réponse générée
    
    Returns:
        True si c'est une analyse d'image, False sinon
    """
    # Mots-clés génériques pour les analyses d'images
    image_keywords = [
        r"(?i)(cette image montre)",
        r"(?i)(dans cette image,)",
        r"(?i)(l'image présente)",
        r"(?i)(on peut voir sur cette image)",
        r"(?i)(je vois une image qui)",
        r"(?i)(la photo montre)",
        r"(?i)(on observe sur cette image)",
        r"(?i)(il s'agit d'une image (de|qui))",
        r"(?i)(cette photographie (montre|présente|contient))",
        r"(?i)(l'illustration (montre|représente))",
        r"(?i)(sur cette (capture|prise de vue))",
        r"(?i)(ce visuel (montre|présente))",
    ]
    
    # Mots-clés par catégories d'images
    category_keywords = {
        # Images astronomiques
        "astronomie": [
            r"(?i)(constellation[s]? (de|du|des))",
            r"(?i)(carte (du|céleste|du ciel))",
            r"(?i)(ciel nocturne)",
            r"(?i)(étoile[s]? (visible|brillante|nommée))",
            r"(?i)(position (de la|des) (lune|planète|étoile))",
            r"(?i)(trajectoire (de|des|du))",
        ],
        # Œuvres d'art et images créatives
        "art": [
            r"(?i)(tableau|peinture|œuvre d'art)",
            r"(?i)(style (artistique|pictural))",
            r"(?i)(composition (artistique|visuelle))",
            r"(?i)(perspective|arrière-plan|premier plan)",
            r"(?i)(couleurs|teintes|nuances|palette)",
        ],
        # Scènes naturelles et paysages
        "nature": [
            r"(?i)(paysage (de|montagneux|marin|rural|urbain))",
            r"(?i)(vue (panoramique|aérienne))",
            r"(?i)(environnement naturel)",
            r"(?i)(flore|faune|végétation)",
            r"(?i)(forêt|montagne|océan|rivière|lac)",
        ],
        # Schémas et diagrammes
        "technique": [
            r"(?i)(schéma|diagramme|graphique)",
            r"(?i)(représentation (technique|schématique))",
            r"(?i)(illustration technique)",
            r"(?i)(structure|composants|éléments)",
            r"(?i)(légende|annotation|étiquette)",
        ]
    }
    
    # Vérifier les mots-clés génériques
    for pattern in image_keywords:
        if re.search(pattern, response):
            return True
    
    # Vérifier les mots-clés par catégorie
    for category, patterns in category_keywords.items():
        for pattern in patterns:
            if re.search(pattern, response):
                return True
    
    # Ou si la réponse est déjà identifiée comme commençant par un pattern d'analyse d'image
    for pattern in IMAGE_ANALYSIS_PATTERNS:
        if re.search(pattern, response):
            return True
    
    return False

def moderate_emotional_expressions(response: str, is_new_conversation: bool) -> str:
    """
    Modère les expressions émotionnelles excessives dans la réponse.
    
    Args:
        response: La réponse générée
        is_new_conversation: Indicateur si c'est une nouvelle conversation
    
    Returns:
        La réponse modérée
    """
    # Vérifier s'il s'agit d'une analyse d'image
    is_image_analysis = detect_image_analysis(response)
    
    # Si c'est une analyse d'image, supprimer les phrases excessives d'introduction
    if is_image_analysis:
        for pattern in IMAGE_ANALYSIS_PATTERNS:
            # Remplacer les phrases excessives par un début plus neutre
            if re.search(pattern, response):
                image_intro_phrases = [
                    "Analyse de l'image : ",
                    "Voici l'analyse de cette image : ",
                    "Analyse : ",
                    ""  # Option vide pour commencer directement par la description
                ]
                replacement = random.choice(image_intro_phrases)
                response = re.sub(pattern, replacement, response, count=1)
    
    # Modérer les salutations si ce n'est pas une nouvelle conversation
    if not is_new_conversation:
        for pattern in GREETING_PATTERNS:
            # Remplacer les salutations par une phrase de continuité
            if re.search(pattern, response):
                continuity_phrases = [
                    "Pour continuer notre discussion, ",
                    "Pour revenir à notre sujet, ",
                    "En poursuivant notre échange, ",
                    "Pour reprendre là où nous en étions, ",
                    ""  # Option vide pour simplement supprimer la salutation
                ]
                replacement = random.choice(continuity_phrases)
                response = re.sub(pattern, replacement, response, count=1)
    
    # Modérer les expressions émotionnelles excessives
    emotion_count = 0
    for pattern in EXCESSIVE_EMOTION_PATTERNS:
        matches = re.findall(pattern, response)
        emotion_count += len(matches)
        
        # Limiter le nombre d'expressions émotionnelles à 1 par réponse
        if emotion_count > 1:
            response = re.sub(pattern, "", response)
    
    return response

def process_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Traite les données de la requête.
    
    Args:
        data: Les données de la requête
    
    Returns:
        Les données modifiées
    """
    # Vérifier si c'est une nouvelle conversation
    is_new = is_new_conversation(data)
    
    # Stocker l'information pour être utilisée dans la réponse
    if 'context' not in data:
        data['context'] = {}
    
    if isinstance(data['context'], dict):
        data['context']['is_new_conversation'] = is_new
    
    return data

def process_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Traite les données de la réponse.
    
    Args:
        data: Les données de la réponse
    
    Returns:
        Les données modifiées
    """
    # Récupérer l'indicateur de nouvelle conversation
    is_new = True  # Par défaut
    if 'context' in data and isinstance(data['context'], dict):
        is_new = data['context'].get('is_new_conversation', True)
    
    # Extraire la réponse
    response = None
    if 'response' in data:
        response = data['response']
    elif 'content' in data:
        response = data['content']
    
    # Modérer la réponse si elle existe
    if response:
        moderated_response = moderate_emotional_expressions(response, is_new)
        
        # Mettre à jour la réponse dans les données
        if 'response' in data:
            data['response'] = moderated_response
        elif 'content' in data:
            data['content'] = moderated_response
    
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
            return process_request(data)
            
        elif hook == "process_response":
            return process_response(data)
        
        return data
    
    except Exception as e:
        logger.error(f"Erreur dans le module conversation_context_manager: {str(e)}", exc_info=True)
        return data
