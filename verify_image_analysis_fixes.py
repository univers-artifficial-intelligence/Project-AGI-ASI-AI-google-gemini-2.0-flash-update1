"""
Script de vérification des améliorations d'analyse d'image.
Ce script affiche les modifications apportées pour faciliter la validation des changements.
"""

import os
import sys

def print_section(title):
    """Affiche un titre de section formaté."""
    print("\n" + "=" * 50)
    print(f" {title} ".center(50, "="))
    print("=" * 50)

def print_changes():
    """Affiche un résumé des changements effectués."""
    print_section("RÉSUMÉ DES AMÉLIORATIONS")
    
    print("""
1. PROBLÈME: Phrases excessives dans les analyses d'images
   SOLUTION: Ajout de patterns de détection spécifiques dans le module conversation_context_manager.py
             et modification de la fonction moderate_emotional_expressions pour les filtrer

2. PROBLÈME: État émotionnel initial "confused" lors de l'analyse d'images
   SOLUTION: Création d'une fonction initialize_emotion dans emotional_engine.py
             qui initialise l'état à "neutre" pour les analyses d'images

3. PROBLÈME: Formulations répétitives dans les conversations
   SOLUTION: Amélioration du système de détection des conversationd continues
             dans conversation_context_manager.py et conversation_memory_enhancer.py
    """)
    
    print_section("DÉTECTION DES ANALYSES D'IMAGES")
    
    print("""
# Patterns spécifiques pour les réponses d'analyse d'images
IMAGE_ANALYSIS_PATTERNS = [
    r"(?i)^(Absolument\\s?!?\\s?Je suis ravi de pouvoir t'aider\\.?\\s?Oui,?\\s?je ressens des émotions en analysant cette image\\s?Analyse de l'image)",
    r"(?i)^(Je suis (ravi|heureux|content) de pouvoir analyser cette image pour toi\\.?\\s?Analyse de l'image)",
    r"(?i)^(Analyse de l'image\\s?:?\\s?)"
]

def detect_image_analysis(response: str) -> bool:
    # Mots-clés fréquents dans les analyses d'images
    image_keywords = [
        r"(?i)(cette image montre)",
        r"(?i)(dans cette image,)",
        r"(?i)(l'image présente)",
        r"(?i)(on peut voir sur cette image)",
        r"(?i)(je vois une image qui)",
    ]
    
    # Si la réponse contient des mots-clés d'analyse d'image
    for pattern in image_keywords:
        if re.search(pattern, response):
            return True
    # Ou si la réponse est déjà identifiée comme une analyse d'image
    for pattern in IMAGE_ANALYSIS_PATTERNS:
        if re.search(pattern, response):
            return True
    return False
    """)
    
    print_section("ÉTAT ÉMOTIONNEL POUR L'ANALYSE D'IMAGES")
    
    print("""
def initialize_emotion(context_type=None):
    # Si le contexte est l'analyse d'image, commencer avec un état neutre
    if context_type == 'image_analysis':
        update_emotion("neutral", 0.5, trigger="image_analysis_start")
        return
    
    # Pour tout autre contexte, choisir une émotion aléatoire
    states = list(EMOTIONAL_STATES.keys())
    if "neutral" in states:
        states.remove("neutral")
    
    random_state = random.choice(states)
    min_intensity, max_intensity = EMOTIONAL_STATES[random_state]["intensity_range"]
    intensity = round(random.uniform(min_intensity, max_intensity), 2)
    
    update_emotion(random_state, intensity, trigger="initialization")
    """)
    
    print_section("INSTRUCTIONS POUR LE PROMPT GEMINI")
    
    print("""
ANALYSE D'IMAGES: Tu as la capacité d'analyser des images en détail. Quand on te montre une image:
1. Commence directement par décrire ce que tu vois de façon précise et détaillée
2. Identifie les éléments importants dans l'image
3. Si c'est pertinent, explique ce que représente l'image
4. Tu peux exprimer ton impression sur l'image mais de façon modérée et naturelle

IMPORTANT: NE COMMENCE JAMAIS ta réponse par "Absolument ! Je suis ravi de pouvoir t'aider." 
ou "Je ressens des émotions en analysant cette image". 
Commence directement par la description de l'image.
    """)

if __name__ == "__main__":
    print_changes()
