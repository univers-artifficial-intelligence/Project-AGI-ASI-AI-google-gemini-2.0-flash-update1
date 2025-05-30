"""
Script de diagnostic de la continuité des conversations.
Ce script teste la capacité de l'IA à maintenir la continuité des conversations
et à équilibrer ses réponses émotionnelles.
"""

import sys
import logging
import os
import time
from memory_engine import MemoryEngine
from modules.conversation_context_manager import is_new_conversation, moderate_emotional_expressions

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("conversation_continuity_diagnosis")

def print_diagnostic_header(title):
    """
    Affiche un titre de diagnostic formaté.
    """
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def run_continuity_diagnostic():
    """
    Exécute un diagnostic de la continuité des conversations.
    """
    print_diagnostic_header("DIAGNOSTIC DE CONTINUITÉ DES CONVERSATIONS")
    print("Ce script vérifie la capacité de l'IA à maintenir la continuité des conversations")
    print("et à équilibrer ses réponses émotionnelles.")
    
    # Vérifier si le module MemoryEngine est correctement configuré
    memory_engine = MemoryEngine()
    
    try:
        # Test de détection de nouvelles conversations vs conversations en cours
        print("\n1. Test de détection de l'état de la conversation")
        
        # Exemple de données pour une nouvelle conversation
        new_conversation_data = {
            'user_id': 1,
            'session_id': f"test_session_{int(time.time())}",  # ID de session unique
            'text': "Bonjour, comment ça va?"
        }
        
        is_new = is_new_conversation(new_conversation_data)
        print(f"- Nouvelle conversation détectée: {is_new} (attendu: True)")
        
        # Exemple de modération d'expressions émotionnelles
        print("\n2. Test de modération des expressions émotionnelles")
        
        test_responses = [
            "Bonjour ! Je suis vraiment ravi de vous rencontrer. Comment puis-je vous aider aujourd'hui ?",
            "Je ressens beaucoup d'enthousiasme face à cette question ! C'est un sujet passionnant qui me fascine énormément !",
            "Je suis incroyablement excité de pouvoir travailler avec vous sur ce projet !"
        ]
        
        print("Réponses avant et après modération :")
        for i, response in enumerate(test_responses):
            moderated = moderate_emotional_expressions(response, is_new=False)
            print(f"\nOriginal [{i+1}]: {response}")
            print(f"Modéré  [{i+1}]: {moderated}")
        
        print("\nDiagnostic complété avec succès !")
        print("\nPour une vérification plus approfondie, veuillez tester l'application")
        print("complète et observer le comportement de l'IA dans des conversations réelles.")
    
    except Exception as e:
        logger.error(f"Erreur lors du diagnostic: {str(e)}", exc_info=True)
        print(f"\nERREUR: {str(e)}")
        print("Le diagnostic n'a pas pu être complété.")

if __name__ == "__main__":
    run_continuity_diagnostic()
