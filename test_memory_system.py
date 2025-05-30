"""
Script de test pour vérifier le système de mémorisation de GeminiChat.
Ce script teste l'accès aux fichiers de conversations précédentes et la fonction process_memory_request.
"""

import os
import logging
import uuid
import sys
from datetime import datetime

# Configuration du logger pour afficher dans la console
logger = logging.getLogger('memory_test')
logger.setLevel(logging.INFO)

# Créer un formateur avec des couleurs pour une meilleure lisibilité
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Ajouter un handler de console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Import des modules nécessaires
from modules.text_memory_manager import TextMemoryManager
from gemini_api import process_memory_request, get_conversation_history

def create_test_conversation(user_id=1):
    """
    Crée une conversation de test pour vérifier le système de mémorisation.
    
    Args:
        user_id: ID de l'utilisateur pour le test
        
    Returns:
        ID de la session créée
    """
    # Générer un ID de session unique
    session_id = str(uuid.uuid4())
    logger.info(f"Création d'une conversation de test avec session_id: {session_id}")
    
    # Messages pour la conversation de test
    test_messages = [
        ("user", "Bonjour, j'aimerais discuter de l'intelligence artificielle."),
        ("assistant", "Bonjour ! Je serais ravi de discuter d'intelligence artificielle avec vous. Que souhaitez-vous savoir en particulier ?"),
        ("user", "Peux-tu m'expliquer ce qu'est l'apprentissage profond ?"),
        ("assistant", "L'apprentissage profond est une branche du machine learning qui utilise des réseaux de neurones à plusieurs couches pour apprendre à partir des données. Ces réseaux peuvent extraire automatiquement des caractéristiques de haut niveau à partir de données brutes, ce qui les rend particulièrement efficaces pour des tâches comme la vision par ordinateur et le traitement du langage naturel."),
        ("user", "Merci pour cette explication. Quels sont les principaux frameworks d'apprentissage profond ?"),
        ("assistant", "Les principaux frameworks d'apprentissage profond incluent TensorFlow (développé par Google), PyTorch (développé par Facebook/Meta), Keras (maintenant intégré à TensorFlow), JAX (également de Google), et MXNet (utilisé par Amazon). TensorFlow et PyTorch sont actuellement les plus populaires, chacun avec ses propres forces. TensorFlow est souvent utilisé en production, tandis que PyTorch est très apprécié pour la recherche en raison de sa flexibilité.")
    ]
    
    # Enregistrer les messages dans le fichier de conversation
    for message_type, content in test_messages:
        TextMemoryManager.save_message(
            user_id=user_id,
            session_id=session_id,
            message_type=message_type,
            content=content
        )
        # Ajouter un délai fictif pour simuler l'écoulement du temps
        logger.info(f"Message de type '{message_type}' ajouté à la conversation")
    
    logger.info(f"Conversation de test créée avec {len(test_messages)} messages")
    return session_id

def test_memory_system():
    """Teste le système de mémorisation"""
    user_id = 1
    print("\n" + "="*80)
    print("\033[1;36mTESTS DU SYSTÈME DE MÉMORISATION GEMINICHAT\033[0m")
    print("="*80)
    
    # Créer une conversation de test
    session_id = create_test_conversation(user_id)
    
    # Test 1: Récupérer l'historique de la conversation
    print("\n" + "-"*50)
    print("\033[1;33mTEST 1: Récupération de l'historique de conversation\033[0m")
    print("-"*50)
    conversation_history = get_conversation_history(user_id, session_id)
    if conversation_history:
        print("[✓] SUCCÈS: Historique récupéré ({} caractères)".format(len(conversation_history)))
        print("Extrait: {}...".format(conversation_history[:100].replace("\n", " ")))
        logger.info(f"Test 1 réussi: Historique récupéré ({len(conversation_history)} caractères)")
    else:
        print("[✗] ÉCHEC: Impossible de récupérer l'historique")
        logger.error("Test 1 échoué: Impossible de récupérer l'historique")
    
    # Test 2: Réponse à une demande de mémoire spécifique
    print("\n" + "-"*50)
    print("\033[1;33mTEST 2: Traitement d'une demande liée à la mémoire\033[0m")
    print("-"*50)
    memory_request = "Peux-tu me rappeler de quoi nous avons parlé concernant l'apprentissage profond ?"
    print(f"Requête: \"{memory_request}\"")
    memory_context = process_memory_request(memory_request, user_id, session_id)
    
    if memory_context:
        print("[✓] SUCCÈS: La demande de mémoire a généré un contexte")
        print(f"Contexte généré: {memory_context[:100].replace('\n', ' ')}...")
        logger.info(f"Test 2 réussi: La demande de mémoire a généré un contexte")
    else:
        print("[✗] ÉCHEC: Aucun contexte généré pour la demande de mémoire")
        logger.error("Test 2 échoué: Aucun contexte généré pour la demande de mémoire")
    
    # Test 3: Vérifier que les mots-clés de mémoire sont bien détectés
    print("\n" + "-"*50)
    print("\033[1;33mTEST 3: Détection des mots-clés de mémoire\033[0m")
    print("-"*50)
    memory_keywords_tests = [
        ("Tu te souviens de notre discussion sur l'IA ?", True),
        ("Rappelle-moi ce que tu as dit sur TensorFlow", True),
        ("Qu'est-ce que le machine learning ?", False),
        ("Nous avions précédemment parlé des frameworks", True)
    ]
    
    test3_success = 0
    test3_total = len(memory_keywords_tests)
    
    print("Vérification de la détection des requêtes de mémoire:")
    for prompt, expected in memory_keywords_tests:
        result = process_memory_request(prompt, user_id, session_id) is not None
        if result == expected:
            mark = "[✓]"
            status = "SUCCÈS"
            test3_success += 1
            logger.info(f"Prompt '{prompt}' correctement {'identifié' if expected else 'ignoré'} comme demande mémoire")
        else:
            mark = "[✗]"
            status = "ÉCHEC"
            logger.error(f"Prompt '{prompt}' incorrectement {'ignoré' if expected else 'identifié'} comme demande mémoire")
            print(f"{mark} {status}: \"{prompt}\" - {'Devrait être' if expected else 'Ne devrait pas être'} une demande de mémoire")
    
    # Test 4: Vérifier le stockage des images
    print("\n" + "-"*50)
    print("\033[1;33mTEST 4: Stockage des images\033[0m")
    print("-"*50)
    # Créer une petite image de test en base64
    test_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+P+/HgAFeAJYgZIL4AAAAABJRU5ErkJggg=="
    print("Tentative de sauvegarde d'une image de test en base64...")
    
    image_path = TextMemoryManager.save_uploaded_image(user_id, test_image)
    
    if image_path:
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), image_path)
        if os.path.exists(full_path):
            print(f"[✓] SUCCÈS: Image sauvegardée correctement à {image_path}")
            logger.info(f"Test 4 réussi: Image sauvegardée correctement à {image_path}")
        else:
            print(f"[✗] ÉCHEC: Chemin de fichier invalide {full_path}")
            logger.error(f"Test 4 échoué: Chemin de fichier invalide {full_path}")
    else:
        print("[✗] ÉCHEC: Échec de l'enregistrement de l'image")
        logger.error("Test 4 échoué: Échec de l'enregistrement de l'image")
    
    # Résumé des tests
    print("\n" + "="*50)
    print("RÉSUMÉ DES TESTS")
    print("="*50)
    
    # Calculer le résultat final des tests
    test1_success = conversation_history is not None
    test2_success = memory_context is not None
    test4_success = image_path is not None and os.path.exists(full_path) if image_path else False
    
    total_tests = 4  # 4 tests au total
    successful_tests = sum([
        1 if test1_success else 0,
        1 if test2_success else 0,
        test3_success,  # Déjà un nombre
        1 if test4_success else 0
    ])
    
    print(f"Tests réussis: {successful_tests}/{total_tests}")
    print(f"Test 1 (Récupération historique): {'✓' if test1_success else '✗'}")
    print(f"Test 2 (Traitement requête mémoire): {'✓' if test2_success else '✗'}")
    print(f"Test 3 (Détection mots-clés): {test3_success}/{test3_total}")
    print(f"Test 4 (Stockage images): {'✓' if test4_success else '✗'}")
    
    print("\n" + "="*50)
    print("Tests terminés")
    print("="*50)
    
    return session_id

if __name__ == "__main__":
    try:
        print("\nDémarrage des tests du système de mémorisation...")
        session_id = test_memory_system()
        print(f"\nID de session de test: {session_id}")
        print("\nVous pouvez maintenant démarrer l'application et essayer de demander à l'IA:")
        print("\"Peux-tu me rappeler de quoi nous avons parlé concernant l'apprentissage profond ?\"")
    except Exception as e:
        print(f"\n[!] ERREUR pendant l'exécution des tests: {str(e)}")
        import traceback
        print("\nDétails de l'erreur:")
        traceback.print_exc()
