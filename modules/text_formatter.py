"""
Module de formatage avancé pour améliorer la structure des réponses de Gemini.
Ce module assure que les réponses sont bien structurées, correctement espacées,
avec des paragraphes de longueur appropriée et un style d'écriture fluide.
"""

import re
import logging
import random
from typing import Dict, Any, List, Optional

# Métadonnées du module
MODULE_METADATA = {
    "enabled": True,
    "priority": 20,  # Priorité élevée pour s'exécuter après la génération mais avant d'autres modifications
    "description": "Améliore la structure et la présentation des réponses textuelles",
    "version": "1.0.0",
    "dependencies": [],
    "hooks": ["process_response"]
}

# Configuration du logger
logger = logging.getLogger(__name__)

def remove_markdown_symbols(text: str) -> str:
    """
    Supprime les symboles markdown (* et **) du texte.
    
    Args:
        text: Le texte à nettoyer
        
    Returns:
        Le texte sans symboles markdown
    """
    if not text:
        return text
    
    # Supprimer les symboles ** (gras)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    
    # Supprimer les symboles * (italique)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # Supprimer les * isolés qui pourraient rester
    text = re.sub(r'\*+', '', text)
    
    return text

def remove_ending_periods(text: str) -> str:
    """
    Supprime les points en fin de phrase.
    
    Args:
        text: Le texte à nettoyer
        
    Returns:
        Le texte sans points finaux
    """
    if not text:
        return text
    
    # Diviser en paragraphes
    paragraphs = text.split('\n\n')
    cleaned_paragraphs = []
    
    for paragraph in paragraphs:
        if paragraph.strip():
            # Supprimer les points en fin de paragraphe
            paragraph = paragraph.rstrip()
            while paragraph.endswith('.'):
                paragraph = paragraph[:-1].rstrip()
            cleaned_paragraphs.append(paragraph)
    
    return '\n\n'.join(cleaned_paragraphs)

# Connecteurs pour créer des phrases plus longues et plus fluides
SENTENCE_CONNECTORS = [
    ", ce qui signifie que", 
    ", d'autant plus que", 
    " puisque", 
    " tandis que", 
    " de sorte que",
    " alors même que",
    ", tout en sachant que",
    " bien que",
    " étant donné que",
    ", et par conséquent,"
]

# Connecteurs pour introduire des exemples ou des clarifications
CLARIFICATION_CONNECTORS = [
    "En d'autres termes,",
    "Pour être plus précis,",
    "Il est important de noter que",
    "Il convient de souligner que",
    "Pour illustrer ce point,",
    "Ce qui implique essentiellement que",
    "Autrement dit,",
    "En particulier,"
]

# Expressions pour introduire différentes parties du texte
SECTION_TRANSITIONS = [
    "Tout d'abord,",
    "Ensuite,", 
    "De plus,",
    "Par ailleurs,",
    "En outre,",
    "D'autre part,",
    "Pour aller plus loin,",
    "Pour conclure,",
    "En résumé,"
]

def format_paragraphs(text: str) -> str:
    """
    S'assure que le texte est correctement divisé en paragraphes avec espacement approprié.
    
    Args:
        text: Le texte à formater
        
    Returns:
        Le texte formaté avec des paragraphes correctement espacés
    """
    if not text:
        return text
    
    # Éliminer les espaces multiples
    text = re.sub(r' +', ' ', text)
    
    # Standardiser les sauts de ligne
    text = re.sub(r'(\r\n|\r|\n)+', '\n', text)
    
    # Diviser en paragraphes en gardant les sauts de ligne existants
    paragraphs = text.split('\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Si nous avons moins de 2 paragraphes et que le texte est assez long,
    # essayer de le diviser en paragraphes plus courts
    if len(paragraphs) < 2 and len(text) > 500:
        # Diviser les phrases
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Regrouper les phrases en paragraphes (environ 3-5 phrases par paragraphe)
        new_paragraphs = []
        current_paragraph = []
        sentence_count = 0
        sentences_per_paragraph = random.randint(3, 5)
        
        for sentence in sentences:
            current_paragraph.append(sentence)
            sentence_count += 1
            
            if sentence_count >= sentences_per_paragraph:
                new_paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
                sentence_count = 0
                sentences_per_paragraph = random.randint(3, 5)  # Varier la longueur des paragraphes
        
        # Ajouter le dernier paragraphe s'il reste des phrases
        if current_paragraph:
            new_paragraphs.append(' '.join(current_paragraph))
        
        paragraphs = new_paragraphs
    
    # Rejoindre les paragraphes avec double saut de ligne
    formatted_text = '\n\n'.join(paragraphs)
    
    return formatted_text

def combine_short_sentences(text: str) -> str:
    """
    Combine certaines phrases courtes pour créer des phrases plus longues et plus fluides.
    
    Args:
        text: Le texte à modifier
        
    Returns:
        Le texte avec certaines phrases courtes combinées
    """
    # Diviser le texte en phrases
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if len(sentences) <= 2:
        return text
    
    result_sentences = []
    i = 0
    
    while i < len(sentences) - 1:
        current = sentences[i]
        next_sentence = sentences[i + 1]
        
        # Si les deux phrases sont courtes, envisager de les combiner
        if 10 <= len(current) <= 60 and 10 <= len(next_sentence) <= 60:
            # 40% de chance de combiner (pas toutes les phrases courtes)
            if random.random() < 0.4:
                connector = random.choice(SENTENCE_CONNECTORS)
                combined = current.rstrip('.!?') + connector + ' ' + next_sentence[0].lower() + next_sentence[1:]
                result_sentences.append(combined)
                i += 2
                continue
        
        result_sentences.append(current)
        i += 1
    
    # Ajouter la dernière phrase si elle n'a pas été combinée
    if i < len(sentences):
        result_sentences.append(sentences[i])
    
    # Rejoindre les phrases
    return ' '.join(result_sentences)

def enhance_paragraph_structure(text: str) -> str:
    """
    Améliore la structure des paragraphes en ajoutant des transitions et des clarifications.
    
    Args:
        text: Le texte à améliorer
        
    Returns:
        Le texte avec une structure de paragraphe améliorée
    """
    # Diviser en paragraphes
    paragraphs = text.split('\n\n')
    
    # Si nous n'avons qu'un seul paragraphe, le retourner tel quel
    if len(paragraphs) <= 1:
        return text
    
    # Améliorer chaque paragraphe (sauf le premier, pour qu'il reste naturel)
    improved_paragraphs = [paragraphs[0]]
    
    for i in range(1, len(paragraphs)):
        paragraph = paragraphs[i]
        
        # Ajouter des transitions entre paragraphes (50% de chance)
        if random.random() < 0.5 and not paragraph.startswith(tuple(SECTION_TRANSITIONS)):
            transition = random.choice(SECTION_TRANSITIONS)
            paragraph = f"{transition} {paragraph}"
        
        improved_paragraphs.append(paragraph)
    
    # Rejoindre les paragraphes
    return '\n\n'.join(improved_paragraphs)

def add_clarifications(text: str) -> str:
    """
    Ajoute des clarifications ou des exemples dans certains paragraphes.
    
    Args:
        text: Le texte à améliorer
        
    Returns:
        Le texte avec des clarifications ajoutées
    """
    # Diviser en paragraphes
    paragraphs = text.split('\n\n')
    
    # Si nous avons peu de paragraphes, ne rien modifier
    if len(paragraphs) < 2:
        return text
    
    # Pour certains paragraphes longs, ajouter une clarification
    improved_paragraphs = []
    
    for paragraph in paragraphs:
        if len(paragraph) > 200 and random.random() < 0.3:
            # Diviser le paragraphe en deux parties approximativement
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            mid_point = len(sentences) // 2
            
            first_part = ' '.join(sentences[:mid_point])
            second_part = ' '.join(sentences[mid_point:])
            
            # Ajouter une clarification
            clarification = random.choice(CLARIFICATION_CONNECTORS)
            
            improved_paragraph = f"{first_part} {clarification} {second_part}"
            improved_paragraphs.append(improved_paragraph)
        else:
            improved_paragraphs.append(paragraph)
    
    # Rejoindre les paragraphes
    return '\n\n'.join(improved_paragraphs)

def process(data: Dict[str, Any], hook: str) -> Dict[str, Any]:
    """
    Traite les données de réponse pour améliorer la structure du texte.
    
    Args:
        data: Les données à traiter
        hook: Le hook appelé
        
    Returns:
        Les données avec le texte formaté
    """
    # Ne traiter que les réponses
    if hook != "process_response":
        return data
    
    try:
        # Vérifier si le texte est présent et de type str
        if "text" in data and isinstance(data["text"], str):
            original_text = data["text"]
            
            # Appliquer IMMÉDIATEMENT la suppression des symboles markdown et points finaux
            cleaned_text = remove_markdown_symbols(original_text)
            cleaned_text = remove_ending_periods(cleaned_text)
            
            # Appliquer les transformations de formatage sur le texte nettoyé
            formatted_text = format_paragraphs(cleaned_text)
            formatted_text = combine_short_sentences(formatted_text)
            formatted_text = enhance_paragraph_structure(formatted_text)
            formatted_text = add_clarifications(formatted_text)
            
            # Mettre à jour le texte
            data["text"] = formatted_text
            logger.info("Texte reformaté avec suppression des symboles markdown et points finaux")
        
        return data
    
    except Exception as e:
        logger.error(f"Erreur lors du formatage du texte: {str(e)}")
        return data

# Pour les tests autonomes
if __name__ == "__main__":
    test_text = "Voici une première phrase courte. Voici une deuxième phrase courte. Ce paragraphe est assez court. Il ne contient que des phrases simples. L'intelligence artificielle tend à produire ce style d'écriture. Nous voulons améliorer cela. Nous pouvons rendre le texte plus fluide. Les réponses seront alors plus naturelles."
    
    print("===== TEXTE ORIGINAL =====")
    print(test_text)
    
    formatted = format_paragraphs(test_text)
    print("\n===== FORMATAGE DES PARAGRAPHES =====")
    print(formatted)
    
    combined = combine_short_sentences(formatted)
    print("\n===== COMBINAISON DES PHRASES COURTES =====")
    print(combined)
    
    enhanced = enhance_paragraph_structure(combined)
    print("\n===== AMÉLIORATION DE LA STRUCTURE =====")
    print(enhanced)
    
    clarified = add_clarifications(enhanced)
    print("\n===== AJOUT DE CLARIFICATIONS =====")
    print(clarified)
