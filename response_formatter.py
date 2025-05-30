"""
Module pour améliorer la structure et la présentation des réponses textuelles de Gemini.
Ce module assure que les réponses sont bien structurées, avec des paragraphes espacés
et formatés correctement.
"""

import re
import logging
import random

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def remove_markdown_symbols(text):
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

def remove_ending_periods(text):
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

def format_response(text):
    """
    Améliore la structure d'une réponse textuelle.
    
    Args:
        text: Le texte à formater
        
    Returns:
        Le texte formaté avec une meilleure structure
    """
    if not text:
        return text
    
    # 1. Supprimer les symboles markdown et les points finaux AVANT tout formatage
    text = remove_markdown_symbols(text)
    text = remove_ending_periods(text)
    
    # 2. Nettoyer les sauts de ligne superflus
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 2. Diviser en paragraphes
    paragraphs = text.split('\n\n')
    
    # Si tout le texte est sur une seule ligne, essayer de le diviser en paragraphes logiques
    if len(paragraphs) <= 1 and len(text) > 200:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        new_paragraphs = []
        current_paragraph = []
        
        for i, sentence in enumerate(sentences):
            current_paragraph.append(sentence)
            
            # Créer un nouveau paragraphe après 3-4 phrases ou si c'est une phrase longue
            if (len(current_paragraph) >= 3 and i < len(sentences) - 1) or len(sentence) > 150:
                new_paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        
        # Ajouter le dernier paragraphe s'il reste des phrases
        if current_paragraph:
            new_paragraphs.append(' '.join(current_paragraph))
        
        paragraphs = new_paragraphs
    
    # 3. Améliorer les paragraphes individuellement
    improved_paragraphs = []
    
    # Mots de liaison pour introduire les paragraphes
    transitions = [
        "Tout d'abord,", "Par ailleurs,", "En outre,", "De plus,", 
        "Par ailleurs,", "Ensuite,", "D'autre part,", "Également,",
        "En effet,", "En ce qui concerne"
    ]
    
    # Phrases de conclusion possibles
    conclusion_phrases = [
        "En conclusion,", "Pour résumer,", "En somme,", "En définitive,",
        "Pour conclure,", "En résumé,", "Au final,", "En bref,"
    ]
    
    for i, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        
        # Passer les paragraphes vides
        if not paragraph:
            continue
            
        # Pour les listes à puces, les conserver telles quelles
        if re.match(r'^[-*•] ', paragraph):
            improved_paragraphs.append(paragraph)
            continue
            
        # Ajouter des transitions appropriées pour les paragraphes du milieu
        if i > 0 and i < len(paragraphs) - 1 and len(paragraphs) > 2:
            # Vérifier si le paragraphe ne commence pas déjà par une transition
            if not any(paragraph.startswith(trans) for trans in transitions + conclusion_phrases):
                if random.random() < 0.6:  # 60% de chance d'ajouter une transition
                    paragraph = f"{random.choice(transitions)} {paragraph}"
        
        # Ajouter une conclusion pour le dernier paragraphe
        if i == len(paragraphs) - 1 and len(paragraphs) > 1 and len(paragraph) > 50:
            # Vérifier si le paragraphe ne commence pas déjà par une phrase de conclusion
            if not any(paragraph.startswith(concl) for concl in conclusion_phrases):
                if random.random() < 0.7:  # 70% de chance d'ajouter une conclusion
                    paragraph = f"{random.choice(conclusion_phrases)} {paragraph}"
        
        improved_paragraphs.append(paragraph)
    
    # 4. Rejoindre les paragraphes avec un double saut de ligne
    formatted_text = '\n\n'.join(improved_paragraphs)
    
    # 5. Ajouter des sauts de ligne avant et après les listes
    formatted_text = re.sub(r'([.!?])\s*\n([-*•])', r'\1\n\n\2', formatted_text)
    
    # 6. Améliorer la structure des phrases trop longues
    sentences = re.split(r'(?<=[.!?])\s+', formatted_text)
    improved_sentences = []
    
    for sentence in sentences:
        # Ajouter des virgules dans les phrases très longues sans ponctuation
        if len(sentence) > 180 and sentence.count(',') < 2:
            parts = re.split(r'\s+(?:et|ou|car|donc|mais|ainsi|puis|comme)\s+', sentence)
            if len(parts) > 1:
                improved_sentence = ""
                for i, part in enumerate(parts):
                    if i < len(parts) - 1:
                        connector = re.search(r'\s+(et|ou|car|donc|mais|ainsi|puis|comme)\s+', sentence[len(improved_sentence):])
                        if connector:
                            improved_sentence += part + ", " + connector.group(0).strip() + " "
                        else:
                            improved_sentence += part + " "
                    else:
                        improved_sentence += part
                sentence = improved_sentence
        
        improved_sentences.append(sentence)
    
    formatted_text = ' '.join(improved_sentences)
    
    # 7. Finalement, s'assurer qu'il y a deux sauts de ligne entre les paragraphes
    formatted_text = re.sub(r'\n{3,}', '\n\n', formatted_text)
    
    return formatted_text

if __name__ == "__main__":
    # Test
    sample_text = "L'intelligence artificielle est un domaine fascinant qui évolue rapidement. Les modèles comme Gemini utilisent des réseaux de neurones profonds pour comprendre et générer du langage naturel. Ces modèles sont entraînés sur d'énormes quantités de données textuelles provenant d'Internet. Ils apprennent ainsi les patterns et structures du langage humain. Les applications sont nombreuses : assistance virtuelle, génération de contenu, traduction automatique, et bien plus encore. La recherche continue d'avancer avec des architectures toujours plus performantes."
    
    print("=== TEXTE ORIGINAL ===")
    print(sample_text)
    print("\n=== TEXTE FORMATÉ ===")
    print(format_response(sample_text))
