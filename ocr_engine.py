import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
from datetime import datetime
import re
from typing import Dict, Any, List, Tuple, Optional

# Configuration de Tesseract OCR (chemin vers l'exécutable Tesseract)
# À modifier selon l'installation sur votre système
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class OCRProcessor:
    """Classe pour gérer le traitement OCR des images"""
    
    def __init__(self, language: str = 'fra'):
        """
        Initialise le processeur OCR
        
        Args:
            language (str): Code de langue pour Tesseract (fra=français, eng=anglais)
        """
        self.language = language
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    
    def _check_image_format(self, image_path: str) -> bool:
        """Vérifie si le format de l'image est supporté"""
        _, ext = os.path.splitext(image_path.lower())
        return ext in self.supported_formats
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Prétraitement de l'image pour améliorer la précision de l'OCR
        
        Args:
            image: Image au format numpy array
            
        Returns:
            Image prétraitée
        """
        # Conversion en niveaux de gris si l'image est en couleur
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Appliquer un seuillage adaptatif pour améliorer le contraste
        # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        #                              cv2.THRESH_BINARY, 11, 2)
        
        # Réduction du bruit
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Amélioration des bords
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(denoised, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        return eroded
    
    def extract_text_from_image(self, image_path: str) -> Dict[str, Any]:
        """
        Extrait le texte d'une image en utilisant OCR
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Dictionnaire contenant le texte extrait et les métadonnées
        """
        result = {
            "success": False,
            "error": None,
            "text": "",
            "confidence": 0,
            "timestamp": datetime.now().isoformat(),
            "language": self.language,
        }
        
        try:
            # Vérifier le format de l'image
            if not self._check_image_format(image_path):
                result["error"] = f"Format d'image non supporté. Formats acceptés: {', '.join(self.supported_formats)}"
                return result
                
            # Vérifier si le fichier existe
            if not os.path.exists(image_path):
                result["error"] = "Le fichier image n'existe pas"
                return result
                
            # Charger l'image avec OpenCV
            image = cv2.imread(image_path)
            if image is None:
                result["error"] = "Impossible de charger l'image"
                return result
                
            # Prétraiter l'image
            processed_image = self._preprocess_image(image)
            
            # Effectuer l'OCR
            ocr_data = pytesseract.image_to_data(
                processed_image, 
                lang=self.language,
                output_type=pytesseract.Output.DICT
            )
            
            # Extraire le texte et les scores de confiance
            text_parts = []
            total_confidence = 0
            word_count = 0
            
            for i in range(len(ocr_data["text"])):
                if int(ocr_data["conf"][i]) > 0:  # Ignorer les entrées avec confiance nulle
                    word = ocr_data["text"][i].strip()
                    if word:
                        text_parts.append(word)
                        total_confidence += int(ocr_data["conf"][i])
                        word_count += 1
            
            # Assembler le texte complet
            full_text = " ".join(text_parts)
            
            # Calculer la confiance moyenne
            avg_confidence = total_confidence / word_count if word_count > 0 else 0
            
            # Préparer le résultat
            result["success"] = True
            result["text"] = full_text
            result["confidence"] = avg_confidence
            result["word_count"] = word_count
            
        except Exception as e:
            result["error"] = f"Erreur lors de l'OCR: {str(e)}"
            
        return result
    
    def analyze_text_content(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyse le contenu du texte extrait pour en tirer des informations structurées
        
        Args:
            ocr_result: Résultat de l'extraction OCR
            
        Returns:
            Dictionnaire avec l'analyse du contenu
        """
        analysis = {
            "type": "unknown",
            "key_information": {},
            "summary": "",
            "keywords": []
        }
        
        if not ocr_result["success"] or not ocr_result["text"]:
            return analysis
            
        text = ocr_result["text"]
        
        # Détection du type de document
        if re.search(r'facture|reçu|paiement|total', text, re.IGNORECASE):
            analysis["type"] = "document_financier"
            
            # Extraction de montants
            amounts = re.findall(r'\d+[,\.]\d+\s*€|\d+[,\.]\d+\s*EUR|\d+\s*€|\d+\s*EUR', text)
            if amounts:
                analysis["key_information"]["montants"] = amounts
                
            # Extraction de dates
            dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s\d{4}', text)
            if dates:
                analysis["key_information"]["dates"] = dates
                
        elif re.search(r'carte d\'identité|passeport|permis|conduire', text, re.IGNORECASE):
            analysis["type"] = "document_identité"
            
            # Extraction de noms
            names = re.findall(r'nom\s*:\s*([^\n\r]+)|prénom\s*:\s*([^\n\r]+)', text, re.IGNORECASE)
            if names:
                analysis["key_information"]["noms"] = [n[0] or n[1] for n in names if n[0] or n[1]]
                
        elif re.search(r'article|nouvelles|journal|reportage', text, re.IGNORECASE):
            analysis["type"] = "article"
            
            # Extraire un titre potentiel (première ligne)
            lines = text.split('\n')
            if lines and len(lines[0]) > 5:
                analysis["key_information"]["titre"] = lines[0]
                
        # Extraction de mots-clés (mots qui apparaissent fréquemment)
        words = re.findall(r'\b\w{4,}\b', text.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
            
        # Obtenir les top 5 mots les plus fréquents comme mots-clés
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        analysis["keywords"] = [word for word, _ in sorted_words[:5]]
        
        # Générer un résumé simple (premiers 100 caractères ou première phrase)
        first_sentence = re.split(r'[.!?]', text)[0].strip()
        analysis["summary"] = first_sentence[:100] + ("..." if len(first_sentence) > 100 else "")
        
        return analysis

def process_image_request(request_text: str) -> Tuple[bool, Optional[str]]:
    """
    Analyse une demande utilisateur pour détecter s'il s'agit d'une demande d'OCR
    
    Args:
        request_text: Texte de la demande de l'utilisateur
        
    Returns:
        (is_ocr_request, image_path): Indique si c'est une requête OCR et le chemin de l'image
    """
    # Motifs pour détecter une demande d'OCR
    ocr_patterns = [
        r'analyse[rz]?\s+cette\s+image',
        r'analyse[rz]?\s+l[\'e]\s*image',
        r'extraire\s+(?:du|le)\s+texte\s+(?:de|depuis|dans)\s+(?:cette|l[\'e])\s+image',
        r'OCR\s+(?:sur|de|pour)\s+(?:cette|l[\'e])\s+image',
        r'lis\s+(?:cette|l[\'e])\s+image',
        r'que\s+(?:dit|contient)\s+(?:cette|l[\'e])\s+image',
        r'texte\s+(?:dans|sur|de)\s+(?:cette|l[\'e])\s+image'
    ]
    
    # Motifs pour extraire le chemin de l'image
    path_patterns = [
        r'(?:image|photo|fichier|document)\s+(?:à|a|dans|de|:)\s+[\'"]?([^\'"\s]+\.(jpg|jpeg|png|bmp|tiff|gif))[\'"]?',
        r'[\'"]([^\'"\s]+\.(jpg|jpeg|png|bmp|tiff|gif))[\'"]',
        r'([a-zA-Z]:\\[^\\/:*?"<>|\r\n]+\.(jpg|jpeg|png|bmp|tiff|gif))',
        r'(/[^\\/:*?"<>|\r\n]+\.(jpg|jpeg|png|bmp|tiff|gif))'
    ]
    
    # Vérifier si la demande correspond à une demande d'OCR
    is_ocr_request = any(re.search(pattern, request_text, re.IGNORECASE) for pattern in ocr_patterns)
    
    # Si c'est une demande d'OCR, essayer d'extraire le chemin de l'image
    image_path = None
    if is_ocr_request:
        for pattern in path_patterns:
            match = re.search(pattern, request_text, re.IGNORECASE)
            if match:
                image_path = match.group(1)
                break
    
    return is_ocr_request, image_path
