"""
Module de gestion de la mémoire textuelle pour les conversations.
Ce module permet de stocker et récupérer des conversations sous forme de fichiers texte,
en complément du stockage SQLite.
"""

import os
import json
import logging
from datetime import datetime
import pathlib
from typing import Dict, List, Any, Optional, Union

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("text_memory_manager")

# Dossier de base pour les conversations textuelles
BASE_CONVERSATIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     'data', 'conversations_text')
                                     
# Dossier pour les uploads d'images
UPLOADS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          'data', 'uploads')

# Créer les dossiers s'ils n'existent pas
os.makedirs(BASE_CONVERSATIONS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

class TextMemoryManager:
    """
    Gestionnaire de mémoire textuelle qui permet de stocker et récupérer
    des conversations sous forme de fichiers texte.
    """
    
    @staticmethod
    def get_user_dir(user_id: int) -> str:
        """
        Obtient le chemin du dossier utilisateur pour les conversations textuelles.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Chemin du dossier utilisateur
        """
        user_dir = os.path.join(BASE_CONVERSATIONS_DIR, str(user_id))
        os.makedirs(user_dir, exist_ok=True)
        return user_dir
        
    @staticmethod
    def get_conversation_file_path(user_id: int, session_id: str) -> str:
        """
        Obtient le chemin du fichier de conversation.
        
        Args:
            user_id: ID de l'utilisateur
            session_id: ID de la session
            
        Returns:
            Chemin du fichier de conversation
        """
        user_dir = TextMemoryManager.get_user_dir(user_id)
        return os.path.join(user_dir, f"{session_id}.txt")
        
    @staticmethod
    def save_message(user_id: int, 
                    session_id: str, 
                    message_type: str, 
                    content: str,
                    image_path: Optional[str] = None,
                    title: Optional[str] = None) -> bool:
        """
        Sauvegarde un message dans le fichier de conversation.
        
        Args:
            user_id: ID de l'utilisateur
            session_id: ID de la session
            message_type: Type de message ('user' ou 'assistant')
            content: Contenu du message
            image_path: Chemin de l'image (optionnel)
            title: Titre de la conversation (optionnel, pour la première entrée)
            
        Returns:
            True si le message a été sauvegardé avec succès
        """
        file_path = TextMemoryManager.get_conversation_file_path(user_id, session_id)
        file_exists = os.path.exists(file_path)
        
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                # Si le fichier vient d'être créé, ajouter un en-tête
                if not file_exists:
                    conversation_title = title or f"Conversation du {datetime.now().strftime('%d/%m/%Y')}"
                    f.write(f"# {conversation_title}\n")
                    f.write(f"Session ID: {session_id}\n")
                    f.write(f"User ID: {user_id}\n")
                    f.write(f"Date de création: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
                    f.write("---\n\n")
                
                # Formater le message
                timestamp = datetime.now().strftime('%H:%M:%S')
                prefix = "Utilisateur" if message_type == "user" else "Assistant"
                
                f.write(f"**{prefix}** ({timestamp}):\n")
                f.write(f"{content}\n\n")
                
                # Ajouter une référence à l'image si présente
                if image_path:
                    f.write(f"[Image: {os.path.basename(image_path)}]\n\n")
                    
                f.write("---\n\n")
            
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du message: {e}")
            return False

    @staticmethod
    def read_conversation(user_id: int, session_id: str) -> Optional[str]:
        """
        Lit le contenu d'une conversation depuis un fichier texte.
        
        Args:
            user_id: ID de l'utilisateur
            session_id: ID de la session
            
        Returns:
            Contenu de la conversation ou None si le fichier n'existe pas
        """
        file_path = TextMemoryManager.get_conversation_file_path(user_id, session_id)
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Erreur lors de la lecture de la conversation: {e}")
            return None
    
    @staticmethod
    def list_conversations(user_id: int) -> List[Dict[str, Any]]:
        """
        Liste toutes les conversations d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Liste des conversations de l'utilisateur
        """
        user_dir = TextMemoryManager.get_user_dir(user_id)
        conversations = []
        
        try:
            for file_name in os.listdir(user_dir):
                if file_name.endswith('.txt'):
                    session_id = file_name.replace('.txt', '')
                    file_path = os.path.join(user_dir, file_name)
                    
                    # Obtenir les métadonnées du fichier
                    file_stats = os.stat(file_path)
                    creation_time = datetime.fromtimestamp(file_stats.st_ctime)
                    modification_time = datetime.fromtimestamp(file_stats.st_mtime)
                    
                    # Lire le titre depuis le fichier
                    title = ""
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            first_line = f.readline().strip()
                            if first_line.startswith('# '):
                                title = first_line[2:]
                    except:
                        title = f"Conversation {session_id}"
                    
                    conversations.append({
                        'session_id': session_id,
                        'title': title,
                        'created_at': creation_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'last_updated': modification_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'file_path': file_path
                    })
            
            # Trier par date de modification (plus récent d'abord)
            conversations.sort(key=lambda x: x['last_updated'], reverse=True)
            
            return conversations
        except Exception as e:
            logger.error(f"Erreur lors du listing des conversations: {e}")
            return []

    @staticmethod
    def delete_conversation(user_id: int, session_id: str) -> bool:
        """
        Supprime une conversation.
        
        Args:
            user_id: ID de l'utilisateur
            session_id: ID de la session
            
        Returns:
            True si la suppression a réussi
        """
        file_path = TextMemoryManager.get_conversation_file_path(user_id, session_id)
        
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                return True
            except Exception as e:
                logger.error(f"Erreur lors de la suppression de la conversation: {e}")
                return False
        return False

    @staticmethod
    def search_conversations(user_id: int, query: str) -> List[Dict[str, Any]]:
        """
        Recherche dans les conversations d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            query: Texte à rechercher
            
        Returns:
            Liste des conversations contenant la requête
        """
        conversations = TextMemoryManager.list_conversations(user_id)
        results = []
        
        for conversation in conversations:
            file_path = conversation['file_path']
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if query.lower() in content.lower():
                        # Ajouter un extrait avec le contexte
                        lines = content.split('\n')
                        matching_lines = []
                        for i, line in enumerate(lines):
                            if query.lower() in line.lower():
                                start = max(0, i - 2)
                                end = min(len(lines), i + 3)
                                context = '\n'.join(lines[start:end])
                                matching_lines.append(context)
                        
                        conversation['matching_excerpts'] = matching_lines[:5]  # Limiter à 5 extraits
                        results.append(conversation)
            except Exception as e:
                logger.error(f"Erreur lors de la recherche dans {file_path}: {e}")
        
        return results
        
    @staticmethod
    def get_user_uploads_dir(user_id: int) -> str:
        """
        Obtient le dossier des uploads d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Chemin du dossier des uploads de l'utilisateur
        """
        user_uploads_dir = os.path.join(UPLOADS_DIR, str(user_id))
        os.makedirs(user_uploads_dir, exist_ok=True)
        return user_uploads_dir
        
    @staticmethod
    def save_uploaded_image(user_id: int, image_data: str, filename: Optional[str] = None) -> Optional[str]:
        """
        Sauvegarde une image téléchargée.
        
        Args:
            user_id: ID de l'utilisateur
            image_data: Données de l'image en base64
            filename: Nom de fichier (optionnel)
            
        Returns:
            Chemin relatif de l'image sauvegardée ou None en cas d'erreur
        """
        import base64
        
        user_uploads_dir = TextMemoryManager.get_user_uploads_dir(user_id)
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            filename = f"upload_{timestamp}.jpg"
        
        file_path = os.path.join(user_uploads_dir, filename)
        
        try:
            # Vérifier si les données d'image sont vides
            if not image_data:
                logger.error("Données d'image vides")
                return None
                
            # Vérifier si c'est déjà au format base64 ou si c'est un format URI Data
            if isinstance(image_data, str):
                # Extraire les données binaires de l'image (retirer le préfixe data:image/xxx;base64,)
                if ',' in image_data:
                    image_data = image_data.split(',', 1)[1]
                # Nettoyer les caractères non base64 qui pourraient causer des problèmes
                image_data = image_data.strip()
                
                # Décoder et sauvegarder l'image
                try:
                    with open(file_path, 'wb') as f:
                        f.write(base64.b64decode(image_data, validate=True))
                    logger.info(f"Image sauvegardée avec succès à {file_path}")
                except Exception as decode_error:
                    logger.error(f"Erreur de décodage base64: {str(decode_error)}")
                    return None
            else:
                logger.error("Format d'image non supporté")
                return None
                
            # Retourner le chemin relatif
            return os.path.relpath(file_path, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'image: {str(e)}")
            return None

    @staticmethod
    def list_uploaded_images(user_id: int) -> List[Dict[str, Any]]:
        """
        Liste les images téléchargées par un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Liste des images téléchargées
        """
        user_uploads_dir = TextMemoryManager.get_user_uploads_dir(user_id)
        images = []
        
        try:
            for file_name in os.listdir(user_uploads_dir):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    file_path = os.path.join(user_uploads_dir, file_name)
                    file_stats = os.stat(file_path)
                    
                    images.append({
                        'filename': file_name,
                        'path': file_path,
                        'relative_path': os.path.relpath(file_path, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        'size': file_stats.st_size,
                        'created_at': datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
                    })
            
            return images
        except Exception as e:
            logger.error(f"Erreur lors du listing des images: {e}")
            return []
