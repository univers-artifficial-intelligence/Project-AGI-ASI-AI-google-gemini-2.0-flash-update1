import os
import json
import sqlite3
import logging
import datetime
import base64
import uuid
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import pytz

from time_engine import should_remember_conversation, timestamp_to_readable_time_diff, MEMORY_RETENTION
from modules.text_memory_manager import TextMemoryManager  # Nouveau module de gestion de mémoire textuelle

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('memory_engine')

# Chemins pour le stockage des données
from database import DB_PATH
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'conversation_images')
os.makedirs(IMAGES_DIR, exist_ok=True)

class MemoryEngine:
    """
    Gère la mémoire des conversations de l'IA Gemini, y compris le texte et les images.
    """
    
    def __init__(self):
        """Initialise le moteur de mémoire."""
        self.db_path = DB_PATH
        self.setup_database()
        
        # Initialiser le gestionnaire de mémoire textuelle
        self.text_memory_enabled = True  # Activer la sauvegarde en fichiers texte
        self.upload_folder_enabled = True  # Activer le dossier d'uploads
        
    def setup_database(self):
        """Configure la base de données pour stocker les conversations."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table des conversations (sessions)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            session_id TEXT UNIQUE NOT NULL,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # Table des messages dans les conversations
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_messages (
            id INTEGER PRIMARY KEY,
            conversation_id INTEGER NOT NULL,
            message_type TEXT NOT NULL,  -- 'user' ou 'assistant'
            content TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            has_image BOOLEAN DEFAULT FALSE,
            has_file BOOLEAN DEFAULT FALSE,
            emotional_state TEXT,
            metadata TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        )
        ''')
        
        # Table des images liées aux messages
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS message_images (
            id INTEGER PRIMARY KEY,
            message_id INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            original_filename TEXT,
            thumbnail_path TEXT,
            content_type TEXT,
            description TEXT,
            metadata TEXT,
            FOREIGN KEY (message_id) REFERENCES conversation_messages (id)
        )
        ''')
        
        # Table des fichiers liés aux messages
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS message_files (
            id INTEGER PRIMARY KEY,
            message_id INTEGER NOT NULL,
            file_id INTEGER NOT NULL,
            FOREIGN KEY (message_id) REFERENCES conversation_messages (id),
            FOREIGN KEY (file_id) REFERENCES uploaded_files (id)
        )
        ''')
        
        # Table pour la mémoire à court terme (conversations récentes)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            content TEXT NOT NULL,
            memory_level TEXT DEFAULT 'MEDIUM_TERM',
            metadata TEXT
        )
        ''')
        
        # Table pour la mémoire à long terme (faits importants, préférences, etc.)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS long_term_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            category TEXT NOT NULL,
            content TEXT NOT NULL,
            importance INTEGER DEFAULT 5,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Index pour améliorer les performances des requêtes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations (user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_conversation ON conversation_messages (conversation_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_images_message ON message_images (message_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_message ON message_files (message_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_user_id ON conversation_memory (user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_session ON conversation_memory (session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ltm_user_id ON long_term_memory (user_id)')
        
        conn.commit()
        conn.close()
        
    def create_conversation(self, user_id: int, title: Optional[str] = None) -> str:
        """
        Crée une nouvelle conversation pour un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            title: Titre optionnel de la conversation
            
        Returns:
            ID de session unique pour la conversation
        """
        session_id = str(uuid.uuid4())
        title = title or f"Conversation du {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO conversations (user_id, session_id, title) VALUES (?, ?, ?)',
            (user_id, session_id, title)
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Created new conversation for user {user_id}, session_id: {session_id}")
        return session_id
    
    def get_or_create_conversation(self, user_id: int, session_id: Optional[str] = None) -> str:
        """
        Récupère une conversation existante ou en crée une nouvelle.
        
        Args:
            user_id: ID de l'utilisateur
            session_id: ID de session optionnel
            
        Returns:
            ID de session de la conversation
        """
        if not session_id:
            return self.create_conversation(user_id)
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Vérifier si la conversation existe
        cursor.execute(
            'SELECT id FROM conversations WHERE session_id = ? AND user_id = ?',
            (session_id, user_id)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return session_id
        else:
            logger.warning(f"Session {session_id} not found for user {user_id}, creating new one")
            return self.create_conversation(user_id)
    
    def add_message(self, session_id: str, user_id: int, 
                   message_type: str, content: str,
                   image_data: Optional[str] = None, 
                   file_id: Optional[int] = None,
                   emotional_state: Optional[Dict] = None) -> int:
        """
        Ajoute un message à une conversation.
        
        Args:
            session_id: ID de la session de conversation
            user_id: ID de l'utilisateur
            message_type: Type de message ('user' ou 'assistant')
            content: Contenu textuel du message
            image_data: Données d'image en base64 (optionnel)
            file_id: ID du fichier associé (optionnel)
            emotional_state: État émotionnel de l'assistant (optionnel)
            
        Returns:
            ID du message créé
        """
        # Valider le type de message
        if message_type not in ['user', 'assistant']:
            raise ValueError(f"Invalid message type: {message_type}")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Récupérer l'ID de conversation
        cursor.execute(
            'SELECT id FROM conversations WHERE session_id = ? AND user_id = ?',
            (session_id, user_id)
        )
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            raise ValueError(f"Conversation not found: {session_id}")
            
        conversation_id = result[0]
        
        # Mettre à jour la date de dernière mise à jour
        cursor.execute(
            'UPDATE conversations SET last_updated = CURRENT_TIMESTAMP WHERE id = ?',
            (conversation_id,)
        )
        
        # Préparer les métadonnées du message
        has_image = bool(image_data)
        has_file = bool(file_id)
        
        # Préparer l'état émotionnel
        emotional_state_json = json.dumps(emotional_state) if emotional_state else None
        
        # Insérer le message
        cursor.execute(
            '''INSERT INTO conversation_messages 
               (conversation_id, message_type, content, has_image, has_file, emotional_state)
               VALUES (?, ?, ?, ?, ?, ?)''',
            (conversation_id, message_type, content, has_image, has_file, emotional_state_json)
        )
        
        message_id = cursor.lastrowid
        
        # Traiter l'image si présente
        if image_data:
            image_path = self._save_image(image_data, session_id, message_id)
            
            cursor.execute(
                '''INSERT INTO message_images 
                   (message_id, image_path, content_type)
                   VALUES (?, ?, ?)''',
                (message_id, image_path, 'image/jpeg')  # Suppose que c'est un JPEG, à adapter si nécessaire
            )
        
        # Associer le fichier si présent
        if file_id:
            cursor.execute(
                'INSERT INTO message_files (message_id, file_id) VALUES (?, ?)',
                (message_id, file_id)
            )
        
        conn.commit()
        conn.close()
        
        # Sauvegarde parallèle dans le fichier texte
        if self.text_memory_enabled:
            # Récupérer le titre pour la première sauvegarde
            title = None
            if message_id == 1:  # Premier message de la conversation
                title_conn = sqlite3.connect(self.db_path)
                title_cursor = title_conn.cursor()
                title_cursor.execute(
                    'SELECT title FROM conversations WHERE session_id = ? AND user_id = ?',
                    (session_id, user_id)
                )
                result = title_cursor.fetchone()
                if result:
                    title = result[0]
                title_conn.close()
            
            # Récupérer le chemin de l'image si elle a été sauvegardée
            img_path = None
            if has_image and image_data:
                # Image déjà sauvegardée par _save_image, récupérer le chemin
                img_conn = sqlite3.connect(self.db_path)
                img_cursor = img_conn.cursor()
                img_cursor.execute(
                    'SELECT image_path FROM message_images WHERE message_id = ?',
                    (message_id,)
                )
                img_result = img_cursor.fetchone()
                if img_result:
                    img_path = img_result[0]
                img_conn.close()
            
            # Sauvegarder dans le fichier texte
            self.save_to_text_file(
                user_id=user_id,
                session_id=session_id,
                message_type=message_type,
                content=content,
                image_path=img_path,
                title=title
            )
            
            # Sauvegarder également dans le dossier d'uploads si c'est une image
            if has_image and image_data and self.upload_folder_enabled:
                upload_filename = f"{session_id}_{message_id}.jpg"
                self.save_uploaded_image(user_id, image_data, upload_filename)
        
        logger.info(f"Added {message_type} message to conversation {session_id}")
        return message_id
    
    def _save_image(self, image_data: str, session_id: str, message_id: int) -> str:
        """
        Sauvegarde une image sur le disque.
        
        Args:
            image_data: Image en format base64
            session_id: ID de la session
            message_id: ID du message
            
        Returns:
            Chemin de l'image sauvegardée
        """
        # Créer un dossier pour la session si nécessaire
        session_dir = os.path.join(IMAGES_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Générer un nom de fichier unique
        image_filename = f"msg_{message_id}_{int(datetime.datetime.now().timestamp())}.jpg"
        image_path = os.path.join(session_dir, image_filename)
        
        try:
            # Extraire les données binaires de l'image (retirer le préfixe data:image/xxx;base64,)
            if ',' in image_data:
                image_data = image_data.split(',', 1)[1]
            
            # Décoder et sauvegarder l'image
            with open(image_path, 'wb') as f:
                f.write(base64.b64decode(image_data))
                
            return os.path.relpath(image_path, os.path.dirname(os.path.abspath(__file__)))
            
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            return ""
    
    def get_conversation_history(self, session_id: str, user_id: int, limit: int = 50) -> List[Dict]:
        """
        Récupère l'historique d'une conversation.
        
        Args:
            session_id: ID de la session
            user_id: ID de l'utilisateur
            limit: Nombre maximum de messages à récupérer
            
        Returns:
            Liste des messages de la conversation
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Pour avoir les résultats sous forme de dictionnaires
        cursor = conn.cursor()
        
        # Récupérer l'ID de conversation
        cursor.execute(
            'SELECT id FROM conversations WHERE session_id = ? AND user_id = ?',
            (session_id, user_id)
        )
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            raise ValueError(f"Conversation not found: {session_id}")
            
        conversation_id = result['id']
        
        # Récupérer les messages
        cursor.execute(
            '''SELECT id, message_type, content, timestamp, has_image, has_file, emotional_state
               FROM conversation_messages 
               WHERE conversation_id = ?
               ORDER BY timestamp DESC LIMIT ?''',
            (conversation_id, limit)
        )
        
        messages = []
        for row in cursor.fetchall():
            message = dict(row)
            
            # Transformer l'état émotionnel en dictionnaire
            if message['emotional_state']:
                try:
                    message['emotional_state'] = json.loads(message['emotional_state'])
                except:
                    message['emotional_state'] = None
                    
            # Récupérer l'image si présente
            if message['has_image']:
                cursor.execute(
                    'SELECT image_path FROM message_images WHERE message_id = ?',
                    (message['id'],)
                )
                image_result = cursor.fetchone()
                if image_result:
                    message['image_path'] = image_result['image_path']
            
            # Récupérer les informations du fichier si présent
            if message['has_file']:
                cursor.execute(
                    '''SELECT f.id, f.original_filename, f.file_type, f.file_size
                       FROM message_files mf
                       JOIN uploaded_files f ON mf.file_id = f.id
                       WHERE mf.message_id = ?''',
                    (message['id'],)
                )
                file_result = cursor.fetchone()
                if file_result:
                    message['file'] = dict(file_result)
            
            messages.append(message)
        
        conn.close()
        
        # Inverser pour avoir l'ordre chronologique
        return messages[::-1]
    
    def get_user_conversations(self, user_id: int, limit: int = 20) -> List[Dict]:
        """
        Récupère la liste des conversations d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            limit: Nombre maximum de conversations à récupérer
            
        Returns:
            Liste des conversations
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            '''SELECT id, session_id, title, created_at, last_updated
               FROM conversations
               WHERE user_id = ?
               ORDER BY last_updated DESC LIMIT ?''',
            (user_id, limit)
        )
        
        conversations = [dict(row) for row in cursor.fetchall()]
        
        # Récupérer le dernier message pour chaque conversation
        for conversation in conversations:
            cursor.execute(
                '''SELECT content, message_type
                   FROM conversation_messages
                   WHERE conversation_id = ?
                   ORDER BY timestamp DESC LIMIT 1''',
                (conversation['id'],)
            )
            last_message = cursor.fetchone()
            if last_message:
                conversation['last_message'] = dict(last_message)
            
            # Compter le nombre de messages
            cursor.execute(
                'SELECT COUNT(*) as count FROM conversation_messages WHERE conversation_id = ?',
                (conversation['id'],)
            )
            count_result = cursor.fetchone()
            conversation['message_count'] = count_result['count'] if count_result else 0
        
        conn.close()
        return conversations
    
    def update_conversation_title(self, session_id: str, user_id: int, title: str) -> bool:
        """
        Met à jour le titre d'une conversation.
        
        Args:
            session_id: ID de la session
            user_id: ID de l'utilisateur
            title: Nouveau titre
            
        Returns:
            True si la mise à jour a réussi, False sinon
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'UPDATE conversations SET title = ? WHERE session_id = ? AND user_id = ?',
            (title, session_id, user_id)
        )
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        # Mettre à jour le titre dans le fichier texte si la fonctionnalité est activée
        if success and self.text_memory_enabled:
            file_path = os.path.join(TextMemoryManager.get_user_dir(user_id), f"{session_id}.txt")
            if os.path.exists(file_path):
                try:
                    # Lire le contenu du fichier
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Remplacer la première ligne (titre)
                    if lines and lines[0].startswith('# '):
                        lines[0] = f"# {title}\n"
                        
                        # Réécrire le fichier
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.writelines(lines)
                except Exception as e:
                    logger.error(f"Erreur lors de la mise à jour du titre dans le fichier texte: {e}")
        
        return success
    
    def delete_conversation(self, session_id: str, user_id: int) -> bool:
        """
        Supprime une conversation et tous ses messages.
        
        Args:
            session_id: ID de la session
            user_id: ID de l'utilisateur
            
        Returns:
            True si la suppression a réussi, False sinon
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Récupérer l'ID de conversation
        cursor.execute(
            'SELECT id FROM conversations WHERE session_id = ? AND user_id = ?',
            (session_id, user_id)
        )
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return False
            
        conversation_id = result[0]
        
        try:
            # Supprimer les images liées aux messages
            cursor.execute(
                '''SELECT mi.image_path 
                   FROM message_images mi
                   JOIN conversation_messages cm ON mi.message_id = cm.id
                   WHERE cm.conversation_id = ?''',
                (conversation_id,)
            )
            
            for (image_path,) in cursor.fetchall():
                try:
                    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), image_path)
                    if os.path.exists(full_path):
                        os.remove(full_path)
                except Exception as e:
                    logger.warning(f"Failed to delete image {image_path}: {str(e)}")
            
            # Utiliser une transaction pour assurer l'intégrité
            cursor.execute('BEGIN TRANSACTION')
            
            # Supprimer les références aux images
            cursor.execute(
                '''DELETE FROM message_images 
                   WHERE message_id IN (
                       SELECT id FROM conversation_messages WHERE conversation_id = ?
                   )''',
                (conversation_id,)
            )
            
            # Supprimer les références aux fichiers
            cursor.execute(
                '''DELETE FROM message_files 
                   WHERE message_id IN (
                       SELECT id FROM conversation_messages WHERE conversation_id = ?
                   )''',
                (conversation_id,)
            )
            
            # Supprimer les messages
            cursor.execute(
                'DELETE FROM conversation_messages WHERE conversation_id = ?',
                (conversation_id,)
            )
            
            # Supprimer la conversation
            cursor.execute(
                'DELETE FROM conversations WHERE id = ?',
                (conversation_id,)
            )
            
            cursor.execute('COMMIT')
            conn.close()
            
            # Nettoyer le dossier d'images si vide
            session_dir = os.path.join(IMAGES_DIR, session_id)
            if os.path.exists(session_dir) and not os.listdir(session_dir):
                os.rmdir(session_dir)
            
            # Supprimer également le fichier texte si la fonctionnalité est activée
            if self.text_memory_enabled:
                self.delete_text_conversation(user_id, session_id)
                
            logger.info(f"Deleted conversation {session_id} for user {user_id}")
            return True
            
        except Exception as e:
            cursor.execute('ROLLBACK')
            conn.close()
            logger.error(f"Error deleting conversation {session_id}: {str(e)}")
            return False
    
    def search_conversations(self, user_id: int, query: str, limit: int = 20) -> List[Dict]:
        """
        Recherche dans les conversations d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            query: Texte à rechercher
            limit: Nombre maximum de résultats
            
        Returns:
            Liste des conversations correspondantes
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Recherche dans les titres et le contenu des messages
        cursor.execute(
            '''SELECT DISTINCT c.id, c.session_id, c.title, c.created_at, c.last_updated
               FROM conversations c
               JOIN conversation_messages m ON c.id = m.conversation_id
               WHERE c.user_id = ? AND (c.title LIKE ? OR m.content LIKE ?)
               ORDER BY c.last_updated DESC LIMIT ?''',
            (user_id, f'%{query}%', f'%{query}%', limit)
        )
        
        conversations = [dict(row) for row in cursor.fetchall()]
        
        # Récupérer les extraits pertinents pour chaque conversation
        for conversation in conversations:
            cursor.execute(
                '''SELECT id, content, message_type
                   FROM conversation_messages
                   WHERE conversation_id = ? AND content LIKE ?
                   ORDER BY timestamp DESC LIMIT 3''',
                (conversation['id'], f'%{query}%')
            )
            
            matching_messages = [dict(row) for row in cursor.fetchall()]
            conversation['matching_messages'] = matching_messages
            
            # Compter le nombre total de messages
            cursor.execute(
                'SELECT COUNT(*) as count FROM conversation_messages WHERE conversation_id = ?',
                (conversation['id'],)
            )
            count_result = cursor.fetchone()
            conversation['message_count'] = count_result['count'] if count_result else 0
            
        conn.close()
        return conversations
    
    def get_context_for_gemini(self, session_id: str, user_id: int, max_messages: int = 10) -> str:
        """
        Construit un contexte pour Gemini à partir de l'historique de conversation.
        
        Args:
            session_id: ID de la session
            user_id: ID de l'utilisateur
            max_messages: Nombre maximum de messages à inclure
            
        Returns:
            Contexte formaté pour Gemini
        """
        messages = self.get_conversation_history(session_id, user_id, max_messages)
        
        if not messages:
            return ""
        
        context = ["Voici le contexte des messages précédents de cette conversation:"]
        
        for msg in messages:
            if msg['message_type'] == 'user':
                prefix = "Utilisateur: "
            else:
                prefix = "Gemini: "
                
            # Ajouter le contenu textuel
            text = msg['content'] or ""
            if text:
                context.append(f"{prefix}{text}")
            
            # Mentionner les images et fichiers mais pas leur contenu
            if msg.get('image_path'):
                context.append(f"{prefix}[a partagé une image]")
            if msg.get('file'):
                context.append(f"{prefix}[a partagé un fichier: {msg['file']['original_filename']}]")
        
        return "\n\n".join(context)
    
    def cleanup_old_conversations(self, days_threshold: int = 30) -> int:
        """
        Nettoie les conversations inactives depuis un certain temps.
        
        Args:
            days_threshold: Nombre de jours après lesquels une conversation est considérée vieille
            
        Returns:
            Nombre de conversations supprimées
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trouver les conversations inactives
        cursor.execute(
            '''SELECT session_id, user_id FROM conversations
               WHERE datetime(last_updated) < datetime('now', ?)''',
            (f'-{days_threshold} days',)
        )
        
        old_conversations = cursor.fetchall()
        conn.close()
        
        # Supprimer chaque conversation
        deleted_count = 0
        for session_id, user_id in old_conversations:
            if self.delete_conversation(session_id, user_id):
                deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old conversations")
        return deleted_count
    
    def store_conversation(self, 
                           session_id: str, 
                           user_id: int, 
                           content: str, 
                           memory_level: str = "MEDIUM_TERM",
                           metadata: Dict[str, Any] = None) -> int:
        """
        Stocke une conversation dans la mémoire.
        
        Args:
            session_id: ID de la session de conversation
            user_id: ID de l'utilisateur
            content: Contenu à mémoriser
            memory_level: Niveau de rétention (SHORT_TERM, MEDIUM_TERM, LONG_TERM, PERMANENT)
            metadata: Métadonnées supplémentaires en format dictionnaire
            
        Returns:
            ID de l'enregistrement créé
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convertir les métadonnées en JSON si présentes
        metadata_json = None
        if metadata:
            metadata_json = json.dumps(metadata)
        
        cursor.execute('''
        INSERT INTO conversation_memory 
        (session_id, user_id, content, memory_level, metadata, timestamp) 
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (session_id, user_id, content, memory_level, metadata_json))
        
        memory_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return memory_id
        
    def get_recent_conversations(self, 
                                 user_id: int, 
                                 session_id: Optional[str] = None,
                                 limit: int = 10,
                                 include_time_context: bool = True) -> List[Dict[str, Any]]:
        """
        Récupère les conversations récentes d'un utilisateur, en tenant compte
        de la durée de rétention configurée.
        
        Args:
            user_id: ID de l'utilisateur
            session_id: Optionnellement filtrer par session spécifique
            limit: Nombre maximal de conversations à récupérer
            include_time_context: Ajouter des informations sur le temps écoulé
            
        Returns:
            Liste des conversations récentes
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Pour accéder aux colonnes par nom
        cursor = conn.cursor()
        
        # Construire la requête SQL en fonction des paramètres
        sql = '''
        SELECT id, session_id, user_id, timestamp, content, memory_level, metadata
        FROM conversation_memory
        WHERE user_id = ?
        '''
        params = [user_id]
        
        if session_id:
            sql += ' AND session_id = ?'
            params.append(session_id)
            
        sql += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        
        # Convertir les résultats en liste de dictionnaires
        result = []
        current_time = datetime.datetime.now(pytz.utc)
        
        for row in rows:
            # Convertir la chaîne de timestamp en objet datetime
            timestamp_str = row['timestamp']
            if isinstance(timestamp_str, str):
                timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                timestamp = pytz.utc.localize(timestamp)
            else:
                # Si c'est déjà un datetime ou autre format
                timestamp = datetime.datetime.utcfromtimestamp(timestamp_str)
                timestamp = pytz.utc.localize(timestamp)
                
            # Vérifier si cette conversation doit être mémorisée en fonction du niveau de rétention
            memory_level = row['memory_level']
            if not should_remember_conversation(timestamp, memory_level):
                continue
                
            # Créer l'entrée pour cette conversation
            conversation = dict(row)
            
            # Deserialiser les métadonnées JSON
            if conversation['metadata']:
                try:
                    conversation['metadata'] = json.loads(conversation['metadata'])
                except:
                    conversation['metadata'] = {}
                    
            # Ajouter des informations de contexte temporel si demandé
            if include_time_context:
                conversation['time_ago'] = timestamp_to_readable_time_diff(timestamp, user_id)
                conversation['seconds_ago'] = (current_time - timestamp).total_seconds()
                
            result.append(conversation)
        
        conn.close()
        
        return result
    
    def store_text_memory(self, 
                          session_id: str, 
                          user_id: int, 
                          content: str, 
                          metadata: Dict[str, Any] = None) -> str:
        """
        Stocke un contenu textuel dans un fichier.
        
        Args:
            session_id: ID de la session
            user_id: ID de l'utilisateur
            content: Contenu textuel à stocker
            metadata: Métadonnées supplémentaires (optionnel)
            
        Returns:
            Chemin du fichier texte créé
        """
        if not self.text_memory_enabled:
            raise RuntimeError("Text memory storage is disabled.")
        
        # Créer un dossier pour la session si nécessaire
        session_dir = os.path.join(IMAGES_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Générer un nom de fichier unique
        file_name = f"memory_{user_id}_{int(datetime.datetime.now().timestamp())}.txt"
        file_path = os.path.join(session_dir, file_name)
        
        try:
            # Sauvegarder le contenu dans le fichier
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            logger.info(f"Stored text memory for user {user_id} in file {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error storing text memory: {str(e)}")
            return ""
    
    def enable_text_memory(self, enabled: bool = True):
        """Active ou désactive la mémoire textuelle."""
        self.text_memory_enabled = enabled
    
    def is_text_memory_enabled(self) -> bool:
        """Vérifie si la mémoire textuelle est activée."""
        return self.text_memory_enabled
    
    def upload_file(self, user_id: int, file_data: bytes, file_name: str, metadata: Dict[str, Any] = None) -> int:
        """
        Télécharge un fichier et l'associe à un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            file_data: Données du fichier
            file_name: Nom original du fichier
            metadata: Métadonnées supplémentaires (optionnel)
            
        Returns:
            ID du fichier téléchargé
        """
        if not self.upload_folder_enabled:
            raise RuntimeError("File upload is disabled.")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insérer le fichier dans la table des fichiers téléchargés
        cursor.execute('''
        INSERT INTO uploaded_files (user_id, original_filename, file_type, file_size, metadata, created_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''', (user_id, file_name, 'text/plain', len(file_data), json.dumps(metadata) if metadata else None))
        
        file_id = cursor.lastrowid
        
        # Sauvegarder le fichier sur le disque
        user_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'uploads', str(user_id))
        os.makedirs(user_dir, exist_ok=True)
        
        file_path = os.path.join(user_dir, f"{file_id}_{file_name}")
        with open(file_path, 'wb') as f:
            f.write(file_data)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Uploaded file {file_name} for user {user_id}, file_id: {file_id}")
        return file_id
    
    def enable_upload_folder(self, enabled: bool = True):
        """Active ou désactive le dossier d'uploads."""
        self.upload_folder_enabled = enabled
    
    def is_upload_folder_enabled(self) -> bool:
        """Vérifie si le dossier d'uploads est activé."""
        return self.upload_folder_enabled
    
    def search_files(self, user_id: int, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Recherche des fichiers téléchargés par un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            query: Texte à rechercher dans les métadonnées
            limit: Nombre maximum de résultats
            
        Returns:
            Liste des fichiers correspondants
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            '''SELECT id, original_filename, file_type, file_size, created_at, metadata
               FROM uploaded_files
               WHERE user_id = ? AND (original_filename LIKE ? OR metadata LIKE ?)
               ORDER BY created_at DESC LIMIT ?''',
            (user_id, f'%{query}%', f'%{query}%', limit)
        )
        
        files = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return files
    
    def get_file_metadata(self, file_id: int) -> Dict[str, Any]:
        """
        Récupère les métadonnées d'un fichier.
        
        Args:
            file_id: ID du fichier
            
        Returns:
            Dictionnaire des métadonnées du fichier
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT id, original_filename, file_type, file_size, created_at, metadata FROM uploaded_files WHERE id = ?',
            (file_id,)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        return dict(result) if result else {}
    
    def delete_file(self, file_id: int) -> bool:
        """
        Supprime un fichier téléchargé.
        
        Args:
            file_id: ID du fichier
            
        Returns:
            True si la suppression a réussi, False sinon
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Récupérer les informations du fichier
        cursor.execute(
            'SELECT original_filename, user_id FROM uploaded_files WHERE id = ?',
            (file_id,)
        )
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return False
            
        original_filename, user_id = result
        
        try:
            # Supprimer le fichier du disque
            file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'uploads', str(user_id), f"{file_id}_{original_filename}")
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Supprimer l'enregistrement de la base de données
            cursor.execute('DELETE FROM uploaded_files WHERE id = ?', (file_id,))
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file {file_id}: {str(e)}")
            return False
        finally:
            conn.close()
    
    def clear_uploads(self, user_id: int) -> int:
        """
        Supprime tous les fichiers téléchargés par un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Nombre de fichiers supprimés
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Récupérer les fichiers de l'utilisateur
        cursor.execute(
            'SELECT id, original_filename FROM uploaded_files WHERE user_id = ?',
            (user_id,)
        )
        files = cursor.fetchall()
        
        # Supprimer les fichiers du disque
        deleted_count = 0
        for file_id, original_filename in files:
            try:
                file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'uploads', str(user_id), f"{file_id}_{original_filename}")
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete file {original_filename}: {str(e)}")
        
        # Supprimer les enregistrements de la base de données
        cursor.execute('DELETE FROM uploaded_files WHERE user_id = ?', (user_id,))
        conn.commit()
        conn.close()
        
        logger.info(f"Cleared {deleted_count} uploaded files for user {user_id}")
        return deleted_count
    
    def search_memories(self, 
                       user_id: int, 
                       query: str, 
                       search_long_term: bool = True,
                       search_conversations: bool = True,
                       limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Recherche dans les souvenirs d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            query: Texte à rechercher
            search_long_term: Rechercher dans les souvenirs à long terme
            search_conversations: Rechercher dans les conversations
            limit: Nombre maximal de résultats par catégorie
            
        Returns:
            Dictionnaire avec les résultats par catégorie
        """
        results = {}
        
        if search_conversations:
            # Recherche dans les conversations avec une recherche de texte simple
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT id, session_id, user_id, timestamp, content, memory_level, metadata
            FROM conversation_memory
            WHERE user_id = ? AND content LIKE ?
            ORDER BY timestamp DESC LIMIT ?
            ''', (user_id, f'%{query}%', limit))
            
            rows = cursor.fetchall()
            
            # Traiter les résultats comme dans get_recent_conversations
            conversations = []
            current_time = datetime.datetime.now(pytz.utc)
            
            for row in rows:
                # Convertir timestamp
                timestamp_str = row['timestamp']
                if isinstance(timestamp_str, str):
                    timestamp = datetime.datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    timestamp = pytz.utc.localize(timestamp)
                else:
                    timestamp = datetime.datetime.utcfromtimestamp(timestamp_str)
                    timestamp = pytz.utc.localize(timestamp)
                    
                # Vérifier la rétention
                memory_level = row['memory_level']
                if not should_remember_conversation(timestamp, memory_level):
                    continue
                    
                conversation = dict(row)
                
                # Deserialiser les métadonnées
                if conversation['metadata']:
                    try:
                        conversation['metadata'] = json.loads(conversation['metadata'])
                    except:
                        conversation['metadata'] = {}
                        
                conversation['time_ago'] = timestamp_to_readable_time_diff(timestamp, user_id)
                conversation['seconds_ago'] = (current_time - timestamp).total_seconds()
                    
                conversations.append(conversation)
                
            conn.close()
            results['conversations'] = conversations
            
        if search_long_term:
            # Recherche dans les souvenirs à long terme
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT id, user_id, category, content, importance, created_at, last_accessed
            FROM long_term_memory
            WHERE user_id = ? AND content LIKE ?
            ORDER BY importance DESC, last_accessed DESC LIMIT ?
            ''', (user_id, f'%{query}%', limit))
            
            rows = cursor.fetchall()
            
            # Mettre à jour la date de dernier accès
            if rows:
                memory_ids = [row['id'] for row in rows]
                placeholders = ','.join(['?'] * len(memory_ids))
                cursor.execute(f'''
                UPDATE long_term_memory
                SET last_accessed = CURRENT_TIMESTAMP
                WHERE id IN ({placeholders})
                ''', memory_ids)
                conn.commit()
            
            long_term = [dict(row) for row in rows]
            conn.close()
            results['long_term'] = long_term
            
        return results
    
    def forget_conversation(self, memory_id: int) -> bool:
        """
        Supprime une conversation spécifique de la mémoire.
        
        Args:
            memory_id: ID de l'enregistrement à supprimer
            
        Returns:
            True si suppression réussie, False sinon
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM conversation_memory WHERE id = ?', (memory_id,))
        success = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        return success
    
    def get_memory_context(self, 
                           user_id: int, 
                           session_id: Optional[str] = None,
                           max_conversations: int = 5,
                           max_long_term: int = 3,
                           format_as_text: bool = True) -> Union[str, Dict[str, Any]]:
        """
        Génère un contexte mémoire pour être utilisé dans les interactions avec l'IA.
        
        Args:
            user_id: ID de l'utilisateur
            session_id: ID de session optionnel pour se concentrer sur une conversation
            max_conversations: Nombre max de conversations récentes à inclure
            max_long_term: Nombre max de souvenirs à long terme à inclure
            format_as_text: Formatter le résultat en texte ou renvoyer un dictionnaire
            
        Returns:
            Contexte mémoire au format texte ou dictionnaire
        """
        # Récupérer les conversations récentes
        recent = self.get_recent_conversations(
            user_id=user_id,
            session_id=session_id,
            limit=max_conversations,
            include_time_context=True
        )
        
        # Récupérer les souvenirs à long terme les plus importants
        long_term = self.get_long_term_memories(
            user_id=user_id,
            limit=max_long_term
        )
        
        if format_as_text:
            # Formatter en texte pour l'IA
            context = "Mémoire des conversations récentes:\n"
            
            if recent:
                for i, conv in enumerate(recent):
                    context += f"{i+1}. {conv['time_ago']}: {conv['content']}\n"
            else:
                context += "Aucune conversation récente mémorisée.\n"
                
            context += "\nInformations importantes:\n"
            
            if long_term:
                for i, memory in enumerate(long_term):
                    context += f"{i+1}. {memory['category']}: {memory['content']}\n"
            else:
                context += "Aucune information importante mémorisée.\n"
                
            return context
        else:
            # Renvoyer un dictionnaire structuré
            return {
                "recent_conversations": recent,
                "long_term_memories": long_term
            }
    
    def update_memory_importance(self, memory_id: int, importance: int) -> bool:
        """
        Met à jour l'importance d'un souvenir à long terme.
        
        Args:
            memory_id: ID du souvenir
            importance: Nouvelle valeur d'importance (1-10)
            
        Returns:
            True si mise à jour réussie, False sinon
        """
        importance = max(1, min(10, importance))  # Limiter entre 1 et 10
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        UPDATE long_term_memory
        SET importance = ?
        WHERE id = ?
        ''', (importance, memory_id))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success
    
    def clear_expired_memories(self) -> int:
        """
        Supprime les souvenirs expirés selon leur niveau de rétention.
        
        Returns:
            Nombre de souvenirs supprimés
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculer les timestamps de coupure pour chaque niveau
        now = datetime.datetime.now()
        short_term_cutoff = now - datetime.timedelta(minutes=MEMORY_RETENTION["SHORT_TERM"])
        medium_term_cutoff = now - datetime.timedelta(minutes=MEMORY_RETENTION["MEDIUM_TERM"])
        long_term_cutoff = now - datetime.timedelta(minutes=MEMORY_RETENTION["LONG_TERM"])
        
        # Formater en chaîne SQLite
        short_term_cutoff_str = short_term_cutoff.strftime('%Y-%m-%d %H:%M:%S')
        medium_term_cutoff_str = medium_term_cutoff.strftime('%Y-%m-%d %H:%M:%S')
        long_term_cutoff_str = long_term_cutoff.strftime('%Y-%m-%d %H:%M:%S')
        
        # Supprimer les souvenirs expirés selon leur niveau
        cursor.execute('''
        DELETE FROM conversation_memory 
        WHERE (memory_level = 'SHORT_TERM' AND timestamp < ?) 
           OR (memory_level = 'MEDIUM_TERM' AND timestamp < ?)
           OR (memory_level = 'LONG_TERM' AND timestamp < ?)
        ''', (short_term_cutoff_str, medium_term_cutoff_str, long_term_cutoff_str))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted_count
    
    def save_to_text_file(self, 
                      user_id: int,
                      session_id: str,
                      message_type: str,
                      content: str,
                      image_path: Optional[str] = None,
                      title: Optional[str] = None) -> bool:
        """
        Sauvegarde un message dans un fichier texte en plus de la base de données.
        
        Args:
            user_id: ID de l'utilisateur
            session_id: ID de la session
            message_type: Type de message ('user' ou 'assistant')
            content: Contenu du message
            image_path: Chemin de l'image (optionnel)
            title: Titre de la conversation (optionnel)
            
        Returns:
            True si le message a été sauvegardé avec succès
        """
        if not self.text_memory_enabled:
            return False
            
        return TextMemoryManager.save_message(
            user_id=user_id,
            session_id=session_id,
            message_type=message_type,
            content=content,
            image_path=image_path,
            title=title
        )
    
    def get_text_conversation(self, user_id: int, session_id: str) -> Optional[str]:
        """
        Récupère une conversation à partir d'un fichier texte.
        
        Args:
            user_id: ID de l'utilisateur
            session_id: ID de la session
            
        Returns:
            Contenu de la conversation ou None si le fichier n'existe pas
        """
        if not self.text_memory_enabled:
            return None
            
        return TextMemoryManager.read_conversation(user_id, session_id)
    
    def list_text_conversations(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Liste toutes les conversations textuelles d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Liste des conversations textuelles de l'utilisateur
        """
        if not self.text_memory_enabled:
            return []
            
        return TextMemoryManager.list_conversations(user_id)
    
    def search_text_conversations(self, user_id: int, query: str) -> List[Dict[str, Any]]:
        """
        Recherche dans les conversations textuelles d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            query: Texte à rechercher
            
        Returns:
            Liste des conversations textuelles contenant la requête
        """
        if not self.text_memory_enabled:
            return []
            
        return TextMemoryManager.search_conversations(user_id, query)
    
    def delete_text_conversation(self, user_id: int, session_id: str) -> bool:
        """
        Supprime une conversation textuelle.
        
        Args:
            user_id: ID de l'utilisateur
            session_id: ID de la session
            
        Returns:
            True si la suppression a réussi
        """
        if not self.text_memory_enabled:
            return False
            
        return TextMemoryManager.delete_conversation(user_id, session_id)
    
    def save_uploaded_image(self, user_id: int, image_data: str, filename: Optional[str] = None) -> Optional[str]:
        """
        Sauvegarde une image téléchargée.
        
        Args:
            user_id: ID de l'utilisateur
            image_data: Données de l'image en base64
            filename: Nom de fichier (optionnel)
            
        Returns:
            Chemin relatif de l'image sauvegardée ou None en cas d'erreur
        """
        if not self.upload_folder_enabled:
            return None
            
        return TextMemoryManager.save_uploaded_image(user_id, image_data, filename)
    
    def list_uploaded_images(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Liste les images téléchargées par un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Liste des images téléchargées
        """
        if not self.upload_folder_enabled:
            return []
            
        return TextMemoryManager.list_uploaded_images(user_id)
