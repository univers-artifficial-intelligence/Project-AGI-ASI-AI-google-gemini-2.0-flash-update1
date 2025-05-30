import sqlite3
import hashlib
import os
import json
from datetime import datetime

DB_PATH = 'gemini_chat.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Création de la table utilisateurs
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Création de la table historique des chats
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        message TEXT NOT NULL,
        response TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Création de la table d'état émotionnel
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS emotional_state (
        id INTEGER PRIMARY KEY,
        base_state TEXT NOT NULL,
        joy REAL DEFAULT 0.5,
        sadness REAL DEFAULT 0.5,
        anger REAL DEFAULT 0.5,
        fear REAL DEFAULT 0.5,
        surprise REAL DEFAULT 0.5,
        disgust REAL DEFAULT 0.5,
        trust REAL DEFAULT 0.5,
        anticipation REAL DEFAULT 0.5,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Création de la table pour la mémoire à long terme (conscience)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS consciousness_memory (
        id INTEGER PRIMARY KEY,
        topic TEXT NOT NULL,
        knowledge TEXT NOT NULL,
        importance REAL DEFAULT 0.5,
        usage_count INTEGER DEFAULT 1,
        last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Création de la table pour les interactions utilisateur (contexte)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_interaction_context (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        context_data TEXT NOT NULL,
        session_id TEXT NOT NULL,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Vérifier si l'état émotionnel par défaut existe déjà
    cursor.execute("SELECT COUNT(*) FROM emotional_state")
    if cursor.fetchone()[0] == 0:
        # Insérer l'état émotionnel par défaut (neutre)
        cursor.execute('''
        INSERT INTO emotional_state (base_state, joy, sadness, anger, fear, surprise, disgust, trust, anticipation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', ('neutral', 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
        
    conn.commit()
    conn.close()

def hash_password(password):
    # Simple hash pour l'exemple - utilisez bcrypt en production
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email):
    conn = None
    try:
        # Ajouter un timeout de 20 secondes et mode EXCLUSIVE pour gérer les blocages
        conn = sqlite3.connect(DB_PATH, timeout=20.0, isolation_level="EXCLUSIVE")
        cursor = conn.cursor()
        
        hashed_password = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
            (username, hashed_password, email)
        )
        
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Utilisateur déjà existant
        return False
    except sqlite3.OperationalError as e:
        # Log de l'erreur pour le débogage
        print(f"Erreur d'opération SQLite: {e}")
        return False
    finally:
        # S'assurer que la connexion est fermée même en cas d'erreur
        if conn:
            conn.close()

def validate_login(username, password):
    """
    Valide les identifiants de connexion d'un utilisateur.
    
    Args:
        username: Nom d'utilisateur à vérifier
        password: Mot de passe à vérifier (non hashé)
        
    Returns:
        True si les identifiants sont valides, False sinon
    """
    conn = None
    try:
        # Ajouter un timeout et mode de lecture seule pour cette opération
        conn = sqlite3.connect(DB_PATH, timeout=20.0)
        cursor = conn.cursor()
        
        hashed_password = hash_password(password)
        cursor.execute(
            "SELECT id FROM users WHERE username = ? AND password = ?",
            (username, hashed_password)
        )
        
        user = cursor.fetchone()
        return user is not None
    except Exception as e:
        print(f"Erreur lors de la validation de connexion: {str(e)}")
        return False
    finally:
        # S'assurer que la connexion est fermée même en cas d'erreur
        if conn:
            conn.close()

def get_emotional_state():
    """Récupère l'état émotionnel actuel de l'IA"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM emotional_state ORDER BY last_updated DESC LIMIT 1")
    
    state = cursor.fetchone()
    conn.close()
    
    if state:
        return {
            'id': state[0],
            'base_state': state[1],
            'joy': state[2],
            'sadness': state[3],
            'anger': state[4],
            'fear': state[5],
            'surprise': state[6],
            'disgust': state[7],
            'trust': state[8],
            'anticipation': state[9],
            'last_updated': state[10]
        }
    return None

def update_emotional_state(emotions):
    """Met à jour l'état émotionnel de l'IA"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Déterminer l'état de base à partir des émotions dominantes
    emotions_values = {
        'joy': emotions.get('joy', 0.5),
        'sadness': emotions.get('sadness', 0.5),
        'anger': emotions.get('anger', 0.5),
        'fear': emotions.get('fear', 0.5),
        'surprise': emotions.get('surprise', 0.5),
        'disgust': emotions.get('disgust', 0.5),
        'trust': emotions.get('trust', 0.5),
        'anticipation': emotions.get('anticipation', 0.5)
    }
    
    # Trouver l'émotion dominante
    dominant_emotion = max(emotions_values, key=emotions_values.get)
    
    # Calculer l'état de base
    if emotions_values[dominant_emotion] > 0.7:
        base_state = dominant_emotion
    elif sum(emotions_values.values()) / len(emotions_values) < 0.4:
        base_state = 'calm'
    else:
        base_state = 'neutral'
    
    cursor.execute('''
    INSERT INTO emotional_state (base_state, joy, sadness, anger, fear, surprise, disgust, trust, anticipation)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        base_state,
        emotions_values['joy'],
        emotions_values['sadness'],
        emotions_values['anger'],
        emotions_values['fear'],
        emotions_values['surprise'],
        emotions_values['disgust'],
        emotions_values['trust'],
        emotions_values['anticipation']
    ))
    
    conn.commit()
    conn.close()

def store_memory(topic, knowledge, importance=0.5):
    """Stocke une information dans la mémoire de l'IA"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Vérifier si le sujet existe déjà
    cursor.execute("SELECT id, knowledge, importance, usage_count FROM consciousness_memory WHERE topic = ?", (topic,))
    existing = cursor.fetchone()
    
    if existing:
        try:
            # Mettre à jour la connaissance existante (avec gestion d'erreurs robuste)
            try:
                if isinstance(existing[1], str):
                    merged_knowledge = json.loads(existing[1])
                else:
                    merged_knowledge = existing[1]
                    
                # S'assurer que merged_knowledge est bien un dictionnaire
                if not isinstance(merged_knowledge, dict):
                    merged_knowledge = {"data": merged_knowledge}
            except (json.JSONDecodeError, TypeError):
                merged_knowledge = {"error_recovery": True}
            
            # Préparer new_knowledge avec une protection supplémentaire
            try:
                if isinstance(knowledge, str):
                    try:
                        new_knowledge = json.loads(knowledge)
                    except (json.JSONDecodeError, TypeError):
                        new_knowledge = {"text": knowledge}
                else:
                    new_knowledge = knowledge
                    
                # S'assurer que new_knowledge est bien un dictionnaire
                if not isinstance(new_knowledge, dict):
                    new_knowledge = {"data": new_knowledge}
            except Exception:
                new_knowledge = {"error_recovery": True}
            
            # Fusion des connaissances avec protection maximale
            for key, value in new_knowledge.items():
                if key in merged_knowledge:
                    # Triple vérification des types avant d'utiliser update()
                    if isinstance(merged_knowledge[key], dict) and isinstance(value, dict):
                        # Les deux sont des dictionnaires, fusion sécurisée
                        merged_knowledge[key] = merged_knowledge[key].copy()  # Copie pour éviter les modifications en place
                        merged_knowledge[key].update(value)
                    else:
                        # Si l'un d'eux n'est pas un dictionnaire, remplacer simplement
                        merged_knowledge[key] = value
                else:
                    merged_knowledge[key] = value
        except Exception as e:
            # En cas d'erreur critique, créer un nouvel objet plutôt que d'échouer
            import logging
            logging.error(f"Erreur lors de la fusion des connaissances: {str(e)}")
            merged_knowledge = {"error_recovery": True, "text": str(knowledge) if knowledge else ""}
        
        # Augmenter l'importance et le compteur d'utilisation
        new_importance = min(1.0, existing[2] + 0.1)
        new_count = existing[3] + 1
        
        cursor.execute('''
        UPDATE consciousness_memory 
        SET knowledge = ?, importance = ?, usage_count = ?, last_accessed = CURRENT_TIMESTAMP 
        WHERE id = ?
        ''', (json.dumps(merged_knowledge), new_importance, new_count, existing[0]))
    else:
        # Créer une nouvelle entrée
        cursor.execute('''
        INSERT INTO consciousness_memory (topic, knowledge, importance)
        VALUES (?, ?, ?)
        ''', (topic, knowledge if isinstance(knowledge, str) else json.dumps(knowledge), importance))
    
    conn.commit()
    conn.close()

def get_memories(topic=None, min_importance=0.0, limit=5):
    """Récupère des souvenirs de la mémoire de l'IA"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if topic:
        # Recherche par sujet
        cursor.execute('''
        SELECT topic, knowledge, importance FROM consciousness_memory 
        WHERE topic LIKE ? AND importance >= ? 
        ORDER BY importance DESC, last_accessed DESC LIMIT ?
        ''', (f"%{topic}%", min_importance, limit))
    else:
        # Récupérer les souvenirs les plus importants
        cursor.execute('''
        SELECT topic, knowledge, importance FROM consciousness_memory 
        WHERE importance >= ? 
        ORDER BY importance DESC, last_accessed DESC LIMIT ?
        ''', (min_importance, limit))
    
    memories = cursor.fetchall()
    conn.close()
    
    return [{'topic': mem[0], 'knowledge': json.loads(mem[1]), 'importance': mem[2]} for mem in memories]

def update_user_context(user_id, context_data, session_id):
    """Met à jour ou crée le contexte d'interaction de l'utilisateur"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Vérifier si un contexte existe déjà pour cet utilisateur et cette session
    cursor.execute("SELECT id FROM user_interaction_context WHERE user_id = ? AND session_id = ?", 
                   (user_id, session_id))
    existing = cursor.fetchone()
    
    if existing:
        cursor.execute('''
        UPDATE user_interaction_context 
        SET context_data = ?, last_updated = CURRENT_TIMESTAMP 
        WHERE id = ?
        ''', (json.dumps(context_data), existing[0]))
    else:
        cursor.execute('''
        INSERT INTO user_interaction_context (user_id, context_data, session_id)
        VALUES (?, ?, ?)
        ''', (user_id, json.dumps(context_data), session_id))
    
    conn.commit()
    conn.close()

def get_user_context(user_id, session_id):
    """Récupère le contexte d'interaction de l'utilisateur"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT context_data FROM user_interaction_context 
    WHERE user_id = ? AND session_id = ? 
    ORDER BY last_updated DESC LIMIT 1
    ''', (user_id, session_id))
    
    context = cursor.fetchone()
    conn.close()
    
    return json.loads(context[0]) if context else {}
