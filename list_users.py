#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour lister tous les utilisateurs de la base de données GeminiChat
"""

import sqlite3
import os
from datetime import datetime

# Chemin vers la base de données
DB_PATH = 'gemini_chat.db'

def list_all_users():
    """
    Affiche la liste de tous les utilisateurs enregistrés dans la base de données
    """
    if not os.path.exists(DB_PATH):
        print(f"❌ Base de données non trouvée : {DB_PATH}")
        return
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Pour avoir accès aux colonnes par nom
        cursor = conn.cursor()
        
        # Récupérer tous les utilisateurs
        cursor.execute("""
            SELECT id, username, email, created_at 
            FROM users 
            ORDER BY created_at DESC
        """)
        
        users = cursor.fetchall()
        
        if not users:
            print("📭 Aucun utilisateur trouvé dans la base de données.")
            return
        
        print("👥 LISTE DES UTILISATEURS")
        print("=" * 60)
        print(f"{'ID':<5} {'Nom d\'utilisateur':<20} {'Email':<25} {'Date création':<15}")
        print("-" * 60)
        
        for user in users:
            # Formater la date
            date_str = user['created_at']
            if date_str:
                try:
                    date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    date_formatted = date_obj.strftime('%d/%m/%Y')
                except:
                    date_formatted = date_str[:10] if len(date_str) >= 10 else date_str
            else:
                date_formatted = "N/A"
            
            print(f"{user['id']:<5} {user['username']:<20} {user['email']:<25} {date_formatted:<15}")
        
        print("-" * 60)
        print(f"📊 Total : {len(users)} utilisateur(s)")
        
    except sqlite3.Error as e:
        print(f"❌ Erreur de base de données : {e}")
    except Exception as e:
        print(f"❌ Erreur : {e}")
    finally:
        if conn:
            conn.close()

def get_user_details(user_id):
    """
    Affiche les détails d'un utilisateur spécifique
    """
    if not os.path.exists(DB_PATH):
        print(f"❌ Base de données non trouvée : {DB_PATH}")
        return
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Récupérer les informations de l'utilisateur
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            print(f"❌ Utilisateur avec ID {user_id} non trouvé.")
            return
        
        print(f"👤 DÉTAILS DE L'UTILISATEUR (ID: {user_id})")
        print("=" * 50)
        print(f"Nom d'utilisateur : {user['username']}")
        print(f"Email           : {user['email']}")
        print(f"Date création   : {user['created_at']}")
        
        # Compter les conversations de l'utilisateur
        cursor.execute("SELECT COUNT(*) FROM conversation_sessions WHERE user_id = ?", (user_id,))
        conv_count = cursor.fetchone()[0]
        print(f"Conversations   : {conv_count}")
        
        # Compter les messages de l'utilisateur
        cursor.execute("""
            SELECT COUNT(*) 
            FROM messages m 
            JOIN conversation_sessions cs ON m.session_id = cs.session_id 
            WHERE cs.user_id = ?
        """, (user_id,))
        msg_count = cursor.fetchone()[0]
        print(f"Messages        : {msg_count}")
        
    except sqlite3.Error as e:
        print(f"❌ Erreur de base de données : {e}")
    except Exception as e:
        print(f"❌ Erreur : {e}")
    finally:
        if conn:
            conn.close()

def search_users(search_term):
    """
    Recherche des utilisateurs par nom d'utilisateur ou email
    """
    if not os.path.exists(DB_PATH):
        print(f"❌ Base de données non trouvée : {DB_PATH}")
        return
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, username, email, created_at 
            FROM users 
            WHERE username LIKE ? OR email LIKE ?
            ORDER BY created_at DESC
        """, (f'%{search_term}%', f'%{search_term}%'))
        
        users = cursor.fetchall()
        
        if not users:
            print(f"🔍 Aucun utilisateur trouvé pour '{search_term}'")
            return
        
        print(f"🔍 RÉSULTATS DE RECHERCHE POUR '{search_term}'")
        print("=" * 60)
        print(f"{'ID':<5} {'Nom d\'utilisateur':<20} {'Email':<25} {'Date création':<15}")
        print("-" * 60)
        
        for user in users:
            date_str = user['created_at']
            date_formatted = date_str[:10] if date_str and len(date_str) >= 10 else "N/A"
            print(f"{user['id']:<5} {user['username']:<20} {user['email']:<25} {date_formatted:<15}")
        
        print("-" * 60)
        print(f"📊 {len(users)} résultat(s) trouvé(s)")
        
    except sqlite3.Error as e:
        print(f"❌ Erreur de base de données : {e}")
    except Exception as e:
        print(f"❌ Erreur : {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # Aucun argument - lister tous les utilisateurs
        list_all_users()
    elif len(sys.argv) == 2:
        arg = sys.argv[1]
        if arg.isdigit():
            # Argument numérique - afficher les détails de l'utilisateur
            get_user_details(int(arg))
        else:
            # Argument texte - rechercher des utilisateurs
            search_users(arg)
    else:
        print("Usage:")
        print("  python list_users.py                 # Lister tous les utilisateurs")
        print("  python list_users.py <ID>            # Détails d'un utilisateur")
        print("  python list_users.py <terme_recherche> # Rechercher des utilisateurs")
