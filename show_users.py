#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script simple pour voir les utilisateurs de la base de données
"""

import sqlite3
import os
from tabulate import tabulate  # Ajout de tabulate pour un meilleur affichage

def show_users():
    """Affiche tous les utilisateurs"""
    db_path = 'gemini_chat.db'
    
    if not os.path.exists(db_path):
        print("Base de données non trouvée")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Vérifier si la table users existe
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not cursor.fetchone():
            print("La table 'users' n'existe pas dans la base de données.")
            return
            
        cursor.execute("SELECT id, username, email, created_at FROM users ORDER BY id")
        users = cursor.fetchall()
        
        if not users:
            print("Aucun utilisateur trouvé")
            return
        
        print("\n=== LISTE DES UTILISATEURS ===")
        
        # Utilisation de tabulate pour un affichage en tableau
        headers = ["ID", "Nom d'utilisateur", "Email", "Date de création"]
        print(tabulate(users, headers=headers, tablefmt="grid"))
        
        print(f"\nTotal: {len(users)} utilisateur(s)")
        
    except sqlite3.Error as e:
        print(f"Erreur SQLite: {e}")
    except Exception as e:
        print(f"Erreur inattendue: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            print("Connexion à la base de données fermée")

if __name__ == "__main__":
    try:
        # Vérifier si tabulate est installé
        import tabulate
    except ImportError:
        print("Le module 'tabulate' est nécessaire. Installation...")
        import subprocess
        try:
            subprocess.check_call(["pip", "install", "tabulate"])
            print("Module 'tabulate' installé avec succès!")
        except Exception as e:
            print(f"Impossible d'installer le module 'tabulate': {e}")
            print("Utilisez la commande: pip install tabulate")
            print("Affichage en mode simple...")
            
    show_users()
