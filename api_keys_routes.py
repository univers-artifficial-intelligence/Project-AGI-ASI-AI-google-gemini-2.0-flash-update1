"""
Routes API pour configurer et gérer les clés API des différentes intégrations.
"""

from flask import Blueprint, request, jsonify, session
from ai_api_manager import get_ai_api_manager
import logging
import json
import os

# Configuration du logger
logger = logging.getLogger(__name__)

# Créer un Blueprint Flask
api_keys_bp = Blueprint('api_keys', __name__)

# Chemin vers le fichier des clés API
API_KEYS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_keys.json')

def load_api_keys():
    """Charge les clés API depuis un fichier JSON."""
    try:
        if os.path.exists(API_KEYS_PATH):
            with open(API_KEYS_PATH, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Erreur lors du chargement des clés API: {str(e)}")
        return {}

def save_api_keys(api_keys):
    """Sauvegarde les clés API dans un fichier JSON."""
    try:
        with open(API_KEYS_PATH, 'w') as f:
            json.dump(api_keys, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des clés API: {str(e)}")
        return False

@api_keys_bp.route('/api/keys', methods=['GET'])
def get_api_keys():
    """Récupère la liste des clés API configurées."""
    if not session.get('logged_in'):
        return jsonify({'error': 'Authentication required'}), 401
    
    try:
        api_keys = load_api_keys()
        
        # Par mesure de sécurité, ne pas renvoyer les clés complètes
        secured_keys = {}
        for api_name, key in api_keys.items():
            # Masquer la clé, ne montrer que les 4 derniers caractères
            if key and len(key) > 4:
                secured_keys[api_name] = '•' * (len(key) - 4) + key[-4:]
            else:
                secured_keys[api_name] = None
        
        return jsonify({'api_keys': secured_keys})
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des clés API: {str(e)}")
        return jsonify({'error': 'Une erreur est survenue'}), 500

@api_keys_bp.route('/api/keys/<api_name>', methods=['POST'])
def set_api_key(api_name):
    """Définit ou met à jour la clé API pour une API spécifique."""
    if not session.get('logged_in'):
        return jsonify({'error': 'Authentication required'}), 401
    
    try:
        data = request.json
        api_key = data.get('api_key')
        
        if not api_key:
            return jsonify({'error': 'Clé API manquante'}), 400
        
        # Vérifier que l'API existe
        api_manager = get_ai_api_manager()
        available_apis = api_manager.list_available_apis()
        
        if api_name not in available_apis:
            return jsonify({'error': f'API "{api_name}" non disponible'}), 404
        
        # Charger les clés existantes
        api_keys = load_api_keys()
        
        # Mettre à jour la clé
        api_keys[api_name] = api_key
        
        # Sauvegarder les clés
        if save_api_keys(api_keys):
            # Mettre à jour la configuration de l'API
            api_manager.configure_api(api_name, {'api_key': api_key})
            
            return jsonify({
                'success': True, 
                'message': f'Clé API pour "{api_name}" mise à jour'
            })
        else:
            return jsonify({'error': 'Impossible de sauvegarder la clé API'}), 500
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour de la clé API pour {api_name}: {str(e)}")
        return jsonify({'error': 'Une erreur est survenue'}), 500

@api_keys_bp.route('/api/keys/<api_name>', methods=['DELETE'])
def delete_api_key(api_name):
    """Supprime la clé API pour une API spécifique."""
    if not session.get('logged_in'):
        return jsonify({'error': 'Authentication required'}), 401
    
    try:
        # Vérifier que l'API existe
        api_manager = get_ai_api_manager()
        available_apis = api_manager.list_available_apis()
        
        if api_name not in available_apis:
            return jsonify({'error': f'API "{api_name}" non disponible'}), 404
        
        # Charger les clés existantes
        api_keys = load_api_keys()
        
        # Supprimer la clé si elle existe
        if api_name in api_keys:
            del api_keys[api_name]
            
            # Sauvegarder les clés
            if save_api_keys(api_keys):
                # Mettre à jour la configuration de l'API
                api_manager.configure_api(api_name, {'api_key': None})
                
                return jsonify({
                    'success': True, 
                    'message': f'Clé API pour "{api_name}" supprimée'
                })
            else:
                return jsonify({'error': 'Impossible de sauvegarder les modifications'}), 500
        else:
            return jsonify({'error': f'Aucune clé API trouvée pour "{api_name}"'}), 404
    except Exception as e:
        logger.error(f"Erreur lors de la suppression de la clé API pour {api_name}: {str(e)}")
        return jsonify({'error': 'Une erreur est survenue'}), 500
