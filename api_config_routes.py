"""
Route API pour configurer quelle API d'IA utiliser.
Ce module expose une API REST pour configurer l'API d'IA à utiliser.
"""

from flask import Blueprint, request, jsonify, session
from ai_api_manager import get_ai_api_manager
import logging

# Configuration du logger
logger = logging.getLogger(__name__)

# Créer un Blueprint Flask
api_config_bp = Blueprint('api_config', __name__)

@api_config_bp.route('/api/config/apis', methods=['GET'])
def get_available_apis():
    """Récupère la liste des APIs d'IA disponibles."""
    if not session.get('logged_in'):
        return jsonify({'error': 'Authentication required'}), 401
    
    try:
        api_manager = get_ai_api_manager()
        available_apis = api_manager.list_available_apis()
        current_api = api_manager.get_current_api_name()
        
        return jsonify({
            'available_apis': available_apis,
            'current_api': current_api
        })
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des APIs disponibles: {str(e)}")
        return jsonify({'error': 'Une erreur est survenue'}), 500

@api_config_bp.route('/api/config/apis/current', methods=['GET', 'POST'])
def manage_current_api():
    """Récupère ou modifie l'API d'IA active."""
    if not session.get('logged_in'):
        return jsonify({'error': 'Authentication required'}), 401
    
    api_manager = get_ai_api_manager()
    
    # Récupérer l'API active
    if request.method == 'GET':
        try:
            current_api = api_manager.get_current_api_name()
            return jsonify({'current_api': current_api})
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'API active: {str(e)}")
            return jsonify({'error': 'Une erreur est survenue'}), 500
    
    # Changer l'API active
    elif request.method == 'POST':
        try:
            data = request.json
            new_api = data.get('api_name')
            
            if not new_api:
                return jsonify({'error': 'Nom d\'API manquant'}), 400
            
            available_apis = api_manager.list_available_apis()
            if new_api not in available_apis:
                return jsonify({'error': f'API "{new_api}" non disponible'}), 400
            
            result = api_manager.set_api(new_api)
            if result:
                return jsonify({
                    'success': True, 
                    'message': f'API changée pour "{new_api}"',
                    'current_api': new_api
                })
            else:
                logger.error(f"Échec du changement d'API pour {new_api}")
                return jsonify({'error': 'Impossible de changer l\'API'}), 500
        except Exception as e:
            logger.error(f"Erreur lors du changement d'API: {str(e)}")
            return jsonify({'error': 'Une erreur est survenue'}), 500

@api_config_bp.route('/api/config/apis/<api_name>', methods=['GET', 'POST'])
def configure_api(api_name):
    """Récupère ou modifie la configuration d'une API spécifique."""
    if not session.get('logged_in'):
        return jsonify({'error': 'Authentication required'}), 401
    
    api_manager = get_ai_api_manager()
    
    # Vérifier que l'API existe
    available_apis = api_manager.list_available_apis()
    if api_name not in available_apis:
        return jsonify({'error': f'API "{api_name}" non disponible'}), 404
      # Récupérer la configuration de l'API
    if request.method == 'GET':
        try:
            # Récupérer la configuration actuelle
            config = api_manager.get_api_config(api_name) if hasattr(api_manager, 'get_api_config') else {}
            
            # Si la méthode n'existe pas, essayer d'accéder directement à la configuration
            if not config and hasattr(api_manager, 'config') and 'apis' in api_manager.config:
                config = api_manager.config['apis'].get(api_name, {})
                
            return jsonify({
                'api_name': api_name,
                'config': config
            })
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la configuration de l'API {api_name}: {str(e)}")
            return jsonify({'error': 'Une erreur est survenue'}), 500
    
    # Modifier la configuration de l'API
    elif request.method == 'POST':
        try:
            data = request.json
            api_config = data.get('config', {})
            
            result = api_manager.configure_api(api_name, api_config)
            if result:
                return jsonify({
                    'success': True, 
                    'message': f'Configuration de l\'API "{api_name}" mise à jour'
                })
            else:
                logger.error(f"Échec de la mise à jour de la configuration de l'API {api_name}")
                return jsonify({'error': 'Impossible de mettre à jour la configuration'}), 500
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de la configuration de l'API {api_name}: {str(e)}")
            return jsonify({'error': 'Une erreur est survenue'}), 500
