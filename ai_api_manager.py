"""
Gestionnaire d'API d'intelligence artificielle.
Ce module permet de charger et gérer différentes implémentations d'API d'IA.
"""
import os
import json
import logging
from typing import Dict, Any, Optional, List, Type

from ai_api_interface import AIApiInterface
from gemini_api_adapter import GeminiAPI
from claude_api_adapter import ClaudeAPI
from custom_llm_adapter import CustomLLMAPI

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chemin vers le fichier de configuration des API
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ai_api_config.json')

class AIApiManager:
    """
    Gestionnaire des API d'IA qui permet de charger et utiliser différentes implémentations.
    """
    # Dictionnaire des API disponibles (nom_api -> classe d'implémentation)
    _available_apis = {
        'gemini': GeminiAPI,
        'claude': ClaudeAPI,
        'custom_llm': CustomLLMAPI
    }
    
    # Instance singleton
    _instance = None
    
    def __new__(cls):
        """Implémentation du pattern Singleton."""
        if cls._instance is None:
            cls._instance = super(AIApiManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialise le gestionnaire d'API."""
        self.default_api_name = 'gemini'
        self.active_api: Optional[AIApiInterface] = None
        self.config: Dict[str, Dict[str, Any]] = {}
        
        # Charger la configuration
        self._load_config()
        
        # Initialiser l'API par défaut
        self._set_active_api(self.default_api_name)
        
        logger.info(f"Gestionnaire d'API initialisé avec l'API par défaut: {self.default_api_name}")
    
    def _load_config(self):
        """Charge la configuration des API depuis le fichier de configuration."""
        try:
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Configuration des API chargée depuis {CONFIG_PATH}")
                
                # Définir l'API par défaut si spécifiée dans la configuration
                if 'default_api' in self.config:
                    self.default_api_name = self.config['default_api']
            else:
                # Créer une configuration par défaut
                self.config = {
                    'default_api': 'gemini',
                    'apis': {
                        'gemini': {
                            'api_key': None,  # Utiliser la clé par défaut
                            'api_url': None   # Utiliser l'URL par défaut
                        },                        'claude': {
                            'api_key': None,  # À configurer par l'utilisateur
                            'api_url': None   # Utiliser l'URL par défaut
                        },
                        'custom_llm': {
                            'api_key': None,  # À configurer par l'utilisateur
                            'api_url': None   # À configurer par l'utilisateur
                        }
                    }
                }
                self._save_config()
                logger.info("Configuration par défaut créée")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
    
    def _save_config(self):
        """Sauvegarde la configuration des API dans le fichier de configuration."""
        try:
            with open(CONFIG_PATH, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Configuration des API sauvegardée dans {CONFIG_PATH}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la configuration: {str(e)}")
    
    def _set_active_api(self, api_name: str) -> bool:
        """
        Définit l'API active par son nom.
        
        Args:
            api_name: Nom de l'API à activer
            
        Returns:
            True si l'API a été activée avec succès, False sinon
        """
        if api_name not in self._available_apis:
            logger.error(f"API '{api_name}' non disponible")
            return False
        
        try:
            # Récupérer la configuration de l'API
            api_config = {}
            if 'apis' in self.config and api_name in self.config['apis']:
                api_config = self.config['apis'][api_name]
            
            # Créer une instance de l'API
            api_class = self._available_apis[api_name]
            self.active_api = api_class(**api_config)
            logger.info(f"API '{api_name}' activée avec succès")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'activation de l'API '{api_name}': {str(e)}")
            return False
    
    def add_api_implementation(self, api_name: str, api_class: Type[AIApiInterface]) -> bool:
        """
        Ajoute une nouvelle implémentation d'API.
        
        Args:
            api_name: Nom de l'API à ajouter
            api_class: Classe d'implémentation de l'API (doit hériter de AIApiInterface)
            
        Returns:
            True si l'API a été ajoutée avec succès, False sinon
        """
        if not issubclass(api_class, AIApiInterface):
            logger.error(f"La classe '{api_class.__name__}' n'implémente pas l'interface AIApiInterface")
            return False
        
        self._available_apis[api_name] = api_class
        logger.info(f"API '{api_name}' ajoutée avec succès")
        return True
    
    def set_api(self, api_name: str) -> bool:
        """
        Change l'API active.
        
        Args:
            api_name: Nom de l'API à activer
            
        Returns:
            True si l'API a été activée avec succès, False sinon
        """
        result = self._set_active_api(api_name)
        if result:
            # Mettre à jour l'API par défaut dans la configuration
            self.config['default_api'] = api_name
            self._save_config()
        return result
    
    def list_available_apis(self) -> List[str]:
        """
        Retourne la liste des APIs disponibles.
        
        Returns:
            Liste des noms d'API disponibles
        """
        return list(self._available_apis.keys())
    
    def get_current_api_name(self) -> str:
        """
        Retourne le nom de l'API active.
        
        Returns:
            Nom de l'API active
        """
        return self.default_api_name
    
    def configure_api(self, api_name: str, config: Dict[str, Any]) -> bool:
        """
        Configure une API spécifique.
        
        Args:
            api_name: Nom de l'API à configurer
            config: Configuration de l'API (clé API, URL, etc.)
            
        Returns:
            True si la configuration a été appliquée avec succès, False sinon
        """
        if api_name not in self._available_apis:
            logger.error(f"API '{api_name}' non disponible")
            return False
        
        try:
            # Mettre à jour la configuration
            if 'apis' not in self.config:
                self.config['apis'] = {}
            self.config['apis'][api_name] = config
            self._save_config()
            
            # Si c'est l'API active, la réinitialiser avec la nouvelle configuration
            if self.default_api_name == api_name:
                self._set_active_api(api_name)
                
            logger.info(f"Configuration de l'API '{api_name}' mise à jour")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la configuration de l'API '{api_name}': {str(e)}")
            return False
    
    def get_api_config(self, api_name: str) -> Dict[str, Any]:
        """
        Récupère la configuration d'une API spécifique.
        
        Args:
            api_name: Nom de l'API dont on veut récupérer la configuration
            
        Returns:
            Configuration de l'API ou un dictionnaire vide si l'API n'existe pas
        """
        if api_name not in self._available_apis:
            logger.warning(f"API '{api_name}' non disponible, impossible de récupérer sa configuration")
            return {}
        
        try:
            # Récupérer la configuration
            if 'apis' in self.config and api_name in self.config['apis']:
                return self.config['apis'][api_name]
            else:
                return {}
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la configuration de l'API '{api_name}': {str(e)}")
            return {}
    
    def get_api_instance(self) -> Optional[AIApiInterface]:
        """
        Retourne l'instance de l'API active.
        
        Returns:
            Instance de l'API active ou None si aucune API n'est active
        """
        return self.active_api
    
    # Méthodes de délégation pour faciliter l'utilisation
    def get_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Délègue l'appel à la méthode get_response de l'API active.
        
        Args:
            prompt: Le texte de la requête
            **kwargs: Arguments supplémentaires à passer à l'API
            
        Returns:
            Dictionnaire contenant la réponse et les métadonnées
        """
        if self.active_api is None:
            logger.error("Aucune API active")
            return {
                'response': "Erreur: Aucune API d'IA n'est active.",
                'status': 'error',
                'error': 'No active API'
            }
        
        return self.active_api.get_response(prompt, **kwargs)
    
    def process_memory_request(self, prompt: str, user_id: int, session_id: str) -> Optional[str]:
        """
        Délègue l'appel à la méthode process_memory_request de l'API active.
        
        Args:
            prompt: La question ou instruction de l'utilisateur
            user_id: ID de l'utilisateur
            session_id: ID de la session actuelle
            
        Returns:
            Un contexte enrichi si la demande est liée à la mémoire, sinon None
        """
        if self.active_api is None:
            logger.error("Aucune API active")
            return None
        
        return self.active_api.process_memory_request(prompt, user_id, session_id)
    
    def get_conversation_history(self, user_id: int, session_id: str, max_messages: int = 10) -> str:
        """
        Délègue l'appel à la méthode get_conversation_history de l'API active.
        
        Args:
            user_id: ID de l'utilisateur
            session_id: ID de la session
            max_messages: Nombre maximal de messages à inclure
            
        Returns:
            Un résumé de la conversation précédente
        """
        if self.active_api is None:
            logger.error("Aucune API active")
            return ""
        
        return self.active_api.get_conversation_history(user_id, session_id, max_messages)


# Fonction globale pour obtenir l'instance du gestionnaire d'API
def get_ai_api_manager() -> AIApiManager:
    """
    Retourne l'instance singleton du gestionnaire d'API.
    
    Returns:
        Instance du gestionnaire d'API
    """
    return AIApiManager()
