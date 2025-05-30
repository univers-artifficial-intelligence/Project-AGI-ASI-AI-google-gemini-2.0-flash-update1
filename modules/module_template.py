"""
Module template pour démontrer comment créer un module compatible avec le système.
Ce module fournit un exemple qu'on peut copier pour créer de nouveaux modules.
"""

# Métadonnées du module
MODULE_METADATA = {
    "enabled": True,
    "priority": 100,
    "description": "Module template d'exemple",
    "version": "0.1.0",
    "dependencies": [],
    "hooks": ["process_request", "process_response"]
}

def process(data: dict, hook: str) -> dict:
    """
    Fonction principale appelée par le gestionnaire de modules.
    
    Args:
        data: Les données à traiter (structure dépendant du hook)
        hook: Le hook qui est appelé (ex: 'process_request', 'process_response')
        
    Returns:
        Les données éventuellement modifiées
    """
    # Ici, nous ne faisons rien, c'est juste un exemple
    print(f"Module template appelé avec le hook: {hook}")
    return data

# Vous pouvez également définir des handlers spécifiques pour chaque hook
def handle_process_request(data):
    """Handler spécifique pour le hook process_request"""
    return data

def handle_process_response(data):
    """Handler spécifique pour le hook process_response"""
    return data

# Autres fonctions/classes utiles...
def utility_function():
    """Une fonction utilitaire quelconque"""
    return "Utility function called"

# Exemple de classe compatible avec le système
class ModuleTemplateProcessor:
    """Classe qui implémente le traitement pour ce module"""
    
    def __init__(self):
        """Initialisation"""
        self.config = {"some_setting": True}
    
    def process(self, data, hook):
        """
        Méthode de traitement principal (alternative à la fonction process)
        Cette méthode sera utilisée si la fonction process n'est pas définie au niveau du module
        """
        print(f"Class-based process called for hook: {hook}")
        return data
    
    def handle_process_request(self, data):
        """Handler spécifique pour le hook process_request"""
        return data
