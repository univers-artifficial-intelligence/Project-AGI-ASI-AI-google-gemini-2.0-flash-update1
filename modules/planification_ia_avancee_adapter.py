"""
Adaptateur pour le module planification_ia_avancee.
Ce module permet de connecter le système de planification IA avancée
au système de gestion de modules.
"""

# Import du module principal de planification
from planification_ia_avancee import BaseDonneesPlanification, Plan, GestionnairePlanification

# Métadonnées du module
MODULE_METADATA = {
    "enabled": True,
    "priority": 80,
    "description": "Adapte le système de planification IA avancée au gestionnaire de modules",
    "version": "0.1.0",
    "dependencies": [],
    "hooks": ["process_request", "process_response", "generate_plan"]
}

# Instance du gestionnaire
_gestionnaire = None

def get_gestionnaire():
    """Récupère l'instance singleton du gestionnaire de planification"""
    global _gestionnaire
    if _gestionnaire is None:
        _gestionnaire = GestionnairePlanification()
    return _gestionnaire

def process(data: dict, hook: str) -> dict:
    """
    Fonction principale de traitement.
    
    Args:
        data: Les données à traiter
        hook: Le hook appelé
        
    Returns:
        Les données modifiées
    """
    # Initialiser le résultat avec les données d'entrée
    result = data.copy()
    
    # Traitement spécifique en fonction du hook
    if hook == "generate_plan":
        # Hook spécial pour la génération de plan
        gestionnaire = get_gestionnaire()
        
        if "plan_request" in data:
            plan_request = data["plan_request"]
            
            # Créer un plan en fonction de la demande
            plan_id = gestionnaire.creer_plan(
                titre=plan_request.get("titre", "Nouveau plan"),
                description=plan_request.get("description", ""),
                niveau=plan_request.get("niveau", "tactique"),
                priorite=plan_request.get("priorite", 5),
                etapes=plan_request.get("etapes", [])
            )
            
            # Récupérer le plan créé
            plan = gestionnaire.obtenir_plan(plan_id)
            
            if plan:
                result["plan"] = plan.to_dict()
                result["success"] = True
                result["message"] = f"Plan créé avec succès, ID: {plan_id}"
            else:
                result["success"] = False
                result["message"] = "Échec de la création du plan"
    
    elif hook == "process_request":
        # Traitement pour les requêtes
        if "text" in result:
            # Analyser la demande pour détecter si elle concerne la planification
            text = result["text"].lower()
            planning_keywords = ["plan", "planifier", "planification", "étapes", "organiser"]
            
            if any(keyword in text for keyword in planning_keywords):
                result["requires_planning"] = True
                result["planning_keywords_detected"] = [k for k in planning_keywords if k in text]
    
    elif hook == "process_response":
        # Traitement pour les réponses
        pass
    
    return result

def handle_generate_plan(data):
    """Handler spécifique pour la génération d'un plan"""
    return process(data, "generate_plan")

# Fonctions auxiliaires (pour démonstration)
def analyser_possibilite_planification(texte):
    """
    Analyse si un texte contient une demande de planification.
    
    Args:
        texte: Le texte à analyser
        
    Returns:
        Une estimation de la probabilité que le texte concerne une planification
    """
    keywords = {
        "plan": 0.7,
        "planifier": 0.8,
        "organiser": 0.6,
        "étapes": 0.5,
        "tâches": 0.5,
        "objectif": 0.4,
        "calendrier": 0.5,
        "programmation": 0.6,
        "échéance": 0.5
    }
    
    score = 0
    count = 0
    
    for keyword, weight in keywords.items():
        if keyword in texte.lower():
            score += weight
            count += 1
    
    if count == 0:
        return 0
    
    return min(1.0, score / (count * 0.7))  # Normaliser
