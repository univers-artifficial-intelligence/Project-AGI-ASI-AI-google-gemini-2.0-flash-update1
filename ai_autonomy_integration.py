import os
import time
import json
import logging
import datetime
import pytz
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import direct_file_access as dfa
import ai_learning_system as als

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='ai_autonomy.log'
)
logger = logging.getLogger('ai_autonomy')

class AIAutonomyIntegration:
    """
    Système d'intégration pour l'autonomie de l'IA permettant l'accès aux fichiers et
    l'apprentissage continu en fonction des interactions avec l'utilisateur.
    """
    
    def __init__(self):
        """Initialise le système d'autonomie de l'IA."""
        self.base_path = Path(os.getcwd())
        self.interaction_memory_file = self.base_path / "ai_interaction_memory.json"
        self.interaction_memory = self._load_interaction_memory()
        self.last_learning_check = 0
        self.learning_interval = 3600  # 1 heure entre les sessions d'apprentissage
        self.auto_learning = True  # Active par défaut
        
    def _load_interaction_memory(self) -> Dict[str, Any]:
        """Charge la mémoire d'interactions."""
        if self.interaction_memory_file.exists():
            try:
                with open(self.interaction_memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return self._create_default_memory()
        return self._create_default_memory()
    
    def _create_default_memory(self) -> Dict[str, Any]:
        """Crée une structure de mémoire d'interactions par défaut."""
        return {
            "interactions": [],
            "topics": {},
            "file_references": {},
            "context_history": [],
            "auto_learning_sessions": []
        }
    
    def _save_interaction_memory(self) -> None:
        """Sauvegarde la mémoire d'interactions."""
        with open(self.interaction_memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.interaction_memory, f, indent=2)
    
    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Traite l'entrée de l'utilisateur pour déterminer les actions autonomes à effectuer.
        
        Args:
            user_input: Entrée de l'utilisateur
            
        Returns:
            Informations sur le traitement et les actions entreprises
        """
        logger.info(f"Traitement de l'entrée utilisateur: {user_input[:50]}...")
        
        # Obtenir le contexte temporel actuel
        temporal_context = self.get_current_temporal_context()
        
        # Mettre à jour la mémoire d'interactions avec le contexte temporel
        self.interaction_memory["interactions"].append({
            "timestamp": time.time(),
            "user_input": user_input,
            "temporal_context": temporal_context
        })
        
        # Extraire des sujets potentiels
        topics = self._extract_topics(user_input)
        
        # Enregistrer les sujets dans la mémoire
        for topic in topics:
            if topic in self.interaction_memory["topics"]:
                self.interaction_memory["topics"][topic] += 1
            else:
                self.interaction_memory["topics"][topic] = 1
        
        # Rechercher des fichiers pertinents
        relevant_files = dfa.get_relevant_files(user_input)
        
        # Enregistrer les références de fichiers
        for file_path in relevant_files:
            if file_path in self.interaction_memory["file_references"]:
                self.interaction_memory["file_references"][file_path] += 1
            else:
                self.interaction_memory["file_references"][file_path] = 1
        
        # Vérifier s'il faut lancer une session d'apprentissage autonome
        should_learn, reason = self._should_trigger_learning()
        learning_results = None
        
        if should_learn:
            focus = als.get_suggested_focus()
            logger.info(f"Déclenchement d'une session d'apprentissage autonome sur {focus}")
            learning_results = als.start_learning_session(focus_area=focus, max_files=3)
            
            # Enregistrer la session d'apprentissage
            self.interaction_memory["auto_learning_sessions"].append({
                "timestamp": time.time(),
                "trigger_reason": reason,
                "focus_area": focus,
                "files_learned": learning_results["session"]["files_learned"]
            })
            
            # Mettre à jour le moment de la dernière vérification
            self.last_learning_check = time.time()
        
        # Mettre à jour l'historique du contexte
        if len(self.interaction_memory["context_history"]) >= 10:
            self.interaction_memory["context_history"].pop(0)  # Supprimer le plus ancien
        self.interaction_memory["context_history"].append({
            "timestamp": time.time(),
            "topics": topics,
            "relevant_files": relevant_files
        })
        
        self._save_interaction_memory()
        
        # Retourner les résultats du traitement
        response = {
            "topics_identified": topics,
            "relevant_files": relevant_files,
            "autonomous_learning": {
                "triggered": should_learn,
                "reason": reason,
                "results": learning_results
            }
        }
        
        return response
    
    def _extract_topics(self, text: str) -> List[str]:
        """
        Extrait les sujets pertinents du texte.
        
        Args:
            text: Texte à analyser
            
        Returns:
            Liste des sujets extraits
        """
        # Liste de mots-clés techniques à rechercher
        technical_keywords = [
            "accès", "fichier", "direct", "lecture", "écriture", "recherche",
            "analyse", "code", "algorithme", "structure", "données", "apprentissage",
            "autonome", "intelligence", "artificielle", "IA", "système", "projet",
            "programmation", "python", "api", "gemini", "configuration"
        ]
        
        topics = []
        text_lower = text.lower()
        
        # Extraire les mots-clés techniques présents
        for keyword in technical_keywords:
            if keyword in text_lower:
                topics.append(keyword)
                
        return topics
    
    def _should_trigger_learning(self) -> tuple[bool, str]:
        """
        Détermine si une session d'apprentissage autonome doit être déclenchée.
        
        Returns:
            (bool, str): Tuple contenant un booléen indiquant si l'apprentissage doit être déclenché
                        et la raison du déclenchement
        """
        # Si l'apprentissage automatique est désactivé
        if not self.auto_learning:
            return False, "Auto-apprentissage désactivé"
        
        current_time = time.time()
        
        # Vérifier si assez de temps s'est écoulé depuis la dernière session d'apprentissage
        if current_time - self.last_learning_check < self.learning_interval:
            return False, "Intervalle d'apprentissage non écoulé"
            
        # Vérifier si de nouveaux sujets/fichiers ont été identifiés depuis la dernière session
        if self.interaction_memory["context_history"]:
            last_timestamp = self.interaction_memory["context_history"][-1]["timestamp"]
            
            # Si aucune session d'apprentissage n'a encore eu lieu ou si un nouveau contexte est apparu
            if (not self.interaction_memory["auto_learning_sessions"] or 
                last_timestamp > self.interaction_memory["auto_learning_sessions"][-1]["timestamp"]):
                return True, "Nouveau contexte détecté"
        
        # Vérifier s'il est temps de faire une session périodique
        auto_sessions = self.interaction_memory["auto_learning_sessions"]
        if not auto_sessions or (current_time - auto_sessions[-1]["timestamp"] > self.learning_interval * 6):
            return True, "Session périodique"
            
        return False, "Aucun déclencheur actif"
    
    def toggle_auto_learning(self, enabled: bool) -> Dict[str, Any]:
        """
        Active ou désactive l'apprentissage automatique.
        
        Args:
            enabled: True pour activer, False pour désactiver
            
        Returns:
            État de la configuration
        """
        self.auto_learning = enabled
        status = "activé" if enabled else "désactivé"
        logger.info(f"Apprentissage automatique {status}")
        
        return {
            "auto_learning": enabled,
            "message": f"L'apprentissage automatique est maintenant {status}"
        }
    
    def get_file_content(self, file_path: str) -> Dict[str, Any]:
        """
        Obtient le contenu d'un fichier avec des métadonnées supplémentaires.
        
        Args:
            file_path: Chemin du fichier
            
        Returns:
            Contenu et métadonnées du fichier
        """
        content = dfa.read_file_content(file_path)
        
        # Si le fichier a été lu avec succès et qu'il n'y a pas d'erreur
        if not content.startswith("Erreur:"):
            # Apprendre de ce fichier
            learning_result = als.learn_specific_file(file_path)
            
            # Incrémenter les références
            if file_path in self.interaction_memory["file_references"]:
                self.interaction_memory["file_references"][file_path] += 1
            else:
                self.interaction_memory["file_references"][file_path] = 1
                
            self._save_interaction_memory()
            
            return {
                "success": True,
                "file_path": file_path,
                "content": content,
                "references": self.interaction_memory["file_references"].get(file_path, 1),
                "learning_result": learning_result
            }
        else:
            return {
                "success": False,
                "file_path": file_path,
                "error": content
            }
    
    def search_project_files(self, query: str) -> Dict[str, Any]:
        """
        Recherche des fichiers dans le projet et apprend des fichiers pertinents.
        
        Args:
            query: Termes de recherche
            
        Returns:
            Résultats de la recherche
        """
        search_results = dfa.search_files(query)
        
        # Apprendre du premier fichier pertinent
        if search_results:
            top_file = search_results[0]["file_path"]
            learning_result = als.learn_specific_file(top_file)
            
            search_results[0]["learning_result"] = learning_result
        
        return {
            "query": query,
            "results": search_results
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Obtient un rapport de statut complet du système d'autonomie.
        
        Returns:
            Statut du système
        """
        # Obtenir des statistiques sur les fichiers du projet
        all_files = dfa.scan_project_files()
        
        # Obtenir le résumé d'apprentissage
        learning_summary = als.get_learning_summary()
        
        # Calculer des statistiques sur les interactions
        interaction_count = len(self.interaction_memory["interactions"])
        topic_count = len(self.interaction_memory["topics"])
        referenced_files = len(self.interaction_memory["file_references"])
        auto_learning_sessions = len(self.interaction_memory["auto_learning_sessions"])
        
        # Obtenir le contexte temporel actuel
        temporal_context = self.get_current_temporal_context()
        
        # Générer le rapport de statut
        return {
            "temporal_awareness": {
                "current_time": temporal_context.get("current_time", "Inconnu"),
                "current_date": temporal_context.get("current_date", "Inconnu"),
                "timezone": temporal_context.get("timezone", "Inconnu"),
                "full_context": temporal_context.get("formatted_full", "Inconnu")
            },
            "project": {
                "total_files": len(all_files),
                "file_types": self._count_file_types(all_files)
            },
            "learning": {
                "files_learned": learning_summary["files_learned"],
                "knowledge_areas": learning_summary["knowledge_areas"],
                "suggested_focus": als.get_suggested_focus(),
                "auto_sessions": auto_learning_sessions
            },
            "interactions": {
                "count": interaction_count,
                "topics": topic_count,
                "top_topics": self._get_top_items(self.interaction_memory["topics"], 5),
                "referenced_files": referenced_files,
                "top_files": self._get_top_items(self.interaction_memory["file_references"], 5)
            },
            "system": {
                "auto_learning": self.auto_learning,
                "learning_interval": self.learning_interval,
                "last_learning_check": self.last_learning_check
            }
        }
        
    def _count_file_types(self, files: List[str]) -> Dict[str, int]:
        """Compte les types de fichiers dans la liste."""
        extension_counts = {}
        for file_path in files:
            ext = Path(file_path).suffix
            if ext in extension_counts:
                extension_counts[ext] += 1
            else:
                extension_counts[ext] = 1
        return extension_counts
        
    def _get_top_items(self, counter: Dict[str, int], limit: int) -> List[Dict[str, Union[str, int]]]:
        """Retourne les éléments les plus fréquents d'un compteur."""
        sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        return [{"item": item, "count": count} for item, count in sorted_items[:limit]]
    
    def get_current_temporal_context(self) -> Dict[str, Any]:
        """
        Obtient le contexte temporel actuel pour la conscience autonome de l'IA.
        
        Returns:
            Dictionnaire contenant les informations temporelles actuelles
        """
        try:
            # Obtenir l'heure actuelle dans le fuseau horaire français
            current_datetime = datetime.datetime.now(pytz.timezone('Europe/Paris'))
            
            # Formater différentes représentations du temps
            temporal_context = {
                "current_datetime": current_datetime.isoformat(),
                "current_time": current_datetime.strftime("%H:%M:%S"),
                "current_date": current_datetime.strftime("%A %d %B %Y"),
                "timestamp": time.time(),
                "day_of_week": current_datetime.strftime("%A"),
                "day_number": current_datetime.day,
                "month": current_datetime.strftime("%B"),
                "year": current_datetime.year,
                "timezone": "Europe/Paris",
                "formatted_full": current_datetime.strftime("%A %d %B %Y à %H:%M:%S")
            }
            
            return temporal_context
            
        except Exception as e:
            logger.error(f"Erreur lors de l'obtention du contexte temporel: {str(e)}")
            return {
                "error": "Impossible d'obtenir l'heure actuelle",
                "timestamp": time.time()
            }
    
# Interface principale pour l'utilisation par l'IA
ai_autonomy = AIAutonomyIntegration()

def process_input(user_input):
    """Traite l'entrée utilisateur et déclenche les actions autonomes appropriées."""
    return ai_autonomy.process_user_input(user_input)

def get_file(file_path):
    """Obtient le contenu d'un fichier avec apprentissage autonome."""
    return ai_autonomy.get_file_content(file_path)

def search_files(query):
    """Recherche des fichiers dans le projet avec apprentissage autonome."""
    return ai_autonomy.search_project_files(query)

def get_status():
    """Obtient un rapport de statut du système d'autonomie."""
    return ai_autonomy.get_system_status()

def set_auto_learning(enabled):
    """Active ou désactive l'apprentissage automatique."""
    return ai_autonomy.toggle_auto_learning(enabled)

def get_temporal_context():
    """Obtient le contexte temporel actuel pour l'IA."""
    return ai_autonomy.get_current_temporal_context()

def get_autonomous_time_awareness():
    """Obtient la conscience temporelle autonome complète."""
    try:
        from autonomous_time_awareness import get_full_temporal_awareness
        return get_full_temporal_awareness()
    except ImportError:
        return ai_autonomy.get_current_temporal_context()

if __name__ == "__main__":
    # Test du système d'autonomie
    print("Test du système d'autonomie de l'IA")
    
    print("\n1. Traitement d'une entrée utilisateur")
    result = process_input("Je voudrais comprendre comment fonctionne l'accès aux fichiers dans ce projet")
    print(f"Sujets identifiés: {result['topics_identified']}")
    print(f"Fichiers pertinents: {[f for f in result['relevant_files'][:3]]}")
    
    print("\n2. Recherche de fichiers")
    search = search_files("apprentissage autonome")
    if search["results"]:
        print(f"Premier résultat: {search['results'][0]['file_path']}")
    else:
        print("Aucun résultat trouvé")
    
    print("\n3. Statut du système")
    status = get_status()
    print(f"Fichiers dans le projet: {status['project']['total_files']}")
    print(f"Fichiers appris: {status['learning']['files_learned']}")
    print(f"Sessions d'apprentissage automatiques: {status['learning']['auto_sessions']}")
