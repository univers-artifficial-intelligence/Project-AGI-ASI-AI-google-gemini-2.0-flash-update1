import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import direct_file_access as dfa

class AILearningSystem:
    """
    Système permettant à l'IA d'apprendre de manière autonome à partir des fichiers du projet.
    Ce système utilise l'accès direct aux fichiers pour améliorer les capacités de l'IA.
    """
    
    def __init__(self):
        """Initialise le système d'apprentissage autonome."""
        self.base_path = Path(os.getcwd())
        self.learning_memory_file = self.base_path / "ai_learning_memory.json"
        self.learning_memory = self._load_learning_memory()
        self.learning_priorities = {
            "code_examples": 5,
            "documentation": 4,
            "data_structures": 4,
            "algorithms": 5,
            "configuration": 3,
            "project_structure": 3
        }
        
    def _load_learning_memory(self) -> Dict[str, Any]:
        """Charge la mémoire d'apprentissage."""
        if self.learning_memory_file.exists():
            try:
                with open(self.learning_memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return self._create_default_memory()
        return self._create_default_memory()
    
    def _create_default_memory(self) -> Dict[str, Any]:
        """Crée une structure de mémoire d'apprentissage par défaut."""
        return {
            "learned_files": {},
            "knowledge_areas": {
                "code_examples": [],
                "documentation": [],
                "data_structures": [],
                "algorithms": [],
                "configuration": [],
                "project_structure": []
            },
            "learning_sessions": [],
            "last_learning_time": 0
        }
    
    def _save_learning_memory(self) -> None:
        """Sauvegarde la mémoire d'apprentissage."""
        with open(self.learning_memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.learning_memory, f, indent=2)
    
    def classify_file_content(self, file_path: str, content: str) -> List[str]:
        """
        Classifie le contenu d'un fichier dans différentes catégories de connaissances.
        
        Args:
            file_path: Chemin du fichier
            content: Contenu du fichier
            
        Returns:
            Liste des catégories de connaissances associées au fichier
        """
        categories = []
        file_ext = Path(file_path).suffix.lower()
        
        # Classification basée sur l'extension
        if file_ext in ['.py', '.js', '.cpp', '.java', '.go']:
            categories.append("code_examples")
            
            # Analyse plus détaillée du contenu
            if "class " in content and "def " in content:
                categories.append("data_structures")
            
            if any(algo in content.lower() for algo in ["sort", "search", "algorithm", "optimize", 
                                                       "recursive", "iteration"]):
                categories.append("algorithms")
        
        # Fichiers de documentation
        if file_ext in ['.md', '.rst', '.txt'] or "readme" in file_path.lower():
            categories.append("documentation")
            
        # Fichiers de configuration
        if file_ext in ['.json', '.yml', '.yaml', '.ini', '.cfg', '.conf', '.toml']:
            categories.append("configuration")
            
        # Détection de la structure du projet
        if any(term in file_path.lower() for term in ["setup", "main", "init", "config", "structure"]):
            categories.append("project_structure")
            
        return categories
    
    def learn_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Apprend à partir du contenu d'un fichier.
        
        Args:
            file_path: Chemin du fichier à apprendre
            
        Returns:
            Résultats de l'apprentissage
        """
        content = dfa.read_file_content(file_path)
        if content.startswith("Erreur:"):
            return {"success": False, "message": content}
        
        # Classifier le contenu
        categories = self.classify_file_content(file_path, content)
        
        # Analyser et mémoriser le contenu
        insights = self._extract_insights(file_path, content, categories)
        
        # Mettre à jour la mémoire d'apprentissage
        if file_path not in self.learning_memory["learned_files"]:
            self.learning_memory["learned_files"][file_path] = {
                "categories": categories,
                "last_learned": time.time(),
                "learn_count": 1,
                "insights": insights
            }
        else:
            self.learning_memory["learned_files"][file_path]["categories"] = categories
            self.learning_memory["learned_files"][file_path]["last_learned"] = time.time()
            self.learning_memory["learned_files"][file_path]["learn_count"] += 1
            self.learning_memory["learned_files"][file_path]["insights"] = insights
        
        # Mettre à jour les domaines de connaissances
        for category in categories:
            if category in self.learning_memory["knowledge_areas"]:
                if file_path not in self.learning_memory["knowledge_areas"][category]:
                    self.learning_memory["knowledge_areas"][category].append(file_path)
        
        self._save_learning_memory()
        
        return {
            "success": True,
            "file_path": file_path,
            "categories": categories,
            "insights": insights
        }
    
    def _extract_insights(self, file_path: str, content: str, categories: List[str]) -> Dict[str, Any]:
        """
        Extrait des informations pertinentes du contenu d'un fichier.
        
        Args:
            file_path: Chemin du fichier
            content: Contenu du fichier
            categories: Catégories associées au fichier
            
        Returns:
            Informations extraites
        """
        insights = {
            "summary": "",
            "key_concepts": [],
            "code_patterns": [],
            "dependencies": []
        }
        
        # Générer un résumé en fonction du type de fichier
        file_size = len(content)
        file_ext = Path(file_path).suffix.lower()
        lines = content.split('\n')
        
        # Résumé simple basé sur les premières lignes non vides
        summary_lines = []
        for line in lines[:20]:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and len(stripped) > 5:
                summary_lines.append(stripped)
                if len(summary_lines) >= 3:
                    break
                    
        insights["summary"] = " ".join(summary_lines)
        
        # Identifier les concepts clés
        key_concepts = set()
        
        # Pour les fichiers Python
        if file_ext == '.py':
            # Identifier les classes et fonctions
            for line in lines:
                if line.strip().startswith("def "):
                    fn_name = line.strip()[4:].split('(')[0]
                    key_concepts.add(f"function:{fn_name}")
                elif line.strip().startswith("class "):
                    class_name = line.strip()[6:].split('(')[0].split(':')[0]
                    key_concepts.add(f"class:{class_name}")
            
            # Identifier les imports
            for line in lines:
                if line.strip().startswith("import ") or line.strip().startswith("from "):
                    module = line.strip().split()[1]
                    insights["dependencies"].append(module)
        
        insights["key_concepts"] = list(key_concepts)[:10]  # Limiter à 10 concepts
        
        # Identifier des patterns de code courants
        if "code_examples" in categories:
            if "if __name__ == \"__main__\":" in content:
                insights["code_patterns"].append("main_guard")
            if "try:" in content and "except:" in content:
                insights["code_patterns"].append("error_handling")
            if "class " in content and ("def __init__" in content):
                insights["code_patterns"].append("class_definition")
            if "def " in content and "return " in content:
                insights["code_patterns"].append("function_with_return")
            if "with " in content:
                insights["code_patterns"].append("context_manager")
        
        return insights
    
    def execute_learning_session(self, focus_area: Optional[str] = None, max_files: int = 5) -> Dict[str, Any]:
        """
        Exécute une session d'apprentissage autonome.
        
        Args:
            focus_area: Domaine de connaissances à privilégier
            max_files: Nombre maximum de fichiers à analyser
            
        Returns:
            Résultats de la session d'apprentissage
        """
        session_start = time.time()
        files = dfa.scan_project_files()
        
        # Filtrer les fichiers déjà appris récemment (moins de 24h)
        recent_cutoff = time.time() - (24 * 3600)
        learned_recently = {
            file_path for file_path, info in self.learning_memory["learned_files"].items()
            if info["last_learned"] > recent_cutoff
        }
        
        candidate_files = [file for file in files if file not in learned_recently]
        
        # Si un domaine de focus est spécifié, privilégier les fichiers de ce domaine
        if focus_area and focus_area in self.learning_memory["knowledge_areas"]:
            # Compléter avec des fichiers du domaine de focus s'ils n'ont pas été appris récemment
            focus_files = [
                file for file in self.learning_memory["knowledge_areas"][focus_area]
                if file not in learned_recently and file in files
            ]
            
            # Ajouter d'autres fichiers pour atteindre max_files
            if len(focus_files) < max_files:
                remaining_files = [file for file in candidate_files if file not in focus_files]
                focus_files.extend(remaining_files[:max_files - len(focus_files)])
                
            files_to_learn = focus_files[:max_files]
        else:
            # Sélectionner des fichiers à apprendre
            files_to_learn = candidate_files[:max_files]
        
        # Apprendre de chaque fichier
        learning_results = []
        for file_path in files_to_learn:
            result = self.learn_from_file(file_path)
            learning_results.append(result)
        
        # Enregistrer la session d'apprentissage
        session = {
            "timestamp": session_start,
            "duration": time.time() - session_start,
            "focus_area": focus_area,
            "files_learned": [result["file_path"] for result in learning_results if result["success"]],
            "insights_gained": sum(len(result.get("insights", {}).get("key_concepts", [])) 
                                  for result in learning_results if result["success"])
        }
        
        self.learning_memory["learning_sessions"].append(session)
        self.learning_memory["last_learning_time"] = time.time()
        self._save_learning_memory()
        
        return {
            "session": session,
            "results": learning_results
        }
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """
        Génère un résumé des connaissances acquises par l'IA.
        
        Returns:
            Résumé des connaissances
        """
        summary = {
            "files_learned": len(self.learning_memory["learned_files"]),
            "knowledge_areas": {},
            "recent_sessions": min(5, len(self.learning_memory["learning_sessions"])),
            "total_sessions": len(self.learning_memory["learning_sessions"]),
            "top_insights": []
        }
        
        # Résumer les domaines de connaissances
        for area, files in self.learning_memory["knowledge_areas"].items():
            summary["knowledge_areas"][area] = len(files)
            
        # Identifier les concepts clés les plus fréquents
        concept_frequency = {}
        for file_info in self.learning_memory["learned_files"].values():
            if "insights" in file_info and "key_concepts" in file_info["insights"]:
                for concept in file_info["insights"]["key_concepts"]:
                    if concept in concept_frequency:
                        concept_frequency[concept] += 1
                    else:
                        concept_frequency[concept] = 1
        
        # Trier les concepts par fréquence
        sorted_concepts = sorted(concept_frequency.items(), key=lambda x: x[1], reverse=True)
        summary["top_insights"] = sorted_concepts[:10]  # Top 10 des concepts
        
        return summary
    
    def suggest_learning_focus(self) -> str:
        """
        Suggère un domaine de connaissances à privilégier pour la prochaine session.
        
        Returns:
            Domaine de connaissances recommandé
        """
        # Analyser la distribution des connaissances
        area_counts = {area: len(files) for area, files in self.learning_memory["knowledge_areas"].items()}
        
        # Calculer un score pour chaque domaine basé sur la priorité et le nombre de fichiers appris
        scores = {}
        for area, priority in self.learning_priorities.items():
            count = area_counts.get(area, 0)
            # Formule: plus la priorité est élevée et moins il y a de fichiers, plus le score est élevé
            scores[area] = priority * (1 + 1/(count + 1))
        
        # Identifier le domaine avec le score le plus élevé
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        else:
            return "code_examples"  # Valeur par défaut

# Interface principale pour l'IA
ai_learning = AILearningSystem()

def start_learning_session(focus_area=None, max_files=5):
    """Démarre une session d'apprentissage autonome."""
    return ai_learning.execute_learning_session(focus_area, max_files)

def learn_specific_file(file_path):
    """Apprend à partir d'un fichier spécifique."""
    return ai_learning.learn_from_file(file_path)

def get_learning_summary():
    """Obtient un résumé des connaissances acquises."""
    return ai_learning.get_knowledge_summary()

def get_suggested_focus():
    """Suggère un domaine de connaissances à privilégier."""
    return ai_learning.suggest_learning_focus()

if __name__ == "__main__":
    # Test du système d'apprentissage
    print("Démarrage d'une session d'apprentissage autonome...")
    session = start_learning_session(max_files=3)
    
    print(f"Session terminée. {len(session['results'])} fichiers analysés.")
    
    print("\nRésumé des connaissances:")
    summary = get_learning_summary()
    print(f"Fichiers appris: {summary['files_learned']}")
    print("Domaines de connaissances:")
    for area, count in summary['knowledge_areas'].items():
        print(f"- {area}: {count} fichiers")
        
    print(f"\nDomaine de focus recommandé: {get_suggested_focus()}")
