import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

class DirectFileAccess:
    """
    Système permettant à l'IA d'accéder directement aux fichiers texte du projet
    pour favoriser son apprentissage autonome.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialise le système d'accès direct aux fichiers.
        
        Args:
            base_path: Chemin de base du projet. Si non spécifié, utilise le répertoire courant.
        """
        self.base_path = Path(base_path) if base_path else Path(os.getcwd())
        self.memory_file = self.base_path / "ai_file_memory.json"
        self.file_memory = self._load_memory()
        self.text_extensions = {'.txt', '.py', '.js', '.html', '.css', '.md', '.json', '.csv', '.xml', '.yml', '.yaml', '.ini', '.cfg'}
        self.excluded_dirs = {'.git', '__pycache__', 'venv', 'env', 'node_modules', '.vscode', '.idea'}
        
        # Définir le chemin vers le dossier de données textuelles
        self.text_data_folder = self.base_path / 'gemini_text_data'
        self.ensure_text_data_folder_exists()
        
    def ensure_text_data_folder_exists(self):
        """Créer le dossier de données textuelles s'il n'existe pas."""
        if not self.text_data_folder.exists():
            os.makedirs(self.text_data_folder)
            print(f"Dossier de données textuelles créé: {self.text_data_folder}")
        return self.text_data_folder

    def _load_memory(self) -> Dict[str, Any]:
        """Charge la mémoire des fichiers consultés."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {"files": {}, "last_scan": 0}
        return {"files": {}, "last_scan": 0}
    
    def _save_memory(self) -> None:
        """Sauvegarde la mémoire des fichiers consultés."""
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.file_memory, f, indent=2)
    
    def scan_files(self, force: bool = False) -> List[str]:
        """
        Parcourt le système de fichiers pour identifier tous les fichiers texte.
        
        Args:
            force: Force un nouveau scan même si un scan récent a été effectué

        Returns:
            Liste des chemins des fichiers texte trouvés
        """
        current_time = time.time()
        
        # Éviter un scan trop fréquent (max une fois toutes les 5 minutes)
        if not force and current_time - self.file_memory.get("last_scan", 0) < 300:
            return [file for file in self.file_memory.get("files", {})]
            
        file_list = []
        
        for root, dirs, files in os.walk(self.base_path):
            # Filtrer les répertoires exclus
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(self.base_path)
                
                if file_path.suffix.lower() in self.text_extensions:
                    str_path = str(rel_path)
                    file_list.append(str_path)
                    
                    # Mise à jour des métadonnées du fichier
                    if str_path not in self.file_memory["files"]:
                        self.file_memory["files"][str_path] = {
                            "last_accessed": None,
                            "access_count": 0,
                            "relevance_score": 0
                        }
        
        # Supprimer les fichiers qui n'existent plus
        for file_path in list(self.file_memory["files"].keys()):
            if file_path not in file_list:
                del self.file_memory["files"][file_path]
                
        self.file_memory["last_scan"] = current_time
        self._save_memory()
        
        return file_list
    
    def read_file(self, file_path: str) -> str:
        """
        Lit le contenu d'un fichier texte.
        
        Args:
            file_path: Chemin relatif du fichier à lire

        Returns:
            Contenu du fichier
        """
        full_path = self.base_path / file_path
        
        if not full_path.exists():
            return f"Erreur: Le fichier {file_path} n'existe pas."
            
        if not full_path.is_file():
            return f"Erreur: {file_path} n'est pas un fichier."
            
        if full_path.suffix.lower() not in self.text_extensions:
            return f"Erreur: {file_path} n'est pas un fichier texte."
            
        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                
            # Mettre à jour les métadonnées du fichier
            rel_path = str(full_path.relative_to(self.base_path))
            if rel_path in self.file_memory["files"]:
                self.file_memory["files"][rel_path]["last_accessed"] = time.time()
                self.file_memory["files"][rel_path]["access_count"] += 1
                self._save_memory()
                
            return content
        except Exception as e:
            return f"Erreur lors de la lecture du fichier {file_path}: {str(e)}"
    
    def search_in_files(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Recherche des fichiers contenant le texte spécifié.
        
        Args:
            query: Texte à rechercher dans les fichiers
            max_results: Nombre maximum de résultats à retourner

        Returns:
            Liste des fichiers correspondants avec des extraits pertinents
        """
        results = []
        files = self.scan_files()
        
        for file_path in files:
            full_path = self.base_path / file_path
            try:
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                
                if query.lower() in content.lower():
                    # Trouver un extrait pertinent
                    index = content.lower().find(query.lower())
                    start = max(0, index - 100)
                    end = min(len(content), index + len(query) + 100)
                    
                    # Ajuster pour ne pas couper les mots
                    while start > 0 and content[start] != ' ' and content[start] != '\n':
                        start -= 1
                    
                    while end < len(content) and content[end] != ' ' and content[end] != '\n':
                        end += 1
                        
                    excerpt = content[start:end]
                    if start > 0:
                        excerpt = "..." + excerpt
                    if end < len(content):
                        excerpt = excerpt + "..."
                    
                    results.append({
                        "file_path": file_path,
                        "excerpt": excerpt,
                        "relevance": self._calculate_relevance(file_path, query, content)
                    })
                    
                    if len(results) >= max_results:
                        break
            except Exception:
                continue
                
        # Trier par pertinence
        results.sort(key=lambda x: x["relevance"], reverse=True)
        
        return results
    
    def _calculate_relevance(self, file_path: str, query: str, content: str) -> float:
        """
        Calcule un score de pertinence pour un fichier par rapport à une requête.
        
        Args:
            file_path: Chemin du fichier
            query: Termes de recherche
            content: Contenu du fichier

        Returns:
            Score de pertinence
        """
        score = 0.0
        
        # Nombre d'occurrences du terme de recherche
        occurrences = content.lower().count(query.lower())
        score += min(occurrences * 0.5, 5.0)  # Plafonné à 5.0
        
        # Récence d'accès
        file_info = self.file_memory["files"].get(file_path, {})
        last_accessed = file_info.get("last_accessed")
        if last_accessed:
            days_since_access = (time.time() - last_accessed) / (24 * 3600)
            recency_score = max(0, 3.0 - (days_since_access / 7))  # Score décroît avec le temps
            score += recency_score
        
        # Fréquence d'accès
        access_count = file_info.get("access_count", 0)
        score += min(access_count * 0.2, 2.0)  # Plafonné à 2.0
            
        # Mise à jour du score de pertinence dans la mémoire
        if file_path in self.file_memory["files"]:
            # Moyenne glissante
            current_score = self.file_memory["files"][file_path].get("relevance_score", 0)
            self.file_memory["files"][file_path]["relevance_score"] = current_score * 0.7 + score * 0.3
            self._save_memory()
            
        return score
    
    def suggest_relevant_files(self, context: str, max_suggestions: int = 5) -> List[str]:
        """
        Suggère des fichiers pertinents en fonction du contexte actuel.
        
        Args:
            context: Contexte actuel (question, discussion, etc.)
            max_suggestions: Nombre maximum de suggestions à retourner

        Returns:
            Liste des fichiers suggérés
        """
        # Extraire des mots-clés du contexte
        keywords = self._extract_keywords(context)
        
        # Calculer les scores pour chaque fichier
        scores = {}
        for file_path in self.file_memory["files"]:
            score = 0.0
            
            # Score de base à partir de la mémoire
            file_info = self.file_memory["files"][file_path]
            score += file_info.get("relevance_score", 0) * 0.5
            
            # Ajouter des points pour correspondance de mots-clés dans le nom de fichier
            for keyword in keywords:
                if keyword.lower() in file_path.lower():
                    score += 2.0
                    
            scores[file_path] = score
            
        # Retourner les fichiers les plus pertinents
        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:max_suggestions]
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extraire des mots-clés pertinents du texte."""
        # Supprimer la ponctuation et convertir en minuscules
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
        words = cleaned.split()
        
        # Filtrer les mots courts et les mots communs
        common_words = {"le", "la", "les", "un", "une", "des", "et", "ou", "à", "de", "du", "dans",
                       "en", "sur", "pour", "par", "avec", "sans", "ce", "cette", "ces", "mon", "ma",
                       "mes", "ton", "ta", "tes", "son", "sa", "ses", "qui", "que", "quoi", "comment"}
        
        keywords = {word for word in words if len(word) > 3 and word not in common_words}
        return keywords

    def get_file_structure(self) -> Dict[str, Any]:
        """
        Retourne une structure hiérarchique des fichiers du projet.
        
        Returns:
            Dictionnaire représentant l'arborescence du projet
        """
        structure = {}
        files = self.scan_files()
        
        for file_path in files:
            parts = Path(file_path).parts
            current = structure
            
            # Parcourir l'arborescence et la construire
            for i, part in enumerate(parts):
                if i == len(parts) - 1:  # C'est un fichier
                    if "files" not in current:
                        current["files"] = []
                    current["files"].append(part)
                else:  # C'est un répertoire
                    if "dirs" not in current:
                        current["dirs"] = {}
                    if part not in current["dirs"]:
                        current["dirs"][part] = {}
                    current = current["dirs"][part]
                    
        return structure

# Interface simple pour l'utilisation par l'IA
file_access = DirectFileAccess()

def scan_project_files(force=False):
    """Analyse le projet et retourne la liste des fichiers texte disponibles."""
    return file_access.scan_files(force)

def read_file_content(file_path):
    """Lit le contenu d'un fichier texte."""
    return file_access.read_file(file_path)

def search_files(query, max_results=10):
    """Recherche des fichiers contenant le texte spécifié."""
    return file_access.search_in_files(query, max_results)

def get_relevant_files(context, max_suggestions=5):
    """Suggère des fichiers pertinents en fonction du contexte."""
    return file_access.suggest_relevant_files(context, max_suggestions)

def get_project_structure():
    """Retourne la structure hiérarchique du projet."""
    return file_access.get_file_structure()

if __name__ == "__main__":
    # Test du système
    print("Analyse des fichiers du projet...")
    files = scan_project_files(force=True)
    print(f"Nombre de fichiers trouvés: {len(files)}")
    
    if files:
        print("\nPremiers fichiers trouvés:")
        for i, file in enumerate(files[:5]):
            print(f"- {file}")
            
        print("\nRecherche de fichiers contenant 'IA':")
        results = search_files("IA", 3)
        for result in results:
            print(f"- {result['file_path']}")
            print(f"  Extrait: {result['excerpt'][:100]}...")
