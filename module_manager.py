import os
import sys
import importlib
import importlib.util
import inspect
import time
import logging
import json
from typing import Dict, List, Any, Callable, Optional, Union
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent, FileDeletedEvent

# Configuration des chemins
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.join(PROJECT_ROOT, 'modules')
MODULE_REGISTRY_PATH = os.path.join(PROJECT_ROOT, 'module_registry.json')

# Création du dossier modules s'il n'existe pas
if not os.path.exists(MODULES_DIR):
    os.makedirs(MODULES_DIR)
    # Créer un fichier __init__.py pour que Python reconnaisse le dossier comme un package
    with open(os.path.join(MODULES_DIR, '__init__.py'), 'w') as f:
        f.write('# Module package initialization\n')

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, 'module_manager.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ModuleManager')

class ModuleInfo:
    """Classe pour stocker les informations sur un module d'amélioration."""
    
    def __init__(
        self, 
        name: str, 
        path: str,
        module_obj: Any = None,
        enabled: bool = True,
        priority: int = 100,
        description: str = "",
        version: str = "0.1",
        dependencies: List[str] = None,
        hooks: List[str] = None,
        processor: Callable = None,
        creation_time: float = None,
        last_modified: float = None,
        module_type: str = "standard"
    ):
        self.name = name
        self.path = path
        self.module_obj = module_obj
        self.enabled = enabled
        self.priority = priority  # Plus bas = plus prioritaire
        self.description = description
        self.version = version
        self.dependencies = dependencies or []
        self.hooks = hooks or []
        self.processor = processor
        self.creation_time = creation_time or time.time()
        self.last_modified = last_modified or time.time()
        self.error = None
        self.module_type = module_type  # 'standard', 'class_based', 'function_based', 'auto_generated'
        self.available_functions = {}  # Dictionnaire des fonctions disponibles dans ce module
    
    def to_dict(self) -> Dict:
        """Convertit l'objet en dictionnaire pour la sérialisation."""
        return {
            "name": self.name,
            "path": self.path,
            "enabled": self.enabled,
            "priority": self.priority,
            "description": self.description,
            "version": self.version,
            "dependencies": self.dependencies,
            "hooks": self.hooks,
            "creation_time": self.creation_time,
            "last_modified": self.last_modified,
            "module_type": self.module_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModuleInfo':
        """Crée une instance à partir d'un dictionnaire."""
        return cls(
            name=data.get("name", ""),
            path=data.get("path", ""),
            enabled=data.get("enabled", True),
            priority=data.get("priority", 100),
            description=data.get("description", ""),
            version=data.get("version", "0.1"),
            dependencies=data.get("dependencies", []),
            hooks=data.get("hooks", []),
            creation_time=data.get("creation_time"),
            last_modified=data.get("last_modified"),
            module_type=data.get("module_type", "standard")
        )

class ModuleRegistry:
    """Gère le registre des modules d'amélioration."""
    
    def __init__(self, registry_path: str = MODULE_REGISTRY_PATH):
        self.registry_path = registry_path
        self.modules: Dict[str, ModuleInfo] = {}
        self.load_registry()
    
    def load_registry(self) -> None:
        """Charge le registre des modules depuis un fichier JSON."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                    for module_name, module_data in data.items():
                        self.modules[module_name] = ModuleInfo.from_dict(module_data)
                logger.info(f"Module registry loaded with {len(self.modules)} modules")
            except Exception as e:
                logger.error(f"Error loading module registry: {e}")
        else:
            logger.info("No module registry found, starting with empty registry")
    
    def save_registry(self) -> None:
        """Sauvegarde le registre des modules dans un fichier JSON."""
        try:
            serialized = {name: module.to_dict() for name, module in self.modules.items()}
            with open(self.registry_path, 'w') as f:
                json.dump(serialized, f, indent=4)
            logger.info(f"Module registry saved with {len(self.modules)} modules")
        except Exception as e:
            logger.error(f"Error saving module registry: {e}")
    
    def register_module(self, module_info: ModuleInfo) -> bool:
        """Ajoute un module au registre."""
        try:
            self.modules[module_info.name] = module_info
            self.save_registry()
            logger.info(f"Module '{module_info.name}' registered successfully")
            return True
        except Exception as e:
            logger.error(f"Error registering module '{module_info.name}': {e}")
            return False
    
    def unregister_module(self, module_name: str) -> bool:
        """Supprime un module du registre."""
        if module_name in self.modules:
            del self.modules[module_name]
            self.save_registry()
            logger.info(f"Module '{module_name}' unregistered")
            return True
        else:
            logger.warning(f"Attempted to unregister non-existent module '{module_name}'")
            return False
    
    def get_module(self, module_name: str) -> Optional[ModuleInfo]:
        """Récupère un module par son nom."""
        return self.modules.get(module_name)
    
    def get_modules_by_hook(self, hook_name: str) -> List[ModuleInfo]:
        """Récupère tous les modules qui implémentent un hook spécifique."""
        modules = [
            m for m in self.modules.values() 
            if hook_name in (m.hooks or []) and m.enabled and m.processor is not None
        ]
        # Trier par priorité (plus petit nombre = plus grande priorité)
        return sorted(modules, key=lambda m: m.priority)
    
    def get_all_enabled_modules(self) -> List[ModuleInfo]:
        """Récupère tous les modules activés."""
        return sorted(
            [m for m in self.modules.values() if m.enabled],
            key=lambda m: m.priority
        )

class ModuleAdapter:
    """
    Adapte différents types de modules pour qu'ils s'intègrent dans le système.
    Génère des processeurs pour les modules qui n'en ont pas.
    """
    
    @staticmethod
    def create_generic_processor(module_obj):
        """
        Crée un processeur générique pour un module qui n'a pas de fonction process().
        
        Args:
            module_obj: L'objet module Python
            
        Returns:
            Une fonction process() générée automatiquement
        """
        # Trouver les fonctions utiles dans le module
        functions = {}
        classes = {}
        
        for name, obj in inspect.getmembers(module_obj):
            # Ignorer les attributs privés/spéciaux et les imports
            if name.startswith('_') or name == 'process':
                continue
            
            # Collecter les fonctions utiles
            if inspect.isfunction(obj):
                functions[name] = obj
            # Collecter les classes utiles
            elif inspect.isclass(obj) and obj.__module__ == module_obj.__name__:
                classes[name] = obj
        
        # Fonction qui tente d'utiliser les fonctionnalités du module
        def generic_process(data, hook):
            """
            Processeur générique qui essaie d'utiliser les fonctionnalités du module
            en fonction des données d'entrée et du hook.
            """
            result = data.copy()
            
            # Journaliser les informations sur le hook et les données
            logger.debug(f"Generic processor for {module_obj.__name__}: hook={hook}, data_keys={list(data.keys()) if isinstance(data, dict) else 'non-dict'}")
            
            # Essayer d'invoquer des fonctions utiles selon le hook et les données
            try:
                # Si le module a une fonction spécifique au hook
                hook_fn_name = f"handle_{hook}"
                if hook_fn_name in functions:
                    logger.debug(f"Using specific hook handler {hook_fn_name}")
                    functions[hook_fn_name](result)
                    return result
                
                # Si le module a une classe avec le même nom que le module
                module_name = module_obj.__name__.split('.')[-1]
                if module_name in classes:
                    cls = classes[module_name]
                    instance = cls()
                    
                    # Chercher des méthodes pertinentes
                    if hasattr(instance, hook_fn_name):
                        logger.debug(f"Using class method {hook_fn_name}")
                        getattr(instance, hook_fn_name)(result)
                        return result
                    
                    if hasattr(instance, "process"):
                        logger.debug(f"Using class method process")
                        instance.process(result, hook)
                        return result
            except Exception as e:
                logger.warning(f"Error in generic processor for {module_obj.__name__}: {e}")
            
            # Tout simplement renvoyer les données inchangées s'il n'y a pas de méthode appropriée
            return result
        
        # Attacher des métadonnées à notre fonction
        generic_process.__module__ = module_obj.__name__
        generic_process.__name__ = "generic_process"
        generic_process.__doc__ = f"Auto-generated processor for module {module_obj.__name__}"
        
        return generic_process

    @staticmethod
    def detect_module_type(module_obj) -> str:
        """
        Détecte le type de module en fonction de sa structure.
        
        Args:
            module_obj: L'objet module Python
            
        Returns:
            Le type de module ('standard', 'class_based', 'function_based')
        """
        # Chercher une fonction process standard
        if hasattr(module_obj, 'process') and callable(getattr(module_obj, 'process')):
            return "standard"
        
        # Chercher une classe avec une méthode process
        classes = {name: obj for name, obj in inspect.getmembers(module_obj, inspect.isclass)
                  if obj.__module__ == module_obj.__name__}
        
        for cls_name, cls in classes.items():
            if hasattr(cls, 'process') and callable(getattr(cls, 'process')):
                return "class_based"
        
        # Chercher des fonctions utiles
        functions = {name: obj for name, obj in inspect.getmembers(module_obj, inspect.isfunction)
                    if not name.startswith('_')}
        
        if functions:
            return "function_based"
        
        return "unknown"
    
    @staticmethod
    def get_module_hooks(module_obj) -> List[str]:
        """
        Détecte automatiquement les hooks supportés par un module.
        
        Args:
            module_obj: L'objet module Python
            
        Returns:
            Liste des hooks supportés
        """
        hooks = set()
        default_hooks = ["process_request", "process_response"]
        
        # Chercher des fonctions de type handle_X
        for name, obj in inspect.getmembers(module_obj):
            if callable(obj) and name.startswith('handle_'):
                hook = name[7:]  # Retirer le préfixe 'handle_'
                hooks.add(hook)
        
        # Chercher dans les classes
        classes = {name: obj for name, obj in inspect.getmembers(module_obj, inspect.isclass)
                  if obj.__module__ == module_obj.__name__}
        
        for cls_name, cls in classes.items():
            for name, method in inspect.getmembers(cls, inspect.isfunction):
                if name.startswith('handle_'):
                    hook = name[7:]
                    hooks.add(hook)
        
        # Si MODULE_METADATA existe et contient des hooks
        if hasattr(module_obj, 'MODULE_METADATA') and isinstance(module_obj.MODULE_METADATA, dict):
            if 'hooks' in module_obj.MODULE_METADATA and isinstance(module_obj.MODULE_METADATA['hooks'], list):
                hooks.update(module_obj.MODULE_METADATA['hooks'])
        
        # Ajouter les hooks par défaut si aucun hook n'a été trouvé
        if not hooks:
            hooks.update(default_hooks)
        
        return list(hooks)

class ModuleLoader:
    """Charge dynamiquement les modules Python."""
    
    def __init__(self, modules_dir: str = MODULES_DIR, registry: ModuleRegistry = None):
        self.modules_dir = modules_dir
        self.registry = registry or ModuleRegistry()
    
    def load_module(self, module_path: str) -> Optional[ModuleInfo]:
        """
        Charge un module Python à partir de son chemin et extrait ses métadonnées.
        """
        try:
            if not module_path.endswith('.py') or os.path.basename(module_path).startswith('__'):
                return None
            
            module_name = os.path.basename(module_path).replace('.py', '')
            module_rel_path = os.path.relpath(module_path, PROJECT_ROOT)
            
            # Vérifier si le module est déjà chargé et à jour
            existing_module = self.registry.get_module(module_name)
            if existing_module and existing_module.path == module_rel_path:
                last_modified = os.path.getmtime(module_path)
                if existing_module.last_modified >= last_modified:
                    # Le module est déjà à jour
                    logger.debug(f"Module {module_name} is already up to date")
                    return existing_module
            
            # Charger le module
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if not spec or not spec.loader:
                logger.error(f"Failed to create spec for {module_path}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Extraire les métadonnées du module
            metadata = getattr(module, 'MODULE_METADATA', {})
            
            # Déterminer le type de module et procéder en conséquence
            module_type = ModuleAdapter.detect_module_type(module)
            
            # Obtenir la fonction de traitement appropriée
            process_fn = None
            
            if module_type == "standard":
                process_fn = getattr(module, 'process', None)
            elif module_type == "class_based":
                # Chercher une classe avec une méthode process
                classes = {name: obj for name, obj in inspect.getmembers(module, inspect.isclass)
                          if obj.__module__ == module.__name__}
                
                for cls_name, cls in classes.items():
                    # Préférer une classe portant le même nom que le module
                    if cls_name.lower() == module_name.lower() and hasattr(cls, 'process'):
                        instance = cls()
                        process_fn = instance.process
                        break
                
                # Si aucune classe privilégiée n'est trouvée, prendre la première avec une méthode process
                if not process_fn:
                    for cls_name, cls in classes.items():
                        if hasattr(cls, 'process'):
                            instance = cls()
                            process_fn = instance.process
                            break
            
            # Si aucune fonction process n'est trouvée, générer une fonction générique
            if not process_fn or not callable(process_fn):
                process_fn = ModuleAdapter.create_generic_processor(module)
                module_type = "auto_generated"
                logger.info(f"Created generic processor for module {module_name}")
            
            # Déterminer les hooks supportés
            hooks = metadata.get('hooks', ModuleAdapter.get_module_hooks(module))
            
            # Créer l'objet ModuleInfo
            module_info = ModuleInfo(
                name=module_name,
                path=module_rel_path,
                module_obj=module,
                enabled=metadata.get('enabled', True),
                priority=metadata.get('priority', 100),
                description=metadata.get('description', ''),
                version=metadata.get('version', '0.1'),
                dependencies=metadata.get('dependencies', []),
                hooks=hooks,
                processor=process_fn,
                creation_time=existing_module.creation_time if existing_module else time.time(),
                last_modified=os.path.getmtime(module_path),
                module_type=module_type
            )
            
            # Collecter les fonctions disponibles
            module_info.available_functions = {
                name: obj for name, obj in inspect.getmembers(module, callable)
                if not name.startswith('_') and obj.__module__ == module.__name__
            }
            
            # Enregistrer dans le registre
            self.registry.register_module(module_info)
            
            logger.info(f"Successfully loaded module {module_name} (v{module_info.version}) as {module_type} module")
            return module_info
            
        except Exception as e:
            logger.error(f"Error loading module {module_path}: {str(e)}")
            
            # Créer un module avec erreur si possible
            try:
                module_name = os.path.basename(module_path).replace('.py', '')
                module_rel_path = os.path.relpath(module_path, PROJECT_ROOT)
                error_module = ModuleInfo(
                    name=module_name,
                    path=module_rel_path,
                    enabled=False,
                    description=f"Error: {str(e)}",
                    last_modified=os.path.getmtime(module_path)
                )
                error_module.error = str(e)
                self.registry.register_module(error_module)
            except:
                pass
                
            return None
    
    def load_all_modules(self) -> Dict[str, ModuleInfo]:
        """
        Charge tous les modules Python présents dans le dossier des modules.
        """
        loaded_modules = {}
        for root, _, files in os.walk(self.modules_dir):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    module_path = os.path.join(root, file)
                    module_info = self.load_module(module_path)
                    if module_info:
                        loaded_modules[module_info.name] = module_info
                        
                        # Journaliser des informations sur le type de module
                        if module_info.module_type == "auto_generated":
                            logger.debug(f"Auto-generated processor used for module {module_info.name}")
        
        logger.info(f"Loaded {len(loaded_modules)} modules")
        return loaded_modules
    
    def reload_module(self, module_name: str) -> Optional[ModuleInfo]:
        """
        Recharge un module spécifique par son nom.
        """
        module_info = self.registry.get_module(module_name)
        if not module_info:
            logger.warning(f"Attempted to reload non-existent module '{module_name}'")
            return None
        
        module_path = os.path.join(PROJECT_ROOT, module_info.path)
        
        # Réinitialiser le cache du module
        if module_info.name in sys.modules:
            del sys.modules[module_info.name]
        
        return self.load_module(module_path)

class ModuleFileWatcher(FileSystemEventHandler):
    """Surveille les changements dans le dossier des modules."""
    
    def __init__(self, loader: ModuleLoader):
        super().__init__()
        self.loader = loader
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.py'):
            logger.info(f"New module detected: {event.src_path}")
            self.loader.load_module(event.src_path)
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.py'):
            logger.info(f"Module modified: {event.src_path}")
            self.loader.load_module(event.src_path)
    
    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith('.py'):
            module_name = os.path.basename(event.src_path).replace('.py', '')
            logger.info(f"Module deleted: {module_name}")
            self.loader.registry.unregister_module(module_name)

class ModuleManager:
    """
    Gère le chargement, la surveillance et l'exécution des modules d'amélioration.
    """
    
    def __init__(self):
        self.registry = ModuleRegistry()
        self.loader = ModuleLoader(registry=self.registry)
        self.file_watcher = ModuleFileWatcher(self.loader)
        self.observer = Observer()
        self.started = False
    
    def start(self):
        """Démarre le gestionnaire de modules."""
        if self.started:
            return
        
        # Charger initialement tous les modules
        self.loader.load_all_modules()
        
        # Démarrer la surveillance des fichiers
        self.observer.schedule(self.file_watcher, MODULES_DIR, recursive=True)
        self.observer.start()
        
        self.started = True
        logger.info("Module manager started")
    
    def stop(self):
        """Arrête le gestionnaire de modules."""
        if not self.started:
            return
        
        self.observer.stop()
        self.observer.join()
        
        self.started = False
        logger.info("Module manager stopped")
    
    def process_with_modules(self, 
                           request_data: Dict[str, Any], 
                           hook: str = 'process_request') -> Dict[str, Any]:
        """
        Traite une requête en utilisant tous les modules enregistrés pour un hook spécifique.
        Garantit toujours le retour d'un dictionnaire valide.
        """
        try:
            # Journaliser le type d'entrée pour le débogage
            logger.debug(f"process_with_modules input type for {hook}: {type(request_data)}")
            
            modules = self.registry.get_modules_by_hook(hook)
            
            # PROTECTION DE TYPE: S'assurer que les données d'entrée sont bien un dictionnaire
            if not isinstance(request_data, dict):
                logger.error(f"Invalid request_data type: {type(request_data)}, expected dict")
                
                # Convertir en dictionnaire selon le type
                if isinstance(request_data, str):
                    result = {"text": request_data}
                elif request_data is None:
                    result = {"text": ""}  # Valeur par défaut pour None
                else:
                    # Essayer de convertir en dictionnaire si possible
                    try:
                        result = dict(request_data)  # Tente de convertir en dict si c'est itérable
                    except (TypeError, ValueError):
                        result = {"data": str(request_data)}  # Fallback sécurisé
            else:
                # Créer une copie profonde pour éviter les effets de bord
                import copy
                
                # Utiliser deepcopy pour une véritable copie profonde
                try:
                    result = copy.deepcopy(request_data)
                except Exception as e:
                    # En cas d'échec de deepcopy, faire une copie manuelle
                    logger.warning(f"Deepcopy failed: {str(e)}, falling back to manual copy")
                    result = {}
                    # Copie manuelle pour les clés primaires
                    for key, value in request_data.items():
                        if isinstance(value, str):
                            result[key] = value
                        elif isinstance(value, dict):
                            # Copie récursive et sécurisée des dictionnaires
                            try:
                                result[key] = copy.deepcopy(value)
                            except:
                                # Si deepcopy échoue, faire une copie simple
                                result[key] = value.copy() if hasattr(value, 'copy') else value
                        elif isinstance(value, (list, tuple, set)):
                            # Copie récursive pour les collections
                            try:
                                result[key] = copy.deepcopy(value)
                            except:
                                # Si deepcopy échoue, faire une copie simple
                                result[key] = value.copy() if hasattr(value, 'copy') else list(value)
                        else:
                            # Pour les types immutables ou non copiables, les assigner directement
                            result[key] = value
            
            # Traitement par chaque module
            for module_info in modules:
                try:
                    if not module_info.processor:
                        logger.debug(f"Module {module_info.name} has no processor, skipping")
                        continue
                    
                    # Vérification que result est bien un dictionnaire avant de l'envoyer au module
                    if not isinstance(result, dict):
                        logger.warning(f"Result became non-dict before module {module_info.name}: {type(result)}")
                        result = {"text": str(result) if result is not None else ""}
                    
                    # Appliquer le traitement du module avec protection des entrées
                    try:
                        processed_result = module_info.processor(result.copy() if hasattr(result, 'copy') else result, hook)
                    except Exception as e:
                        logger.error(f"Module {module_info.name} processor raised exception: {str(e)}")
                        # Continuer avec le résultat actuel sans modifier
                        continue
                    
                    # VALIDATION DU RÉSULTAT: S'assurer que le résultat est toujours un dictionnaire
                    if processed_result is not None:
                        if isinstance(processed_result, dict):
                            result = processed_result
                        else:
                            logger.warning(f"Module {module_info.name} returned non-dict result: {type(processed_result)}")
                            # Si le module retourne une chaîne, la mettre dans le champ 'text'
                            if isinstance(processed_result, str):
                                # Conserver les autres clés du résultat précédent
                                prev_result = result.copy() if hasattr(result, 'copy') else {}
                                prev_result["text"] = processed_result
                                result = prev_result
                            else:
                                # Pour les autres types, les stocker dans un champ générique
                                # tout en préservant le dictionnaire précédent
                                prev_result = result.copy() if hasattr(result, 'copy') else {}
                                prev_result["processed_data"] = processed_result
                                result = prev_result
                except Exception as e:
                    logger.error(f"Error processing with module {module_info.name}: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # Continuer avec les autres modules même si l'un d'eux échoue
            
            # VALIDATION FINALE: S'assurer que le résultat est un dictionnaire valide
            if not isinstance(result, dict):
                logger.error(f"Final result is not a dict: {type(result)}. Converting to dict.")
                return {"text": str(result) if result is not None else ""}
            
            # S'assurer que le dictionnaire contient au moins une clé
            if len(result) == 0:
                logger.warning("Result dictionary is empty, adding default text key")
                result["text"] = ""
                
            return result
            
        except Exception as e:
            logger.error(f"Critical error in process_with_modules: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Retourner un dictionnaire minimal en cas d'erreur critique
            return {"text": request_data.get("text", "") if isinstance(request_data, dict) else str(request_data) if request_data is not None else ""}
        
        return result
    
    def get_module_info(self, module_name: str = None) -> Dict[str, Any]:
        """
        Renvoie des informations sur les modules chargés.
        
        Args:
            module_name: Nom du module spécifique ou None pour tous les modules
            
        Returns:
            Informations sur le(s) module(s)
        """
        if module_name:
            module = self.registry.get_module(module_name)
            if not module:
                return {"error": f"Module {module_name} not found"}
            
            return {
                "name": module.name,
                "description": module.description,
                "version": module.version,
                "enabled": module.enabled,
                "hooks": module.hooks,
                "type": module.module_type,
                "has_processor": module.processor is not None,
                "available_functions": list(getattr(module, "available_functions", {}).keys()),
                "error": module.error
            }
        else:
            modules_info = []
            for name, module in self.registry.modules.items():
                modules_info.append({
                    "name": module.name,
                    "description": module.description,
                    "version": module.version,
                    "enabled": module.enabled,
                    "type": module.module_type,
                    "hooks": module.hooks,
                    "has_processor": module.processor is not None,
                    "error": module.error
                })
            return {"modules": modules_info}

# Singleton pour la gestion des modules
_module_manager = None

def get_module_manager() -> ModuleManager:
    """Récupère l'instance singleton du gestionnaire de modules."""
    global _module_manager
    if _module_manager is None:
        _module_manager = ModuleManager()
    return _module_manager
