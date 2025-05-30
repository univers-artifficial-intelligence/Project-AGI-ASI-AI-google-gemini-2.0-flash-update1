"""
Script de test pour le système de modules d'amélioration.
Ce script permet de tester le chargement dynamique des modules et leur application.
"""

import json
import sys
import time
from module_manager import get_module_manager, ModuleInfo

def print_module_info(module_info: ModuleInfo):
    """Affiche les informations d'un module."""
    print(f"Module: {module_info.name} (v{module_info.version})")
    print(f"  Type: {module_info.module_type}")
    print(f"  Enabled: {module_info.enabled}")
    print(f"  Priority: {module_info.priority}")
    print(f"  Description: {module_info.description}")
    print(f"  Path: {module_info.path}")
    print(f"  Hooks: {', '.join(module_info.hooks) if module_info.hooks else 'None'}")
    print(f"  Has processor: {'Yes' if module_info.processor else 'No'}")
    print(f"  Dependencies: {', '.join(module_info.dependencies) if module_info.dependencies else 'None'}")
    print(f"  Last modified: {time.ctime(module_info.last_modified)}")
    if hasattr(module_info, 'available_functions'):
        print(f"  Available functions: {', '.join(module_info.available_functions.keys())}")
    print()

def test_module_loading():
    """Teste le chargement des modules."""
    print("=== TESTING MODULE LOADING ===")
    
    manager = get_module_manager()
    manager.start()
    
    # Afficher tous les modules chargés
    print("\nAll loaded modules:")
    modules = manager.registry.get_all_enabled_modules()
    if not modules:
        print("  No modules loaded.")
    else:
        for module in modules:
            print_module_info(module)
    
    # Vérifier les modules avec des processeurs auto-générés
    auto_generated_modules = [m for m in modules if m.module_type == "auto_generated"]
    if auto_generated_modules:
        print("\nModules with auto-generated processors:")
        for module in auto_generated_modules:
            print(f"  - {module.name}")
    
    # Tester le rechargement d'un module
    if modules:
        module_name = modules[0].name
        print(f"\nReloading module: {module_name}")
        reloaded = manager.loader.reload_module(module_name)
        if reloaded:
            print(f"  Module {module_name} successfully reloaded.")
        else:
            print(f"  Failed to reload module {module_name}.")
    
    return manager

def test_module_processing(manager):
    """Teste le traitement avec les modules."""
    print("\n=== TESTING MODULE PROCESSING ===")
    
    # Exemples de requêtes pour tester
    test_requests = [
        {
            "contents": [{
                "parts": [{
                    "text": "Pourquoi le ciel est-il bleu?"
                }]
            }],
            "generation_config": {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40
            }
        },
        {
            "contents": [{
                "parts": [{
                    "text": "Peux-tu me faire un plan pour organiser une fête d'anniversaire?"
                }]
            }],
            "generation_config": {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40
            }
        }
    ]
    
    for i, request in enumerate(test_requests):
        print(f"\nProcessing request #{i+1}:")
        print(f"  Original: {request['contents'][0]['parts'][0]['text']}")
        
        # Traiter avec les modules
        processed = manager.process_with_modules(request, "process_request")
        
        print(f"  Processed: {processed['contents'][0]['parts'][0]['text']}")
        
        # Vérifier si la requête a été identifiée comme nécessitant une planification
        if 'requires_planning' in processed:
            print(f"  Planning required: {processed['requires_planning']}")
            print(f"  Planning keywords: {processed.get('planning_keywords_detected', [])}")
        print()

def test_get_module_info(manager):
    """Teste la récupération d'informations sur les modules."""
    print("\n=== TESTING MODULE INFO API ===")
    
    # Obtenir des informations sur tous les modules
    info = manager.get_module_info()
    print(f"Total modules: {len(info['modules'])}")
    
    # Obtenir des informations sur un module spécifique
    modules = manager.registry.get_all_enabled_modules()
    if modules:
        module_name = modules[0].name
        module_info = manager.get_module_info(module_name)
        print(f"\nDetails for module '{module_name}':")
        for key, value in module_info.items():
            print(f"  {key}: {value}")

def main():
    """Fonction principale du test."""
    print("Module System Test\n")
    
    manager = test_module_loading()
    test_module_processing(manager)
    test_get_module_info(manager)
    
    print("\nTest completed. Press Ctrl+C to exit.")
    try:
        # Garder le programme en cours d'exécution pour surveiller les changements de fichiers
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.stop()
        print("\nTest stopped.")

if __name__ == "__main__":
    main()
