"""
Test du module d'amélioration de la récupération de mémoire
"""

import unittest
import sys
import os
import json
from typing import Dict, Any
import datetime

# Ajouter le répertoire parent au chemin pour l'import des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.memory_retrieval_enhancer import enhance_memory_instructions, enrich_system_prompt, process
from memory_engine import MemoryEngine

class MockMemoryEngine:
    def get_recent_conversations(self, **kwargs):
        return [
            {
                'content': 'Ceci est une conversation test',
                'time_ago': 'il y a 5 minutes'
            }
        ]
    
    def get_long_term_memories(self, **kwargs):
        return [
            {
                'category': 'preference',
                'content': 'L\'utilisateur préfère le bleu'
            }
        ]
    
    def get_memory_context(self, **kwargs):
        return "Mémoire des conversations récentes:\n1. il y a 5 minutes: Ceci est une conversation test"

class TestMemoryRetrievalEnhancer(unittest.TestCase):
    
    def setUp(self):
        # Remplacer temporairement l'instance de MemoryEngine par notre mock
        import modules.memory_retrieval_enhancer as module
        self.original_memory_engine = module.memory_engine
        module.memory_engine = MockMemoryEngine()
        
    def tearDown(self):
        # Restaurer l'instance originale
        import modules.memory_retrieval_enhancer as module
        module.memory_engine = self.original_memory_engine
    
    def test_enhance_memory_instructions(self):
        # Préparer les données de test
        data = {
            'user_id': 1,
            'session_id': 'test_session',
            'context': {}
        }
        
        # Appeler la fonction à tester
        result = enhance_memory_instructions(data)
        
        # Vérifier les résultats
        self.assertTrue('instructions' in result['context'])
        self.assertTrue(isinstance(result['context']['instructions'], list))
        self.assertTrue(len(result['context']['instructions']) > 0)
        self.assertTrue('memory_instructions_enhanced' in result['context'])
        self.assertTrue(result['context']['memory_instructions_enhanced'])
        
        # Vérifier le contenu des instructions
        found_memory_instruction = False
        for instruction in result['context']['instructions']:
            if "INSTRUCTION CRITIQUE DE MÉMOIRE" in instruction:
                found_memory_instruction = True
                break
                
        self.assertTrue(found_memory_instruction)
        
    def test_enrich_system_prompt(self):
        # Préparer les données de test
        data = {
            'user_id': 1,
            'system_prompt': 'Instruction système originale'
        }
        
        # Appeler la fonction à tester
        result = enrich_system_prompt(data)
        
        # Vérifier les résultats
        self.assertTrue('system_prompt' in result)
        self.assertIn("INSTRUCTION CRITIQUE DE MÉMOIRE", result['system_prompt'])
        self.assertTrue(result['metadata']['memory_prompt_enriched'])
        
    def test_process_function(self):
        # Préparer les données de test
        data = {
            'user_id': 1,
            'session_id': 'test_session',
            'context': {},
            'system_prompt': 'Instruction système originale'
        }
        
        # Appeler la fonction à tester
        result = process(data, 'process_request')
        
        # Vérifier les résultats
        self.assertTrue('instructions' in result['context'])
        self.assertIn("INSTRUCTION CRITIQUE DE MÉMOIRE", result['system_prompt'])
        self.assertTrue(result['context']['memory_instructions_enhanced'])
        self.assertTrue(result['metadata']['memory_prompt_enriched'])
        
if __name__ == '__main__':
    unittest.main()
