"""
Système de Raisonnement Analogique et Transfert Conceptuel
Ce module implémente un framework complet pour l'identification de patterns,
le transfert de connaissances, et la construction d'analogies conceptuelles.
"""

from typing import Dict, List, Tuple, Set, Any, Optional, Callable
import json
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class ConceptualEntity:
    """Représente une entité conceptuelle avec ses attributs et relations."""
    
    def __init__(self, name: str, attributes: Dict[str, Any] = None, relations: List[Dict] = None, domain: str = ""):
        self.name = name
        self.attributes = attributes or {}
        self.relations = relations or []
        self.domain = domain
    
    def __str__(self):
        return f"Entity({self.name}, domain={self.domain})"
    
    def __repr__(self):
        return self.__str__()
    
    def add_relation(self, relation_type: str, target: 'ConceptualEntity', 
                    strength: float = 1.0, metadata: Dict = None):
        """Ajoute une relation avec une autre entité."""
        relation = {
            "type": relation_type,
            "target": target,
            "strength": strength,
            "metadata": metadata or {}
        }
        self.relations.append(relation)
        
    def similarity_to(self, other_entity: 'ConceptualEntity', metric: str = "cosine") -> float:
        """Calcule la similarité avec une autre entité conceptuelle."""
        if metric == "cosine":
            return self._calculate_cosine_similarity(other_entity)
        elif metric == "jaccard":
            return self._calculate_jaccard_similarity(other_entity)
        else:
            return self._calculate_string_similarity(other_entity)
    
    def _calculate_cosine_similarity(self, other_entity: 'ConceptualEntity') -> float:
        """Calcule la similarité cosinus basée sur les attributs."""
        common_keys = set(self.attributes.keys()) & set(other_entity.attributes.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1 = self.attributes[key]
            val2 = other_entity.attributes[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Normaliser et calculer la similarité
                max_val = max(abs(val1), abs(val2), 1)
                sim = 1.0 - abs(val1 - val2) / max_val
                similarities.append(sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _calculate_jaccard_similarity(self, other_entity: 'ConceptualEntity') -> float:
        """Calcule la similarité de Jaccard."""
        attrs1 = set(self.attributes.keys())
        attrs2 = set(other_entity.attributes.keys())
        
        intersection = len(attrs1 & attrs2)
        union = len(attrs1 | attrs2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_string_similarity(self, other_entity: 'ConceptualEntity') -> float:
        """Calcule la similarité basée sur les attributs textuels."""
        common_keys = set(self.attributes.keys()) & set(other_entity.attributes.keys())
        if not common_keys:
            return 0.0
            
        text_similarity_scores = []
        
        for key in common_keys:
            val1 = str(self.attributes[key]).lower()
            val2 = str(other_entity.attributes[key]).lower()
            
            words1 = set(val1.split())
            words2 = set(val2.split())
            
            if words1 and words2:
                intersection = len(words1 & words2)
                union = len(words1 | words2)
                similarity = intersection / union
                text_similarity_scores.append(similarity)
        
        return sum(text_similarity_scores) / len(text_similarity_scores) if text_similarity_scores else 0.0


class PatternIdentifier:
    """Identifie des patterns structurels entre différents domaines."""
    
    def __init__(self):
        self.pattern_cache = {}
        
    def find_common_patterns(self, source_entities: List[ConceptualEntity], 
                           target_entities: List[ConceptualEntity]) -> List[Dict]:
        """Trouve des patterns communs entre deux ensembles d'entités."""
        patterns = []
        
        for s_entity in source_entities:
            best_match = None
            best_score = 0.0
            
            for t_entity in target_entities:
                score = s_entity.similarity_to(t_entity)
                if score > best_score:
                    best_score = score
                    best_match = t_entity
            
            if best_match and best_score > 0.3:
                patterns.append({
                    "source": s_entity.name,
                    "target": best_match.name,
                    "strength": best_score,
                    "type": "entity_mapping"
                })
        
        return patterns
    
    def identify_deep_patterns(self, domains_data: Dict, min_domains: int = 2) -> List[Dict]:
        """Identifie des patterns profonds à travers plusieurs domaines."""
        all_patterns = []
        domain_names = list(domains_data.keys())
        
        for i in range(len(domain_names)):
            for j in range(i+1, len(domain_names)):
                domain1 = domain_names[i]
                domain2 = domain_names[j]
                
                entities1 = domains_data[domain1]
                entities2 = domains_data[domain2]
                
                common = self.find_common_patterns(entities1, entities2)
                
                for pattern in common:
                    pattern["domains"] = [domain1, domain2]
                    all_patterns.append(pattern)
        
        # Filtrer par nombre minimum de domaines
        deep_patterns = [p for p in all_patterns if len(p.get("domains", [])) >= min_domains]
        return deep_patterns


class StructuralPatternIdentifier:
    """Composant responsable de l'identification de patterns structurels profonds."""
    
    def __init__(self, sensitivity: float = 0.7, min_pattern_size: int = 3):
        self.sensitivity = sensitivity
        self.min_pattern_size = min_pattern_size
        self.pattern_library = []
        self.graph_representations = {}
        
    def extract_structural_pattern(self, entities: List[ConceptualEntity]) -> nx.Graph:
        """Extrait un pattern structurel à partir d'un ensemble d'entités conceptuelles."""
        G = nx.Graph()
        
        # Ajouter les nœuds (entités)
        for entity in entities:
            G.add_node(entity.name, attributes=entity.attributes, domain=entity.domain)
            
        # Ajouter les arêtes (relations)
        for entity in entities:
            for relation in entity.relations:
                target = relation["target"]
                G.add_edge(
                    entity.name,
                    target.name,
                    type=relation["type"],
                    strength=relation["strength"]
                )
        
        return G
    
    def find_common_patterns(self, domain1_entities: List[ConceptualEntity], 
                            domain2_entities: List[ConceptualEntity]) -> List[Dict]:
        """Trouve des patterns communs entre deux domaines."""
        pattern1 = self.extract_structural_pattern(domain1_entities)
        pattern2 = self.extract_structural_pattern(domain2_entities)
        
        common_patterns = []
        
        # Utiliser l'algorithme VF2 pour la recherche d'isomorphismes de sous-graphes
        try:
            graph_matcher = nx.algorithms.isomorphism.GraphMatcher(
                pattern1, pattern2,
                node_match=lambda n1, n2: self._node_similarity(n1, n2) > self.sensitivity,
                edge_match=lambda e1, e2: self._edge_similarity(e1, e2) > self.sensitivity
            )
            
            # Trouver tous les sous-graphes isomorphes
            for mapping in graph_matcher.subgraph_isomorphisms_iter():
                if len(mapping) >= self.min_pattern_size:
                    common_patterns.append({
                        "mapping": mapping,
                        "size": len(mapping),
                        "confidence": self._calculate_pattern_confidence(pattern1, pattern2, mapping)
                    })
        except:
            # Fallback simple si l'algorithme VF2 échoue
            pass
        
        # Trier par taille et confiance
        common_patterns.sort(key=lambda x: (x["size"], x["confidence"]), reverse=True)
        
        return common_patterns
    
    def _node_similarity(self, node1_attr: Dict, node2_attr: Dict) -> float:
        """Calcule la similarité entre deux nœuds du graphe."""
        attrs1 = node1_attr.get("attributes", {})
        attrs2 = node2_attr.get("attributes", {})
        
        common_keys = set(attrs1.keys()) & set(attrs2.keys())
        if not common_keys:
            return 0.0
            
        similarities = []
        for key in common_keys:
            val1 = attrs1[key]
            val2 = attrs2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                max_val = max(abs(val1), abs(val2))
                if max_val > 0:
                    sim = 1.0 - abs(val1 - val2) / max_val
                else:
                    sim = 1.0
            elif isinstance(val1, str) and isinstance(val2, str):
                words1 = set(val1.lower().split())
                words2 = set(val2.lower().split())
                
                if words1 and words2:
                    intersection = len(words1 & words2)
                    union = len(words1 | words2)
                    sim = intersection / union
                else:
                    sim = 0.0
            else:
                sim = 0.0
                
            similarities.append(sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _edge_similarity(self, edge1_attr: Dict, edge2_attr: Dict) -> float:
        """Calcule la similarité entre deux arêtes du graphe."""
        if edge1_attr.get("type") != edge2_attr.get("type"):
            return 0.0
            
        strength1 = edge1_attr.get("strength", 0.0)
        strength2 = edge2_attr.get("strength", 0.0)
        
        return 1.0 - abs(strength1 - strength2)
    
    def _calculate_pattern_confidence(self, graph1: nx.Graph, graph2: nx.Graph, 
                                     mapping: Dict[str, str]) -> float:
        """Calcule la confiance dans la correspondance du pattern."""
        node_similarities = []
        edge_similarities = []
        
        # Similarités des nœuds
        for node1, node2 in mapping.items():
            if node1 in graph1.nodes and node2 in graph2.nodes:
                node_sim = self._node_similarity(graph1.nodes[node1], graph2.nodes[node2])
                node_similarities.append(node_sim)
        
        # Similarités des arêtes
        for node1, node2 in mapping.items():
            if node1 in graph1 and node2 in graph2:
                for neighbor1 in graph1.neighbors(node1):
                    if neighbor1 in mapping:
                        neighbor2 = mapping[neighbor1]
                        if graph2.has_edge(node2, neighbor2):
                            edge_sim = self._edge_similarity(
                                graph1.edges[node1, neighbor1],
                                graph2.edges[node2, neighbor2]
                            )
                            edge_similarities.append(edge_sim)
        
        node_conf = sum(node_similarities) / len(node_similarities) if node_similarities else 0.0
        edge_conf = sum(edge_similarities) / len(edge_similarities) if edge_similarities else 0.0
        
        return 0.6 * node_conf + 0.4 * edge_conf


class KnowledgeTransfer:
    """Composant responsable du transfert créatif de connaissances."""
    
    def __init__(self, adaptation_strategies: Dict[str, Callable] = None):
        self.adaptation_strategies = adaptation_strategies or {
            "procedural": self._adapt_procedural_knowledge,
            "declarative": self._adapt_declarative_knowledge,
            "strategic": self._adapt_strategic_knowledge,
            "heuristic": self._adapt_heuristic_knowledge
        }
        self.transfer_history = []
        self.successful_transfers = {}
        
    def transfer_knowledge(self, source_domain: Dict, target_domain: Dict, 
                          pattern_identifier: StructuralPatternIdentifier,
                          knowledge_type: str = "auto") -> Dict:
        """Transfère des connaissances d'un domaine source vers un domaine cible."""
        source_entities = source_domain.get("entities", [])
        target_entities = target_domain.get("entities", [])
        
        common_patterns = pattern_identifier.find_common_patterns(source_entities, target_entities)
        
        if not common_patterns:
            return {"success": False, "message": "Aucun pattern structurel commun trouvé pour le transfert"}
        
        best_pattern = common_patterns[0]
        mapping = best_pattern["mapping"]
        
        if knowledge_type == "auto":
            knowledge_type = self._determine_knowledge_type(source_domain)
        
        if knowledge_type not in self.adaptation_strategies:
            return {"success": False, "message": f"Type de connaissance non supporté: {knowledge_type}"}
        
        source_knowledge = source_domain.get("knowledge", {})
        adaptation_strategy = self.adaptation_strategies[knowledge_type]
        adapted_knowledge = adaptation_strategy(source_knowledge, mapping, target_domain)
        
        transfer_record = {
            "source": source_domain.get("name"),
            "target": target_domain.get("name"),
            "knowledge_type": knowledge_type,
            "success": True
        }
        self.transfer_history.append(transfer_record)
        
        return {
            "success": True,
            "adapted_knowledge": adapted_knowledge,
            "mapping": mapping,
            "confidence": best_pattern["confidence"],
            "knowledge_type": knowledge_type
        }
    
    def evaluate_transfer(self, transfer_result: Dict, evaluation_criteria: Dict = None) -> Dict:
        """Évalue la qualité d'un transfert de connaissances."""
        criteria = evaluation_criteria or {
            "structural_alignment": 0.4,
            "contextual_relevance": 0.3,
            "novelty": 0.2,
            "usability": 0.1
        }
        
        scores = {}
        scores["structural_alignment"] = transfer_result.get("confidence", 0.0)
        scores["contextual_relevance"] = 0.7
        scores["novelty"] = 0.65
        scores["usability"] = 0.8
        
        weighted_score = sum(score * criteria[criterion] 
                           for criterion, score in scores.items())
        
        quality_interpretation = "Excellent" if weighted_score > 0.8 else \
                               "Bon" if weighted_score > 0.6 else \
                               "Moyen" if weighted_score > 0.4 else \
                               "Faible"
        
        return {
            "quality": weighted_score,
            "quality_level": quality_interpretation,
            "detailed_scores": scores,
            "message": f"Transfert évalué comme {quality_interpretation} (score: {weighted_score:.2f})"
        }
    
    def _determine_knowledge_type(self, domain: Dict) -> str:
        """Détermine le type de connaissance automatiquement."""
        knowledge = domain.get("knowledge", {})
        
        if "procedures" in knowledge or "steps" in knowledge or "methods" in knowledge:
            return "procedural"
        elif "facts" in knowledge or "concepts" in knowledge or "definitions" in knowledge:
            return "declarative"
        elif "strategies" in knowledge or "approaches" in knowledge or "plans" in knowledge:
            return "strategic"
        elif "rules_of_thumb" in knowledge or "shortcuts" in knowledge:
            return "heuristic"
        else:
            return "general"
    
    def _adapt_procedural_knowledge(self, source_knowledge: Dict, mapping: Dict, target_domain: Dict) -> Dict:
        """Adapte des connaissances procédurales au domaine cible."""
        procedures = source_knowledge.get("procedures", [])
        adapted_procedures = []
        
        for procedure in procedures:
            adapted_procedure = self._adapt_term_to_target_domain(str(procedure), target_domain)
            adapted_procedures.append(adapted_procedure)
        
        return {"procedures": adapted_procedures}
    
    def _adapt_declarative_knowledge(self, source_knowledge: Dict, mapping: Dict, target_domain: Dict) -> Dict:
        """Adapte des connaissances déclaratives au domaine cible."""
        facts = source_knowledge.get("facts", [])
        concepts = source_knowledge.get("concepts", {})
        
        adapted_facts = [self._adapt_term_to_target_domain(fact, target_domain) for fact in facts]
        
        adapted_concepts = {}
        for concept_name, concept_def in concepts.items():
            adapted_name = self._adapt_term_to_target_domain(concept_name, target_domain)
            adapted_def = self._adapt_term_to_target_domain(str(concept_def), target_domain)
            adapted_concepts[adapted_name] = adapted_def
        
        return {
            "facts": adapted_facts,
            "concepts": adapted_concepts
        }
    
    def _adapt_strategic_knowledge(self, source_knowledge: Dict, mapping: Dict, target_domain: Dict) -> Dict:
        """Adapte des connaissances stratégiques au domaine cible."""
        strategies = source_knowledge.get("strategies", [])
        adapted_strategies = []
        
        for strategy in strategies:
            adapted_strategy = self._adapt_term_to_target_domain(str(strategy), target_domain)
            adapted_strategies.append(adapted_strategy)
        
        return {"strategies": adapted_strategies}
    
    def _adapt_heuristic_knowledge(self, source_knowledge: Dict, mapping: Dict, target_domain: Dict) -> Dict:
        """Adapte des connaissances heuristiques au domaine cible."""
        heuristics = source_knowledge.get("heuristics", [])
        adapted_heuristics = []
        
        for heuristic in heuristics:
            adapted_heuristic = self._adapt_term_to_target_domain(str(heuristic), target_domain)
            adapted_heuristics.append(adapted_heuristic)
        
        return {"heuristics": adapted_heuristics}
    
    def _adapt_term_to_target_domain(self, term: str, target_domain: Dict, creativity: float = 0.5) -> str:
        """Adapte un terme ou phrase au domaine cible."""
        if not isinstance(term, str):
            return str(term)
            
        target_vocabulary = target_domain.get("vocabulary", {})
        domain_name = target_domain.get("name", "")
        
        words = term.split()
        adapted_words = []
        
        for word in words:
            adapted_word = target_vocabulary.get(word.lower(), word)
            adapted_words.append(adapted_word)
        
        adapted_term = " ".join(adapted_words)
        
        if creativity > 0.7 and domain_name:
            adapted_term = f"{adapted_term} (dans le contexte de {domain_name})"
        
        return adapted_term


class MultiLevelAnalogicalReasoning:
    """Composant responsable du raisonnement par analogie à plusieurs niveaux."""
    
    def __init__(self):
        self.levels = {
            "surface": {"weight": 0.2, "analyzer": self._analyze_surface_level},
            "structure": {"weight": 0.5, "analyzer": self._analyze_structural_level},
            "function": {"weight": 0.3, "analyzer": self._analyze_functional_level}
        }
        self.analogy_library = []
        
    def analyze_analogy(self, source: Dict, target: Dict, focus_levels: List[str] = None) -> Dict:
        """Analyse une analogie entre domaine source et cible à plusieurs niveaux."""
        levels_to_analyze = focus_levels or list(self.levels.keys())
        
        invalid_levels = [level for level in levels_to_analyze if level not in self.levels]
        if invalid_levels:
            return {
                "success": False,
                "message": f"Niveaux d'analyse invalides: {', '.join(invalid_levels)}"
            }
        
        analysis_results = {}
        total_weight = 0
        
        for level in levels_to_analyze:
            level_info = self.levels[level]
            analyzer = level_info["analyzer"]
            weight = level_info["weight"]
            
            level_result = analyzer(source, target)
            analysis_results[level] = level_result
            total_weight += weight
        
        normalized_weights = {level: self.levels[level]["weight"] / total_weight 
                             for level in levels_to_analyze}
        
        global_score = sum(analysis_results[level]["score"] * normalized_weights[level] 
                          for level in levels_to_analyze)
        
        key_mappings = self._extract_key_mappings(analysis_results)
        
        analogy_result = {
            "success": True,
            "global_score": global_score,
            "level_scores": {level: results["score"] for level, results in analysis_results.items()},
            "key_mappings": key_mappings,
            "detailed_analysis": analysis_results
        }
        
        self.analogy_library.append({
            "source_domain": source.get("name", "unknown"),
            "target_domain": target.get("name", "unknown"),
            "result": analogy_result,
            "timestamp": "2025-05-28"
        })
        
        return analogy_result
    
    def generate_multi_level_analogy(self, source: Dict, target: Dict, 
                                    desired_strength: float = 0.7) -> Dict:
        """Génère une analogie complète entre domaine source et cible."""
        analysis = self.analyze_analogy(source, target)
        
        if not analysis.get("success", False):
            return analysis
        
        summary = self._generate_analogy_summary(analysis, source, target)
        
        return {
            "summary": summary,
            "strength": analysis["global_score"],
            "detailed_analysis": analysis["detailed_analysis"]
        }
    
    def evaluate_analogy_quality(self, analogy: Dict, criteria: Dict = None) -> Dict:
        """Évalue la qualité d'une analogie."""
        default_criteria = {
            "coherence": 0.3,
            "relevance": 0.25,
            "coverage": 0.2,
            "clarity": 0.15,
            "novelty": 0.1
        }
        
        eval_criteria = criteria or default_criteria
        
        scores = {}
        scores["coherence"] = 0.7
        scores["relevance"] = analogy.get("strength", 0.0)
        scores["coverage"] = 0.6
        scores["clarity"] = 0.8
        scores["novelty"] = 0.7
        
        weighted_score = sum(score * eval_criteria[criterion] 
                           for criterion, score in scores.items())
        
        quality_level = "Excellente" if weighted_score > 0.8 else \
                       "Bonne" if weighted_score > 0.6 else \
                       "Moyenne" if weighted_score > 0.4 else \
                       "Faible"
        
        return {
            "overall_quality": weighted_score,
            "quality_level": quality_level,
            "detailed_scores": scores,
            "improvement_suggestions": []
        }
    
    def _analyze_surface_level(self, source: Dict, target: Dict) -> Dict:
        """Analyse l'analogie au niveau de surface."""
        source_entities = source.get("entities", [])
        target_entities = target.get("entities", [])
        
        mappings = []
        
        for s_entity in source_entities:
            best_match = None
            best_score = 0.0
            
            for t_entity in target_entities:
                score = self._calculate_entity_similarity(s_entity, t_entity)
                if score > best_score:
                    best_score = score
                    best_match = t_entity
            
            if best_match and best_score > 0.3:
                mappings.append({
                    "source": s_entity.name,
                    "target": best_match.name,
                    "strength": best_score
                })
        
        avg_mapping_strength = sum(m["strength"] for m in mappings) / len(mappings) if mappings else 0.0
        coverage = len(mappings) / len(source_entities) if source_entities else 0.0
        
        surface_score = 0.7 * avg_mapping_strength + 0.3 * coverage
        
        return {
            "score": surface_score,
            "mappings": mappings,
            "coverage": coverage
        }
    
    def _analyze_structural_level(self, source: Dict, target: Dict) -> Dict:
        """Analyse l'analogie au niveau structurel."""
        source_entities = source.get("entities", [])
        target_entities = target.get("entities", [])
        
        mappings = []
        structural_scores = []
        
        for s_entity in source_entities:
            best_match = None
            best_score = 0.0
            
            for t_entity in target_entities:
                score = self._calculate_entity_similarity(s_entity, t_entity)
                if score > best_score:
                    best_score = score
                    best_match = t_entity
            
            if best_match and best_score > 0.3:
                mappings.append({
                    "source": s_entity.name,
                    "target": best_match.name,
                    "strength": best_score
                })
        
        # Calculer la préservation des relations
        for s_entity in source_entities:
            if s_entity.name not in mapping_dict:
                continue
            
            t_entity_name = mapping_dict[s_entity.name]
            
            # Compter les relations préservées
            preserved_relations = 0
            total_relations = len(s_entity.relations)
            
            for s_relation in s_entity.relations:
                target_rel = s_relation["target"]
                if target_rel.name in mapping_dict:
                    preserved_relations += 1
            
            if total_relations > 0:
                preservation_score = preserved_relations / total_relations
                structural_scores.append(preservation_score)
                mappings.append({
                    "source": s_entity.name,
                    "target": t_entity_name,
                    "strength": preservation_score,
                    "preserved_relations": preserved_relations,
                    "total_relations": total_relations
                })

        avg_structural_score = sum(structural_scores) / len(structural_scores) if structural_scores else 0.0
        coverage = len(mappings) / len(source_entities) if source_entities else 0.0
        
        structural_score = 0.8 * avg_structural_score + 0.2 * coverage
        
        return {
            "score": structural_score,
            "mappings": mappings,
            "coverage": coverage
        }
    
    def _analyze_functional_level(self, source: Dict, target: Dict) -> Dict:
        """Analyse l'analogie au niveau fonctionnel."""
        source_functions = self._extract_functions_from_entities(source.get("entities", []))
        target_functions = self._extract_functions_from_entities(target.get("entities", []))
        
        mappings = []
        
        for s_function in source_functions:
            best_match = None
            best_score = 0.0
            
            for t_function in target_functions:
                # Calculer la similarité fonctionnelle
                s_desc = s_function.get("description", "")
                t_desc = t_function.get("description", "")
                
                if s_desc and t_desc:
                    # Similarité textuelle simple
                    s_words = set(s_desc.lower().split())
                    t_words = set(t_desc.lower().split())
                    
                    intersection = len(s_words & t_words)
                    union = len(s_words | t_words)
                    similarity = intersection / union if union > 0 else 0.0
                    
                    # Vérifier aussi les objectifs
                    s_goal = s_function.get("goal", "")
                    t_goal = t_function.get("goal", "")
                    
                    if s_goal and t_goal:
                        s_goal_words = set(s_goal.lower().split())
                        t_goal_words = set(t_goal.lower().split())
                        
                        goal_intersection = len(s_goal_words & t_goal_words)
                        goal_union = len(s_goal_words | t_goal_words)
                        goal_sim = goal_intersection / goal_union if goal_union > 0 else 0.0
                        
                        # Combiner les similarités (plus de poids aux objectifs)
                        similarity = 0.4 * similarity + 0.6 * goal_sim
                    
                    # Mettre à jour le meilleur match
                    if similarity > best_score:
                        best_score = similarity
                        best_match = t_function
            
            # Ajouter le mapping si suffisamment fort
            if best_match and best_score > 0.3:
                mappings.append({
                    "source": s_function.get("name", "unknown"),
                    "target": best_match.get("name", "unknown"),
                    "strength": best_score,
                    "source_goal": s_function.get("goal", ""),
                    "target_goal": best_match.get("goal", "")
                })
        
        avg_mapping_strength = sum(m["strength"] for m in mappings) / len(mappings) if mappings else 0.0
        coverage = len(mappings) / len(source_functions) if source_functions else 0.0
        
        functional_score = 0.7 * avg_mapping_strength + 0.3 * coverage
        
        return {
            "score": functional_score,
            "mappings": mappings,
            "coverage": coverage
        }
    
    def _extract_functions_from_entities(self, entities: List[ConceptualEntity]) -> List[Dict]:
        """Extrait les aspects fonctionnels des entités."""
        functions = []
        
        for entity in entities:
            # Chercher des attributs liés aux fonctions
            function_attrs = {}
            
            for attr_name, attr_value in entity.attributes.items():
                if any(func_term in attr_name.lower() for func_term in 
                      ["function", "purpose", "goal", "objective", "role"]):
                    function_attrs[attr_name] = attr_value
            
            if function_attrs:
                # Créer une représentation fonctionnelle
                function_desc = " ".join(str(v) for v in function_attrs.values())
                
                functions.append({
                    "name": f"{entity.name}_function",
                    "description": function_desc,
                    "goal": function_attrs.get("goal", function_attrs.get("purpose", "")),
                    "entity": entity.name
                })
        
        return functions
    
    def _extract_key_mappings(self, analysis_results: Dict) -> List[Dict]:
        """Extrait les correspondances clés à travers les différents niveaux d'analyse."""
        all_mappings = {}
        
        for level, results in analysis_results.items():
            for mapping in results.get("mappings", []):
                source = mapping["source"]
                target = mapping["target"]
                
                if source not in all_mappings:
                    all_mappings[source] = {
                        "source": source,
                        "target": target,
                        "levels": {},
                        "average_strength": 0.0
                    }
                
                all_mappings[source]["levels"][level] = mapping["strength"]
        
        for source, mapping_data in all_mappings.items():
            strengths = mapping_data["levels"].values()
            mapping_data["average_strength"] = sum(strengths) / len(strengths) if strengths else 0.0
        
        key_mappings = list(all_mappings.values())
        key_mappings.sort(key=lambda x: x["average_strength"], reverse=True)
        
        return key_mappings
    
    def _generate_analogy_summary(self, analysis: Dict, source: Dict, target: Dict) -> str:
        """Génère un résumé global de l'analogie."""
        source_name = source.get("name", "domaine source")
        target_name = target.get("name", "domaine cible")
        global_score = analysis.get("global_score", 0.0)
        
        if global_score > 0.8:
            quality = "très forte"
        elif global_score > 0.6:
            quality = "forte"
        elif global_score > 0.4:
            quality = "modérée"
        else:
            quality = "faible"
            
        return f"L'analogie entre {source_name} et {target_name} est {quality} (score: {global_score:.2f})."
    
    def _calculate_entity_similarity(self, entity1: ConceptualEntity, entity2: ConceptualEntity) -> float:
        """Calcule la similarité entre deux entités."""
        return entity1.similarity_to(entity2)


class ConceptualMetaphorBuilder:
    """Générateur de métaphores conceptuelles."""
    
    def __init__(self):
        self.metaphor_templates = {
            "process": [
                "{target} est comme {source}, où {mapping}.",
                "Pensez à {target} comme à {source} : {mapping}."
            ],
            "structure": [
                "{target} est structuré comme {source}, où {mapping}.",
                "La structure de {target} ressemble à celle de {source}, avec {mapping}."
            ],
            "function": [
                "{target} fonctionne comme {source}, où {mapping}.",
                "Le rôle de {target} est similaire à celui de {source} : {mapping}."
            ]
        }
        self.metaphor_library = {}
        
    def generate_metaphor(self, target_concept: Dict, source_domain: Dict, 
                         metaphor_type: str = "auto") -> Dict:
        """Génère une métaphore conceptuelle."""
        if metaphor_type == "auto":
            metaphor_type = "structure"
        
        if metaphor_type not in self.metaphor_templates:
            return {
                "success": False,
                "message": f"Type de métaphore non pris en charge: {metaphor_type}"
            }
        
        # Générer une métaphore simple
        target_name = target_concept.get("name", "le concept")
        source_name = source_domain.get("name", "le domaine source")
        
        template = random.choice(self.metaphor_templates[metaphor_type])
        mapping = "ils partagent des caractéristiques similaires"
        
        metaphor_text = template.format(
            target=target_name,
            source=source_name,
            mapping=mapping
        )
        
        metaphor = {
            "text": metaphor_text,
            "type": metaphor_type,
            "target_concept": target_name,
            "source_domain": source_name,
            "mappings": [],
            "explanations": []
        }
        
        return {
            "success": True,
            "metaphor": metaphor
        }


class AbstractGeneralization:
    """Composant responsable de la généralisation abstraite."""
    
    def __init__(self, generalization_threshold: float = 0.7, min_examples: int = 3):
        self.generalization_threshold = generalization_threshold
        self.min_examples = min_examples
        self.abstraction_hierarchy = {}
        self.generalization_history = []
    
    def extract_common_patterns(self, examples: List[Dict], 
                               features_to_consider: List[str] = None) -> Dict:
        """Extrait des patterns communs à partir d'exemples concrets."""
        if len(examples) < self.min_examples:
            return {
                "success": False,
                "message": f"Nombre insuffisant d'exemples ({len(examples)}/{self.min_examples} requis)",
                "patterns": {}
            }
        
        if not features_to_consider:
            all_features = set()
            for example in examples:
                all_features.update(example.keys())
            
            feature_counts = {feature: 0 for feature in all_features}
            for example in examples:
                for feature in example:
                    if feature in feature_counts:
                        feature_counts[feature] += 1
            
            min_presence = len(examples) / 2
            features_to_consider = [f for f, count in feature_counts.items() 
                                  if count >= min_presence]
        
        patterns = {}
        
        for feature in features_to_consider:
            values = [example.get(feature) for example in examples if feature in example]
            valid_values = [v for v in values if v is not None]
            
            if not valid_values:
                continue
            
            if all(isinstance(v, (int, float)) for v in valid_values):
                avg_value = sum(valid_values) / len(valid_values)
                min_value = min(valid_values)
                max_value = max(valid_values)
                
                patterns[feature] = {
                    "type": "numeric",
                    "average": avg_value,
                    "range": (min_value, max_value),
                    "consistency": 1.0 - (max_value - min_value) / max(1.0, max_value) if max_value > 0 else 1.0
                }
            else:
                patterns[feature] = {
                    "type": "other",
                    "values": valid_values,
                    "consistency": 0.5
                }
        
        pattern_consistencies = [p["consistency"] for p in patterns.values()]
        overall_consistency = sum(pattern_consistencies) / len(pattern_consistencies) if pattern_consistencies else 0.0
        
        return {
            "success": True,
            "patterns": patterns,
            "overall_consistency": overall_consistency,
            "num_examples": len(examples),
            "features_analyzed": features_to_consider
        }


class AnalogicalReasoningSystem:
    """Système principal de raisonnement analogique."""
    
    def __init__(self):
        self.domains = {}
        self.pattern_identifier = StructuralPatternIdentifier()
        self.knowledge_transfer = KnowledgeTransfer()
        self.multilevel_reasoning = MultiLevelAnalogicalReasoning()
        self.metaphor_builder = ConceptualMetaphorBuilder()
        self.generalizer = AbstractGeneralization()
    
    def register_domain(self, domain_name: str, entities: List[ConceptualEntity] = None,
                       knowledge: Dict = None, description: str = None) -> Dict:
        """Enregistre un nouveau domaine."""
        self.domains[domain_name] = {
            "name": domain_name,
            "entities": entities or [],
            "knowledge": knowledge or {},
            "description": description or ""
        }
        return {"success": True, "message": f"Domaine {domain_name} enregistré"}
    
    def find_analogy(self, source_domain: str, target_domain: str) -> Dict:
        """Trouve une analogie entre deux domaines."""
        if source_domain not in self.domains:
            return {"success": False, "message": f"Domaine source '{source_domain}' non trouvé"}
        
        if target_domain not in self.domains:
            return {"success": False, "message": f"Domaine cible '{target_domain}' non trouvé"}
        
        source = self.domains[source_domain]
        target = self.domains[target_domain]
        
        return self.multilevel_reasoning.analyze_analogy(source, target)
    
    def transfer_knowledge(self, source_domain: str, target_domain: str) -> Dict:
        """Transfère des connaissances entre domaines."""
        if source_domain not in self.domains or target_domain not in self.domains:
            return {"success": False, "message": "Un ou plusieurs domaines non trouvés"}
        
        source = self.domains[source_domain]
        target = self.domains[target_domain]
        
        return self.knowledge_transfer.transfer_knowledge(source, target, self.pattern_identifier)
    
    def generate_metaphor(self, target_concept: str, source_domain: str) -> Dict:
        """Génère une métaphore pour expliquer un concept."""
        if source_domain not in self.domains:
            return {"success": False, "message": f"Domaine source '{source_domain}' non trouvé"}
        
        target_concept_dict = {"name": target_concept}
        source_domain_dict = self.domains[source_domain]
        
        return self.metaphor_builder.generate_metaphor(target_concept_dict, source_domain_dict)
    
    def export_domain_knowledge(self, domain_name: str, format_type: str = "json") -> Dict:
        """Exporte la connaissance d'un domaine."""
        if domain_name not in self.domains:
            return {"success": False, "message": "Domaine non trouvé"}
        
        if format_type == "json":
            return {"success": True, "data": json.dumps(self.domains[domain_name], default=str)}
        elif format_type == "text":
            return {"success": True, "data": str(self.domains[domain_name])}
        else:
            return {"success": False, "message": "Format non supporté"}