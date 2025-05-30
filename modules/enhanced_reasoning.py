"""
Module d'amélioration du raisonnement multi-niveaux et hiérarchique pour Gemini.
Ce module implémente un système de raisonnement avancé avec plusieurs niveaux d'abstraction,
logiques hybrides et décomposition récursive des problèmes.

Fonctionnalités principales :
- Raisonnement multi-niveaux (opérationnel, tactique, stratégique)
- Logiques hybrides (déductive, inductive, abductive, analogique)
- Décomposition récursive des problèmes complexes
- Système d'apprentissage et d'optimisation
- Gestion de l'incertitude et des conflits
"""

import random
import re
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from enum import Enum, auto
from dataclasses import dataclass, field
import copy
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TypeLogique(Enum):
    """Types de logique supportés par le système de raisonnement."""
    DEDUCTIVE = "déductive"
    INDUCTIVE = "inductive" 
    ABDUCTIVE = "abductive"
    ANALOGIQUE = "analogique"
    FLOUE = "floue"
    TEMPORELLE = "temporelle"
    MODALE = "modale"
    PROBABILISTE = "probabiliste"

class NiveauRaisonnement(Enum):
    """Niveaux hiérarchiques de raisonnement."""
    OPERATIONNEL = 1  # Niveau détaillé, actions concrètes
    TACTIQUE = 2      # Niveau intermédiaire, stratégies locales
    STRATEGIQUE = 3   # Niveau élevé, vision globale
    META = 4          # Méta-raisonnement sur le raisonnement

class StatutProbleme(Enum):
    """Statut d'un problème dans le processus de résolution."""
    NOUVEAU = auto()
    EN_COURS = auto()
    DECOMPOSE = auto()
    RESOLU = auto()
    BLOQUE = auto()
    ABANDONNE = auto()

@dataclass
class Concept:
    """Représente un concept dans le système de raisonnement."""
    nom: str
    proprietes: Dict[str, Any] = field(default_factory=dict)
    relations: Dict[str, List[str]] = field(default_factory=dict)
    confiance: float = 1.0
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def ajouter_relation(self, type_relation: str, cible: str) -> None:
        """Ajoute une relation vers un autre concept."""
        if type_relation not in self.relations:
            self.relations[type_relation] = []
        if cible not in self.relations[type_relation]:
            self.relations[type_relation].append(cible)
    
    def calculer_similarite(self, autre: 'Concept') -> float:
        """Calcule la similarité avec un autre concept."""
        if not autre:
            return 0.0
        
        # Similarité basée sur les propriétés communes
        props_communes = set(self.proprietes.keys()) & set(autre.proprietes.keys())
        if not props_communes:
            return 0.0
        
        score = 0.0
        for prop in props_communes:
            if self.proprietes[prop] == autre.proprietes[prop]:
                score += 1.0
        
        return score / len(props_communes)

@dataclass
class Probleme:
    """Représente un problème à résoudre."""
    id: str
    description: str
    contexte: Dict[str, Any] = field(default_factory=dict)
    contraintes: List[str] = field(default_factory=list)
    objectifs: List[str] = field(default_factory=list)
    niveau: NiveauRaisonnement = NiveauRaisonnement.OPERATIONNEL
    statut: StatutProbleme = StatutProbleme.NOUVEAU
    parent: Optional[str] = None
    sous_problemes: List[str] = field(default_factory=list)
    solutions_candidates: List['Solution'] = field(default_factory=list)
    priorite: int = 5
    deadline: Optional[datetime] = None
    tags: Set[str] = field(default_factory=set)
    
    def ajouter_sous_probleme(self, sous_probleme_id: str) -> None:
        """Ajoute un sous-problème."""
        if sous_probleme_id not in self.sous_problemes:
            self.sous_problemes.append(sous_probleme_id)

@dataclass
class Solution:
    """Représente une solution à un problème."""
    id: str
    probleme_id: str
    description: str
    etapes: List[str] = field(default_factory=list)
    ressources_requises: List[str] = field(default_factory=list)
    confiance: float = 0.5
    cout_estime: float = 0.0
    duree_estimee: float = 0.0
    risques: List[str] = field(default_factory=list)
    avantages: List[str] = field(default_factory=list)
    type_logique: TypeLogique = TypeLogique.DEDUCTIVE
    preuves: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def calculer_score(self) -> float:
        """Calcule un score global pour la solution."""
        score_confiance = self.confiance
        score_cout = max(0, 1 - (self.cout_estime / 100))  # Normalisation approximative
        score_temps = max(0, 1 - (self.duree_estimee / 24))  # Normalisation en heures
        score_risque = max(0, 1 - (len(self.risques) / 10))
        
        return (score_confiance + score_cout + score_temps + score_risque) / 4

class BaseConcepts:
    """Base de connaissances pour les concepts."""
    
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.index_proprietes: Dict[str, Set[str]] = {}
        self.index_relations: Dict[str, Set[str]] = {}
    
    def ajouter_concept(self, concept: Concept) -> None:
        """Ajoute un concept à la base."""
        self.concepts[concept.nom] = concept
        
        # Mise à jour des index
        for prop in concept.proprietes:
            if prop not in self.index_proprietes:
                self.index_proprietes[prop] = set()
            self.index_proprietes[prop].add(concept.nom)
        
        for relation in concept.relations:
            if relation not in self.index_relations:
                self.index_relations[relation] = set()
            self.index_relations[relation].add(concept.nom)
    
    def rechercher_concepts(self, criteres: Dict[str, Any]) -> List[Concept]:
        """Recherche des concepts selon des critères."""
        resultats = []
        
        for concept in self.concepts.values():
            match = True
            for cle, valeur in criteres.items():
                if cle in concept.proprietes:
                    if concept.proprietes[cle] != valeur:
                        match = False
                        break
                else:
                    match = False
                    break
            
            if match:
                resultats.append(concept)
        
        return resultats
    
    def trouver_analogies(self, concept_source: str, seuil_similarite: float = 0.3) -> List[Tuple[str, float]]:
        """Trouve des concepts analogues."""
        if concept_source not in self.concepts:
            return []
        
        source = self.concepts[concept_source]
        analogies = []
        
        for nom, concept in self.concepts.items():
            if nom != concept_source:
                similarite = source.calculer_similarite(concept)
                if similarite >= seuil_similarite:
                    analogies.append((nom, similarite))
        
        return sorted(analogies, key=lambda x: x[1], reverse=True)

class MoteurLogique:
    """Moteur de logique hybride."""
    
    def __init__(self, base_concepts: BaseConcepts):
        self.base_concepts = base_concepts
        self.regles_deduction: Dict[str, List[str]] = {}
        self.patterns_induction: List[Dict[str, Any]] = []
        self.hypotheses_abduction: List[Dict[str, Any]] = []
    
    def raisonnement_deductif(self, premisses: List[str], regles: List[str]) -> List[str]:
        """Applique un raisonnement déductif."""
        conclusions = []
        
        for regle in regles:
            # Format simple : "SI condition ALORS conclusion"
            if " ALORS " in regle:
                condition, conclusion = regle.split(" ALORS ", 1)
                condition = condition.replace("SI ", "").strip()
                
                if condition in premisses:
                    conclusions.append(conclusion.strip())
        
        return conclusions
    
    def raisonnement_inductif(self, observations: List[Dict[str, Any]]) -> List[str]:
        """Génère des généralisations par induction."""
        if len(observations) < 2:
            return []
        
        generalisations = []
        
        # Recherche de patterns communs
        proprietes_communes = set(observations[0].keys())
        for obs in observations[1:]:
            proprietes_communes &= set(obs.keys())
        
        for prop in proprietes_communes:
            valeurs = [obs[prop] for obs in observations]
            if len(set(valeurs)) == 1:  # Même valeur pour tous
                generalisation = f"Tous les éléments observés ont {prop} = {valeurs[0]}"
                generalisations.append(generalisation)
        
        return generalisations
    
    def raisonnement_abductif(self, observation: str, hypotheses_possibles: List[str]) -> List[Tuple[str, float]]:
        """Trouve la meilleure explication pour une observation."""
        explications = []
        
        for hypothese in hypotheses_possibles:
            # Score simple basé sur la plausibilité
            score = random.uniform(0.1, 1.0)  # À remplacer par une vraie évaluation
            explications.append((hypothese, score))
        
        return sorted(explications, key=lambda x: x[1], reverse=True)
    
    def raisonnement_analogique(self, probleme_source: str, probleme_cible: str) -> List[str]:
        """Applique un raisonnement par analogie."""
        suggestions = []
        
        # Recherche d'analogies dans la base de concepts
        analogies = self.base_concepts.trouver_analogies(probleme_source)
        
        for analogie, score in analogies[:3]:  # Top 3
            suggestion = f"Par analogie avec {analogie} (score: {score:.2f}), "
            suggestion += f"considérer des solutions similaires pour {probleme_cible}"
            suggestions.append(suggestion)
        
        return suggestions

class DecomposeurProblemes:
    """Décompose les problèmes complexes en sous-problèmes."""
    
    def __init__(self):
        self.strategies_decomposition = {
            "fonctionnelle": self._decomposition_fonctionnelle,
            "temporelle": self._decomposition_temporelle,
            "hierarchique": self._decomposition_hierarchique,
            "par_contraintes": self._decomposition_par_contraintes
        }
    
    def decomposer(self, probleme: Probleme, strategie: str = "fonctionnelle") -> List[Probleme]:
        """Décompose un problème selon une stratégie."""
        if strategie not in self.strategies_decomposition:
            strategie = "fonctionnelle"
        
        return self.strategies_decomposition[strategie](probleme)
    
    def _decomposition_fonctionnelle(self, probleme: Probleme) -> List[Probleme]:
        """Décomposition basée sur les fonctions."""
        sous_problemes = []
        
        # Analyse des mots-clés pour identifier les fonctions
        mots_cles = ["analyser", "concevoir", "implémenter", "tester", "déployer"]
        
        for i, mot_cle in enumerate(mots_cles):
            if mot_cle in probleme.description.lower():
                sous_pb = Probleme(
                    id=f"{probleme.id}_func_{i}",
                    description=f"{mot_cle.capitalize()} pour {probleme.description}",
                    contexte=probleme.contexte.copy(),
                    parent=probleme.id,
                    niveau=NiveauRaisonnement.OPERATIONNEL,
                    priorite=probleme.priorite
                )
                sous_problemes.append(sous_pb)
        
        return sous_problemes
    
    def _decomposition_temporelle(self, probleme: Probleme) -> List[Probleme]:
        """Décomposition basée sur les phases temporelles."""
        phases = ["court_terme", "moyen_terme", "long_terme"]
        sous_problemes = []
        
        for i, phase in enumerate(phases):
            sous_pb = Probleme(
                id=f"{probleme.id}_temp_{i}",
                description=f"Phase {phase}: {probleme.description}",
                contexte=probleme.contexte.copy(),
                parent=probleme.id,
                niveau=NiveauRaisonnement.TACTIQUE,
                priorite=probleme.priorite - i
            )
            sous_problemes.append(sous_pb)
        
        return sous_problemes
    
    def _decomposition_hierarchique(self, probleme: Probleme) -> List[Probleme]:
        """Décomposition hiérarchique par niveaux."""
        sous_problemes = []
        
        if probleme.niveau.value > 1:
            niveau_inferieur = NiveauRaisonnement(probleme.niveau.value - 1)
            
            # Créer 2-4 sous-problèmes de niveau inférieur
            nb_sous_pb = random.randint(2, 4)
            for i in range(nb_sous_pb):
                sous_pb = Probleme(
                    id=f"{probleme.id}_hier_{i}",
                    description=f"Aspect {i+1} de: {probleme.description}",
                    contexte=probleme.contexte.copy(),
                    parent=probleme.id,
                    niveau=niveau_inferieur,
                    priorite=probleme.priorite
                )
                sous_problemes.append(sous_pb)
        
        return sous_problemes
    
    def _decomposition_par_contraintes(self, probleme: Probleme) -> List[Probleme]:
        """Décomposition basée sur les contraintes."""
        sous_problemes = []
        
        for i, contrainte in enumerate(probleme.contraintes):
            sous_pb = Probleme(
                id=f"{probleme.id}_const_{i}",
                description=f"Gérer contrainte '{contrainte}' pour {probleme.description}",
                contexte=probleme.contexte.copy(),
                contraintes=[contrainte],
                parent=probleme.id,
                niveau=probleme.niveau,
                priorite=probleme.priorite + 1  # Plus prioritaire
            )
            sous_problemes.append(sous_pb)
        
        return sous_problemes

class GenerateurSolutions:
    """Génère des solutions pour les problèmes."""
    
    def __init__(self, moteur_logique: MoteurLogique):
        self.moteur_logique = moteur_logique
        self.templates_solutions = {
            "analyse": ["Collecter données", "Analyser patterns", "Identifier tendances", "Formuler conclusions"],
            "conception": ["Définir spécifications", "Créer architecture", "Détailler composants", "Valider design"],
            "implementation": ["Préparer environnement", "Développer solution", "Tester fonctionnalités", "Optimiser performance"],
            "default": ["Identifier le problème", "Rechercher solutions", "Évaluer options", "Implémenter solution"]
        }
    
    def generer_solutions(self, probleme: Probleme) -> List[Solution]:
        """Génère des solutions pour un problème."""
        solutions = []
        
        # Solution déductive
        sol_deductive = self._generer_solution_deductive(probleme)
        if sol_deductive:
            solutions.append(sol_deductive)
        
        # Solution inductive
        sol_inductive = self._generer_solution_inductive(probleme)
        if sol_inductive:
            solutions.append(sol_inductive)
        
        # Solution analogique
        sol_analogique = self._generer_solution_analogique(probleme)
        if sol_analogique:
            solutions.append(sol_analogique)
        
        # Solution créative (combinaison)
        sol_creative = self._generer_solution_creative(probleme)
        if sol_creative:
            solutions.append(sol_creative)
        
        return solutions
    
    def _generer_solution_deductive(self, probleme: Probleme) -> Optional[Solution]:
        """Génère une solution par déduction logique."""
        template = self._choisir_template(probleme)
        
        solution = Solution(
            id=f"{probleme.id}_sol_deductive",
            probleme_id=probleme.id,
            description=f"Solution déductive pour {probleme.description}",
            etapes=template.copy(),
            type_logique=TypeLogique.DEDUCTIVE,
            confiance=0.8,
            cout_estime=random.uniform(10, 50),
            duree_estimee=random.uniform(1, 8)
        )
        
        return solution
    
    def _generer_solution_inductive(self, probleme: Probleme) -> Optional[Solution]:
        """Génère une solution par induction."""
        template = self._choisir_template(probleme)
        
        # Modifier le template pour l'approche inductive
        etapes_inductives = ["Observer cas similaires"] + template + ["Généraliser pattern"]
        
        solution = Solution(
            id=f"{probleme.id}_sol_inductive",
            probleme_id=probleme.id,
            description=f"Solution inductive pour {probleme.description}",
            etapes=etapes_inductives,
            type_logique=TypeLogique.INDUCTIVE,
            confiance=0.6,
            cout_estime=random.uniform(15, 60),
            duree_estimee=random.uniform(2, 10)
        )
        
        return solution
    
    def _generer_solution_analogique(self, probleme: Probleme) -> Optional[Solution]:
        """Génère une solution par analogie."""
        analogies = self.moteur_logique.raisonnement_analogique(probleme.description, probleme.id)
        
        if not analogies:
            return None
        
        solution = Solution(
            id=f"{probleme.id}_sol_analogique",
            probleme_id=probleme.id,
            description=f"Solution analogique pour {probleme.description}",
            etapes=["Identifier analogies"] + analogies[:3] + ["Adapter solution"],
            type_logique=TypeLogique.ANALOGIQUE,
            confiance=0.5,
            cout_estime=random.uniform(5, 30),
            duree_estimee=random.uniform(0.5, 5)
        )
        
        return solution
    
    def _generer_solution_creative(self, probleme: Probleme) -> Optional[Solution]:
        """Génère une solution créative en combinant approches."""
        template = self._choisir_template(probleme)
        
        # Ajout d'étapes créatives
        etapes_creatives = [
            "Brainstorming créatif",
            "Analyse multi-perspective"
        ] + template + [
            "Synthèse innovante",
            "Validation créative"
        ]
        
        solution = Solution(
            id=f"{probleme.id}_sol_creative",
            probleme_id=probleme.id,
            description=f"Solution créative pour {probleme.description}",
            etapes=etapes_creatives,
            type_logique=TypeLogique.ABDUCTIVE,
            confiance=0.4,
            cout_estime=random.uniform(20, 80),
            duree_estimee=random.uniform(3, 12)
        )
        
        return solution
    
    def _choisir_template(self, probleme: Probleme) -> List[str]:
        """Choisit un template d'étapes basé sur le problème."""
        description_lower = probleme.description.lower()
        
        for mot_cle, template in self.templates_solutions.items():
            if mot_cle in description_lower:
                return template.copy()
        
        return self.templates_solutions["default"].copy()

class EvaluateurSolutions:
    """Évalue et classe les solutions."""
    
    def __init__(self):
        self.criteres_evaluation = {
            "faisabilite": 0.3,
            "efficacite": 0.25,
            "cout": 0.2,
            "temps": 0.15,
            "risque": 0.1
        }
    
    def evaluer_solution(self, solution: Solution, contexte: Dict[str, Any] = None) -> Dict[str, float]:
        """Évalue une solution selon plusieurs critères."""
        if contexte is None:
            contexte = {}
        
        scores = {}
        
        # Faisabilité basée sur la confiance et les ressources
        scores["faisabilite"] = min(solution.confiance, 0.9)
        
        # Efficacité basée sur le nombre d'étapes et la complexité
        nb_etapes = len(solution.etapes)
        scores["efficacite"] = max(0.1, 1.0 - (nb_etapes / 20))
        
        # Coût (inversé pour que moins cher = meilleur score)
        scores["cout"] = max(0.1, 1.0 - min(solution.cout_estime / 100, 0.9))
        
        # Temps (inversé)
        scores["temps"] = max(0.1, 1.0 - min(solution.duree_estimee / 24, 0.9))
        
        # Risque (inversé)
        scores["risque"] = max(0.1, 1.0 - min(len(solution.risques) / 10, 0.9))
        
        return scores
    
    def calculer_score_global(self, solution: Solution, contexte: Dict[str, Any] = None) -> float:
        """Calcule le score global pondéré."""
        scores = self.evaluer_solution(solution, contexte)
        
        score_global = 0.0
        for critere, poids in self.criteres_evaluation.items():
            score_global += scores.get(critere, 0.5) * poids
        
        return score_global
    
    def classer_solutions(self, solutions: List[Solution], contexte: Dict[str, Any] = None) -> List[Tuple[Solution, float]]:
        """Classe les solutions par score décroissant."""
        solutions_scorees = []
        
        for solution in solutions:
            score = self.calculer_score_global(solution, contexte)
            solutions_scorees.append((solution, score))
        
        return sorted(solutions_scorees, key=lambda x: x[1], reverse=True)

class GestionnaireIncertitude:
    """Gère l'incertitude et les conflits dans le raisonnement."""
    
    def __init__(self):
        self.seuil_confiance = 0.7
        self.strategies_resolution = ["vote_majoritaire", "moyenne_ponderee", "expert_system"]
    
    def detecter_conflits(self, solutions: List[Solution]) -> List[Dict[str, Any]]:
        """Détecte les conflits entre solutions."""
        conflits = []
        
        for i, sol1 in enumerate(solutions):
            for j, sol2 in enumerate(solutions[i+1:], i+1):
                if self._solutions_en_conflit(sol1, sol2):
                    conflit = {
                        "type": "contradiction",
                        "solutions": [sol1.id, sol2.id],
                        "description": f"Conflit entre {sol1.description} et {sol2.description}",
                        "severite": self._calculer_severite_conflit(sol1, sol2)
                    }
                    conflits.append(conflit)
        
        return conflits
    
    def resoudre_conflit(self, conflit: Dict[str, Any], solutions: List[Solution], strategie: str = "moyenne_ponderee") -> Solution:
        """Résout un conflit entre solutions."""
        solutions_en_conflit = [s for s in solutions if s.id in conflit["solutions"]]
        
        if strategie == "vote_majoritaire":
            return self._resolution_par_vote(solutions_en_conflit)
        elif strategie == "moyenne_ponderee":
            return self._resolution_par_moyenne(solutions_en_conflit)
        else:
            return self._resolution_expert_system(solutions_en_conflit)
    
    def propager_incertitude(self, solution: Solution, facteur_propagation: float = 0.9) -> Solution:
        """Propage l'incertitude dans une solution."""
        solution_modifiee = copy.deepcopy(solution)
        solution_modifiee.confiance *= facteur_propagation
        
        # Ajout de mise en garde
        if solution_modifiee.confiance < self.seuil_confiance:
            solution_modifiee.risques.append("Niveau de confiance faible")
        
        return solution_modifiee
    
    def _solutions_en_conflit(self, sol1: Solution, sol2: Solution) -> bool:
        """Vérifie si deux solutions sont en conflit."""
        # Conflit simple : si les étapes sont très différentes
        etapes1 = set(sol1.etapes)
        etapes2 = set(sol2.etapes)
        intersection = etapes1 & etapes2
        
        # Conflit si moins de 20% d'étapes communes
        return len(intersection) / max(len(etapes1), len(etapes2)) < 0.2
    
    def _calculer_severite_conflit(self, sol1: Solution, sol2: Solution) -> float:
        """Calcule la sévérité d'un conflit."""
        diff_confiance = abs(sol1.confiance - sol2.confiance)
        diff_cout = abs(sol1.cout_estime - sol2.cout_estime) / max(sol1.cout_estime, sol2.cout_estime, 1)
        
        return (diff_confiance + diff_cout) / 2
    
    def _resolution_par_vote(self, solutions: List[Solution]) -> Solution:
        """Résolution par vote majoritaire."""
        return max(solutions, key=lambda s: s.confiance)
    
    def _resolution_par_moyenne(self, solutions: List[Solution]) -> Solution:
        """Résolution par moyenne pondérée."""
        if not solutions:
            return None
        
        # Créer une solution hybride
        solution_hybride = copy.deepcopy(solutions[0])
        solution_hybride.id += "_hybride"
        solution_hybride.description = "Solution hybride résolvant conflit"
        
        # Moyenne des paramètres numériques
        solution_hybride.confiance = sum(s.confiance for s in solutions) / len(solutions)
        solution_hybride.cout_estime = sum(s.cout_estime for s in solutions) / len(solutions)
        solution_hybride.duree_estimee = sum(s.duree_estimee for s in solutions) / len(solutions)
        
        # Combinaison des étapes
        toutes_etapes = []
        for sol in solutions:
            toutes_etapes.extend(sol.etapes)
        solution_hybride.etapes = list(dict.fromkeys(toutes_etapes))  # Supprime doublons
        
        return solution_hybride
    
    def _resolution_expert_system(self, solutions: List[Solution]) -> Solution:
        """Résolution par système expert."""
        # Privilégier la solution avec le meilleur équilibre confiance/coût
        meilleur_score = -1
        meilleure_solution = solutions[0]
        
        for solution in solutions:
            score = solution.confiance * (1 / max(solution.cout_estime, 1))
            if score > meilleur_score:
                meilleur_score = score
                meilleure_solution = solution
        
        return meilleure_solution

class RaisonnementAmeliore:
    """Classe principale du système de raisonnement amélioré."""
    
    def __init__(self):
        self.base_concepts = BaseConcepts()
        self.moteur_logique = MoteurLogique(self.base_concepts)
        self.decomposeur = DecomposeurProblemes()
        self.generateur_solutions = GenerateurSolutions(self.moteur_logique)
        self.evaluateur = EvaluateurSolutions()
        self.gestionnaire_incertitude = GestionnaireIncertitude()
        
        self.problemes: Dict[str, Probleme] = {}
        self.solutions: Dict[str, List[Solution]] = {}
        self.historique_raisonnement: List[Dict[str, Any]] = []
        
        # Initialisation avec quelques concepts de base
        self._initialiser_concepts_base()
    
    def _initialiser_concepts_base(self):
        """Initialise la base avec quelques concepts fondamentaux."""
        concepts_base = [
            Concept("probleme", {"type": "abstraction", "complexite": "variable"}),
            Concept("solution", {"type": "action", "efficacite": "mesurable"}),
            Concept("analyse", {"type": "processus", "precision": "importante"}),
            Concept("synthese", {"type": "processus", "creativite": "elevee"}),
            Concept("evaluation", {"type": "jugement", "objectivite": "requise"})
        ]
        
        for concept in concepts_base:
            self.base_concepts.ajouter_concept(concept)
    
    async def resoudre_probleme(self, probleme: Probleme, profondeur_max: int = 3) -> Dict[str, Any]:
        """Résout un problème de manière complète et asynchrone."""
        logger.info(f"Début résolution du problème: {probleme.id}")
        
        # Enregistrer le problème
        self.problemes[probleme.id] = probleme
        
        try:
            # Étape 1: Analyse du problème
            resultat_analyse = await self._analyser_probleme(probleme)
            
            # Étape 2: Décomposition si nécessaire
            sous_problemes = []
            if probleme.niveau.value > 1 and profondeur_max > 0:
                sous_problemes = await self._decomposer_probleme(probleme, profondeur_max)
            
            # Étape 3: Génération de solutions
            solutions = await self._generer_solutions_async(probleme)
            
            # Étape 4: Résolution des sous-problèmes
            solutions_sous_problemes = []
            if sous_problemes:
                solutions_sous_problemes = await self._resoudre_sous_problemes(sous_problemes, profondeur_max - 1)
            
            # Étape 5: Évaluation et classement
            toutes_solutions = solutions + solutions_sous_problemes
            solutions_classees = self.evaluateur.classer_solutions(toutes_solutions)
            
            # Étape 6: Gestion des conflits et incertitudes
            conflits = self.gestionnaire_incertitude.detecter_conflits(toutes_solutions)
            solutions_resolues = await self._resoudre_conflits(conflits, toutes_solutions)
            
            # Étape 7: Sélection de la meilleure solution
            meilleure_solution = solutions_resolues[0] if solutions_resolues else None
            
            # Enregistrement du résultat
            resultat = {
                "probleme_id": probleme.id,
                "statut": "resolu" if meilleure_solution else "echec",
                "solution_principale": meilleure_solution,
                "solutions_alternatives": solutions_resolues[1:5],  # Top 5
                "sous_problemes": [sp.id for sp in sous_problemes],
                "conflits_detectes": len(conflits),
                "confiance_globale": meilleure_solution.confiance if meilleure_solution else 0,
                "temps_resolution": datetime.now(),
                "analyse": resultat_analyse
            }
            
            self.solutions[probleme.id] = solutions_resolues
            self.historique_raisonnement.append(resultat)
            
            logger.info(f"Problème {probleme.id} résolu avec confiance {resultat['confiance_globale']:.2f}")
            return resultat
            
        except Exception as e:
            logger.error(f"Erreur lors de la résolution du problème {probleme.id}: {str(e)}")
            return {
                "probleme_id": probleme.id,
                "statut": "erreur",
                "message": str(e),
                "temps_resolution": datetime.now()
            }
    
    async def _analyser_probleme(self, probleme: Probleme) -> Dict[str, Any]:
        """Analyse approfondie d'un problème."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._analyser_probleme_sync, probleme
        )
    
    def _analyser_probleme_sync(self, probleme: Probleme) -> Dict[str, Any]:
        """Version synchrone de l'analyse de problème."""
        analyse = {
            "complexite": self._evaluer_complexite(probleme),
            "domaine": self._identifier_domaine(probleme),
            "type_raisonnement_suggere": self._suggerer_type_raisonnement(probleme),
            "concepts_lies": self._identifier_concepts_lies(probleme),
            "contraintes_critiques": self._analyser_contraintes(probleme)
        }
        
        return analyse
    
    async def _decomposer_probleme(self, probleme: Probleme, profondeur_max: int) -> List[Probleme]:
        """Décompose un problème de manière asynchrone."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.decomposeur.decomposer, probleme
        )
    
    async def _generer_solutions_async(self, probleme: Probleme) -> List[Solution]:
        """Génère des solutions de manière asynchrone."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.generateur_solutions.generer_solutions, probleme
        )
    
    async def _resoudre_sous_problemes(self, sous_problemes: List[Probleme], profondeur_max: int) -> List[Solution]:
        """Résout les sous-problèmes de manière concurrente."""
        if profondeur_max <= 0:
            return []
        
        # Résolution concurrente des sous-problèmes
        taches = [self.resoudre_probleme(sp, profondeur_max) for sp in sous_problemes]
        resultats = await asyncio.gather(*taches, return_exceptions=True)
        
        solutions = []
        for resultat in resultats:
            if isinstance(resultat, dict) and resultat.get("solution_principale"):
                solutions.append(resultat["solution_principale"])
        
        return solutions
    
    async def _resoudre_conflits(self, conflits: List[Dict[str, Any]], solutions: List[Solution]) -> List[Solution]:
        """Résout les conflits détectés."""
        if not conflits:
            return self.evaluateur.classer_solutions(solutions)
        
        solutions_resolues = solutions.copy()
        
        for conflit in conflits:
            solution_resolue = self.gestionnaire_incertitude.resoudre_conflit(conflit, solutions_resolues)
            if solution_resolue:
                # Remplacer les solutions en conflit par la solution résolue
                solutions_resolues = [s for s in solutions_resolues if s.id not in conflit["solutions"]]
                solutions_resolues.append(solution_resolue)
        
        return [sol for sol, score in self.evaluateur.classer_solutions(solutions_resolues)]
    
    def _evaluer_complexite(self, probleme: Probleme) -> str:
        """Évalue la complexité d'un problème."""
        score_complexite = 0
        
        # Facteurs de complexité
        score_complexite += len(probleme.contraintes) * 2
        score_complexite += len(probleme.objectifs) * 1.5
        score_complexite += len(probleme.description.split()) * 0.1
        score_complexite += probleme.niveau.value * 3
        
        if score_complexite < 5:
            return "faible"
        elif score_complexite < 15:
            return "moyenne"
        else:
            return "elevee"
    
    def _identifier_domaine(self, probleme: Probleme) -> str:
        """Identifie le domaine d'un problème."""
        domaines = {
            "technique": ["code", "système", "algorithme", "données"],
            "business": ["stratégie", "marché", "client", "vente"],
            "recherche": ["analyse", "étude", "investigation", "découverte"],
            "créatif": ["design", "innovation", "création", "art"]
        }
        
        description_lower = probleme.description.lower()
        
        for domaine, mots_cles in domaines.items():
            if any(mot in description_lower for mot in mots_cles):
                return domaine
        
        return "general"
    
    def _suggerer_type_raisonnement(self, probleme: Probleme) -> TypeLogique:
        """Suggère le type de raisonnement le plus adapté."""
        description_lower = probleme.description.lower()
        
        if any(mot in description_lower for mot in ["démontrer", "prouver", "déduire"]):
            return TypeLogique.DEDUCTIVE
        elif any(mot in description_lower for mot in ["pattern", "tendance", "généraliser"]):
            return TypeLogique.INDUCTIVE
        elif any(mot in description_lower for mot in ["expliquer", "cause", "pourquoi"]):
            return TypeLogique.ABDUCTIVE
        elif any(mot in description_lower for mot in ["similaire", "comme", "analogue"]):
            return TypeLogique.ANALOGIQUE
        else:
            return TypeLogique.DEDUCTIVE  # Par défaut
    
    def _identifier_concepts_lies(self, probleme: Probleme) -> List[str]:
        """Identifie les concepts liés à un problème."""
        mots_probleme = set(probleme.description.lower().split())
        concepts_lies = []
        
        for nom_concept, concept in self.base_concepts.concepts.items():
            if nom_concept.lower() in mots_probleme:
                concepts_lies.append(nom_concept)
        
        return concepts_lies
    
    def _analyser_contraintes(self, probleme: Probleme) -> List[str]:
        """Analyse les contraintes critiques."""
        contraintes_critiques = []
        
        for contrainte in probleme.contraintes:
            if any(mot in contrainte.lower() for mot in ["temps", "budget", "urgent", "critique"]):
                contraintes_critiques.append(contrainte)
        
        return contraintes_critiques
    
    def obtenir_statistiques(self) -> Dict[str, Any]:
        """Retourne des statistiques sur l'utilisation du système."""
        return {
            "problemes_traites": len(self.problemes),
            "solutions_generees": sum(len(sols) for sols in self.solutions.values()),
            "concepts_bases": len(self.base_concepts.concepts),
            "taux_resolution": len([p for p in self.problemes.values() if p.statut == StatutProbleme.RESOLU]) / max(len(self.problemes), 1),
            "type_logique_populaire": self._type_logique_plus_utilise(),
            "niveau_raisonnement_moyen": self._niveau_raisonnement_moyen()
        }
    
    def _type_logique_plus_utilise(self) -> str:
        """Trouve le type de logique le plus utilisé."""
        compteurs = {}
        
        for solutions_list in self.solutions.values():
            for solution in solutions_list:
                type_logique = solution.type_logique.value
                compteurs[type_logique] = compteurs.get(type_logique, 0) + 1
        
        if not compteurs:
            return "aucun"
        
        return max(compteurs.items(), key=lambda x: x[1])[0]
    
    def _niveau_raisonnement_moyen(self) -> float:
        """Calcule le niveau de raisonnement moyen."""
        if not self.problemes:
            return 0
        
        total = sum(p.niveau.value for p in self.problemes.values())
        return total / len(self.problemes)
    
    def sauvegarder_etat(self, fichier: str) -> bool:
        """Sauvegarde l'état du système."""
        try:
            etat = {
                "concepts": {nom: {
                    "nom": c.nom,
                    "proprietes": c.proprietes,
                    "relations": c.relations,
                    "confiance": c.confiance,
                    "source": c.source
                } for nom, c in self.base_concepts.concepts.items()},
                "historique": self.historique_raisonnement,
                "statistiques": self.obtenir_statistiques()
            }
            
            with open(fichier, 'w', encoding='utf-8') as f:
                json.dump(etat, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"État sauvegardé dans {fichier}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {str(e)}")
            return False
    
    def charger_etat(self, fichier: str) -> bool:
        """Charge l'état du système."""
        try:
            with open(fichier, 'r', encoding='utf-8') as f:
                etat = json.load(f)
            
            # Restaurer les concepts
            for nom, data in etat.get("concepts", {}).items():
                concept = Concept(
                    nom=data["nom"],
                    proprietes=data["proprietes"],
                    relations=data["relations"],
                    confiance=data["confiance"],
                    source=data["source"]
                )
                self.base_concepts.ajouter_concept(concept)
            
            # Restaurer l'historique
            self.historique_raisonnement = etat.get("historique", [])
            
            logger.info(f"État chargé depuis {fichier}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement: {str(e)}")
            return False

# Fonction utilitaire pour créer facilement un système de raisonnement
def creer_systeme_raisonnement() -> RaisonnementAmeliore:
    """Crée et initialise un nouveau système de raisonnement."""
    return RaisonnementAmeliore()

# Exemple d'utilisation
if __name__ == "__main__":
    async def exemple_utilisation():
        # Créer le système
        systeme = creer_systeme_raisonnement()
        
        # Créer un problème exemple
        probleme = Probleme(
            id="pb_001",
            description="Analyser et optimiser les performances d'un système distribué",
            contexte={"environnement": "production", "criticite": "haute"},
            contraintes=["temps limité", "budget restreint", "disponibilité 24/7"],
            objectifs=["améliorer latence", "réduire coûts", "maintenir fiabilité"],
            niveau=NiveauRaisonnement.STRATEGIQUE,
            priorite=8
        )
        
        # Résoudre le problème
        resultat = await systeme.resoudre_probleme(probleme)
        
        print("=== RÉSULTAT DE RÉSOLUTION ===")
        print(f"Statut: {resultat['statut']}")
        if resultat.get('solution_principale'):
            sol = resultat['solution_principale']
            print(f"Solution: {sol.description}")
            print(f"Confiance: {sol.confiance:.2f}")
            print(f"Étapes: {sol.etapes}")
        
        print(f"\nStatistiques: {systeme.obtenir_statistiques()}")
    
    # Exécuter l'exemple
    asyncio.run(exemple_utilisation())
