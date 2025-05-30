"""
Module de métacognition profonde et auto-réflexion pour Gemini.
Ce module implémente un système avancé de métacognition permettant à l'IA de :
- S'auto-évaluer en temps réel
- Détecter et corriger ses biais cognitifs
- Monitorer ses processus de raisonnement
- Prendre conscience de ses limites
- Optimiser ses stratégies d'apprentissage

Fonctionnalités principales :
- Auto-évaluation continue de la qualité du raisonnement
- Système de détection et correction des biais cognitifs
- Monitoring adaptatif des processus de résolution
- Gestion de l'incertitude et conscience des limites
- Stratégies métacognitives d'optimisation
"""

import logging
import random
import json
import math
import statistics
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
from enum import Enum, auto
from dataclasses import dataclass, field
import copy
from collections import defaultdict, deque
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

# Métadonnées du module
MODULE_METADATA = {
    "enabled": True,
    "priority": 60,
    "description": "Module de métacognition profonde et auto-réflexion avancée",
    "version": "2.0.0",
    "dependencies": ["enhanced_reasoning"],
    "hooks": ["process_request", "process_response", "pre_reasoning", "post_reasoning"]
}

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TypeBiais(Enum):
    """Types de biais cognitifs détectables."""
    CONFIRMATION = "confirmation"
    ANCRAGE = "ancrage"
    DISPONIBILITE = "disponibilite"
    REPRESENTATIVITE = "representativite"
    SURCONFIANCE = "surconfiance"
    SOUSCONFIANCE = "sousconfiance"
    EFFET_HALO = "effet_halo"
    ESCALADE_ENGAGEMENT = "escalade_engagement"
    BIAIS_OPTIMISME = "biais_optimisme"
    BIAIS_NEGATIVITE = "biais_negativite"

class NiveauQualite(Enum):
    """Niveaux de qualité du raisonnement."""
    EXCELLENT = 5
    BON = 4
    MOYEN = 3
    FAIBLE = 2
    TRES_FAIBLE = 1

class StatutProcessus(Enum):
    """Statut des processus de raisonnement."""
    INITIALISATION = auto()
    EN_COURS = auto()
    EVALUATION = auto()
    OPTIMISATION = auto()
    TERMINE = auto()
    ERREUR = auto()

class TypeIncertitude(Enum):
    """Types d'incertitude identifiables."""
    EPISTEMIQUE = "epistemique"  # Manque de connaissances
    ALEATOIRE = "aleatoire"      # Incertitude intrinsèque
    MODELISATION = "modelisation" # Limites du modèle
    DONNEES = "donnees"          # Qualité des données
    TEMPORELLE = "temporelle"    # Évolution dans le temps

@dataclass
class MetriqueQualite:
    """Métriques pour évaluer la qualité du raisonnement."""
    coherence: float = 0.0
    completude: float = 0.0
    pertinence: float = 0.0
    originalite: float = 0.0
    precision: float = 0.0
    confiance: float = 0.0
    temps_reponse: float = 0.0
    complexite: float = 0.0
    
    def calculer_score_global(self) -> float:
        """Calcule le score global de qualité."""
        poids = {
            'coherence': 0.25,
            'completude': 0.2,
            'pertinence': 0.2,
            'precision': 0.15,
            'confiance': 0.1,
            'originalite': 0.05,
            'temps_reponse': 0.05
        }
        
        score = 0.0
        for metric, value in self.__dict__.items():
            if metric in poids:
                score += value * poids[metric]
        
        return min(max(score, 0.0), 1.0)

@dataclass
class BiaisCognitif:
    """Représente un biais cognitif détecté."""
    type_biais: TypeBiais
    intensite: float
    confiance_detection: float
    contexte: str
    exemples: List[str] = field(default_factory=list)
    strategies_correction: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    corrige: bool = False
    
    def generer_alerte(self) -> str:
        """Génère une alerte pour ce biais."""
        return (f"⚠️ Biais {self.type_biais.value} détecté "
                f"(intensité: {self.intensite:.2f}, confiance: {self.confiance_detection:.2f})")

@dataclass
class ProcessusRaisonnement:
    """Représente un processus de raisonnement en cours."""
    id: str
    type_processus: str
    statut: StatutProcessus
    timestamp_debut: datetime
    timestamp_fin: Optional[datetime] = None
    etapes: List[Dict[str, Any]] = field(default_factory=list)
    metriques: MetriqueQualite = field(default_factory=MetriqueQualite)
    biais_detectes: List[BiaisCognitif] = field(default_factory=list)
    adaptations: List[str] = field(default_factory=list)
    ressources_utilisees: Dict[str, float] = field(default_factory=dict)
    
    def ajouter_etape(self, description: str, donnees: Dict[str, Any] = None) -> None:
        """Ajoute une étape au processus."""
        etape = {
            'timestamp': datetime.now(),
            'description': description,
            'donnees': donnees or {}
        }
        self.etapes.append(etape)
    
    def calculer_duree(self) -> float:
        """Calcule la durée du processus en secondes."""
        if self.timestamp_fin:
            return (self.timestamp_fin - self.timestamp_debut).total_seconds()
        return (datetime.now() - self.timestamp_debut).total_seconds()

@dataclass
class ZoneIncertitude:
    """Représente une zone d'incertitude identifiée."""
    domaine: str
    type_incertitude: TypeIncertitude
    niveau: float
    description: str
    impact_potentiel: float
    strategies_mitigation: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

class AutoEvaluateur:
    """Système d'auto-évaluation en temps réel de la qualité du raisonnement."""
    
    def __init__(self):
        self.historique_evaluations: List[MetriqueQualite] = []
        self.seuils_qualite = {
            'coherence_min': 0.6,
            'completude_min': 0.5,
            'pertinence_min': 0.7,
            'precision_min': 0.6
        }
        self.patterns_qualite: Dict[str, List[str]] = {
            'excellente_coherence': [
                "logique claire", "arguments structurés", "conclusions cohérentes",
                "raisonnement suivi", "enchaînement logique"
            ],
            'faible_coherence': [
                "contradiction", "incohérence", "logique floue",
                "arguments disparates", "raisonnement confus"
            ],
            'haute_pertinence': [
                "directement lié", "répond à la question", "pertinent",
                "adapté au contexte", "ciblé"
            ],
            'faible_pertinence': [
                "hors sujet", "tangentiel", "non pertinent",
                "éloigné du sujet", "digression"
            ]
        }
    
    def evaluer_reponse(self, texte: str, contexte: Dict[str, Any] = None) -> MetriqueQualite:
        """Évalue la qualité d'une réponse en temps réel."""
        if contexte is None:
            contexte = {}
        
        metriques = MetriqueQualite()
        
        # Évaluation de la cohérence
        metriques.coherence = self._evaluer_coherence(texte)
        
        # Évaluation de la complétude
        metriques.completude = self._evaluer_completude(texte, contexte)
        
        # Évaluation de la pertinence
        metriques.pertinence = self._evaluer_pertinence(texte, contexte)
        
        # Évaluation de l'originalité
        metriques.originalite = self._evaluer_originalite(texte)
        
        # Évaluation de la précision
        metriques.precision = self._evaluer_precision(texte)
        
        # Évaluation de la confiance
        metriques.confiance = self._evaluer_confiance(texte)
        
        # Métriques temporelles et de complexité
        metriques.temps_reponse = contexte.get('temps_generation', 0)
        metriques.complexite = self._evaluer_complexite(texte)
        
        # Enregistrer dans l'historique
        self.historique_evaluations.append(metriques)
        
        # Limiter la taille de l'historique
        if len(self.historique_evaluations) > 100:
            self.historique_evaluations.pop(0)
        
        return metriques
    
    def _evaluer_coherence(self, texte: str) -> float:
        """Évalue la cohérence du raisonnement."""
        score = 0.5  # Score de base
        
        # Recherche de marqueurs de cohérence
        marqueurs_positifs = self.patterns_qualite['excellente_coherence']
        marqueurs_negatifs = self.patterns_qualite['faible_coherence']
        
        texte_lower = texte.lower()
        
        # Compter les occurrences positives
        for marqueur in marqueurs_positifs:
            if marqueur in texte_lower:
                score += 0.1
        
        # Pénaliser les marqueurs négatifs
        for marqueur in marqueurs_negatifs:
            if marqueur in texte_lower:
                score -= 0.15
        
        # Vérifier la structure logique
        connecteurs_logiques = ["donc", "par conséquent", "ainsi", "en effet", "car", "parce que"]
        nb_connecteurs = sum(1 for conn in connecteurs_logiques if conn in texte_lower)
        
        if nb_connecteurs > 0:
            score += min(nb_connecteurs * 0.05, 0.2)
        
        return min(max(score, 0.0), 1.0)
    
    def _evaluer_completude(self, texte: str, contexte: Dict[str, Any]) -> float:
        """Évalue la complétude de la réponse."""
        score = 0.5
        
        # Longueur relative (ni trop court ni trop long)
        longueur = len(texte.split())
        if 50 <= longueur <= 300:
            score += 0.2
        elif longueur < 20:
            score -= 0.3
        
        # Présence d'éléments structurants
        if any(marker in texte for marker in ["premièrement", "deuxièmement", "enfin", "en conclusion"]):
            score += 0.15
        
        # Réponse aux aspects de la question (si disponible)
        question = contexte.get('question_originale', '')
        if question:
            mots_cles_question = set(question.lower().split())
            mots_cles_reponse = set(texte.lower().split())
            overlap = len(mots_cles_question & mots_cles_reponse) / max(len(mots_cles_question), 1)
            score += overlap * 0.3
        
        return min(max(score, 0.0), 1.0)
    
    def _evaluer_pertinence(self, texte: str, contexte: Dict[str, Any]) -> float:
        """Évalue la pertinence de la réponse."""
        score = 0.5
        
        # Utiliser les patterns de pertinence
        marqueurs_positifs = self.patterns_qualite['haute_pertinence']
        marqueurs_negatifs = self.patterns_qualite['faible_pertinence']
        
        texte_lower = texte.lower()
        
        for marqueur in marqueurs_positifs:
            if marqueur in texte_lower:
                score += 0.1
        
        for marqueur in marqueurs_negatifs:
            if marqueur in texte_lower:
                score -= 0.2
        
        # Pertinence contextuelle
        if 'domaine' in contexte:
            domaine = contexte['domaine'].lower()
            if domaine in texte_lower:
                score += 0.2
        
        return min(max(score, 0.0), 1.0)
    
    def _evaluer_originalite(self, texte: str) -> float:
        """Évalue l'originalité de la réponse."""
        # Comparer avec les réponses récentes
        if len(self.historique_evaluations) < 5:
            return 0.7  # Score par défaut pour les premières évaluations
        
        # Simuler une comparaison de similarité
        score_originalite = 0.5 + random.uniform(-0.2, 0.3)
        
        # Bonus pour les formulations créatives
        marqueurs_creativite = ["innovation", "créatif", "original", "unique", "nouveau"]
        for marqueur in marqueurs_creativite:
            if marqueur in texte.lower():
                score_originalite += 0.1
        
        return min(max(score_originalite, 0.0), 1.0)
    
    def _evaluer_precision(self, texte: str) -> float:
        """Évalue la précision de la réponse."""
        score = 0.5
        
        # Présence de données spécifiques
        if any(char.isdigit() for char in texte):
            score += 0.15
        
        # Utilisation de termes précis
        termes_precis = ["précisément", "exactement", "spécifiquement", "notamment", "en particulier"]
        for terme in termes_precis:
            if terme in texte.lower():
                score += 0.1
        
        # Éviter les termes vagues
        termes_vagues = ["peut-être", "probablement", "en général", "souvent", "parfois"]
        for terme in termes_vagues:
            if terme in texte.lower():
                score -= 0.05
        
        return min(max(score, 0.0), 1.0)
    
    def _evaluer_confiance(self, texte: str) -> float:
        """Évalue le niveau de confiance exprimé."""
        score = 0.5
        
        # Marqueurs de haute confiance
        haute_confiance = ["certain", "sûr", "évident", "clairement", "définitivement"]
        for marqueur in haute_confiance:
            if marqueur in texte.lower():
                score += 0.1
        
        # Marqueurs de faible confiance
        faible_confiance = ["incertain", "doute", "peut-être", "possiblement", "il semble"]
        for marqueur in faible_confiance:
            if marqueur in texte.lower():
                score -= 0.1
        
        return min(max(score, 0.0), 1.0)
    
    def _evaluer_complexite(self, texte: str) -> float:
        """Évalue la complexité du raisonnement."""
        # Basé sur la longueur des phrases, le vocabulaire, etc.
        phrases = texte.split('.')
        longueur_moyenne = statistics.mean([len(phrase.split()) for phrase in phrases if phrase.strip()])
        
        # Normaliser la complexité
        complexite = min(longueur_moyenne / 20, 1.0)
        return complexite
    
    def identifier_problemes_qualite(self, metriques: MetriqueQualite) -> List[str]:
        """Identifie les problèmes de qualité détectés."""
        problemes = []
        
        if metriques.coherence < self.seuils_qualite['coherence_min']:
            problemes.append("Cohérence insuffisante du raisonnement")
        
        if metriques.completude < self.seuils_qualite['completude_min']:
            problemes.append("Réponse incomplète")
        
        if metriques.pertinence < self.seuils_qualite['pertinence_min']:
            problemes.append("Pertinence limitée par rapport au sujet")
        
        if metriques.precision < self.seuils_qualite['precision_min']:
            problemes.append("Manque de précision dans les détails")
        
        return problemes
    
    def obtenir_tendances_qualite(self) -> Dict[str, float]:
        """Analyse les tendances de qualité sur l'historique."""
        if len(self.historique_evaluations) < 5:
            return {}
        
        recent = self.historique_evaluations[-10:]  # 10 dernières évaluations
        
        tendances = {}
        for attribut in ['coherence', 'completude', 'pertinence', 'precision']:
            valeurs = [getattr(m, attribut) for m in recent]
            tendances[f'{attribut}_moyenne'] = statistics.mean(valeurs)
            tendances[f'{attribut}_tendance'] = valeurs[-1] - valeurs[0] if len(valeurs) > 1 else 0
        
        return tendances

class DetecteurBiais:
    """Système de détection et correction des biais cognitifs."""
    
    def __init__(self):
        self.historique_biais: List[BiaisCognitif] = []
        self.patterns_biais = {
            TypeBiais.CONFIRMATION: {
                'indicateurs': [
                    "confirme que", "comme prévu", "évidemment", "il est clair que",
                    "sans surprise", "comme attendu"
                ],
                'corrections': [
                    "Chercher des contre-exemples",
                    "Considérer des perspectives alternatives",
                    "Questionner les assumptions initiales"
                ]
            },
            TypeBiais.ANCRAGE: {
                'indicateurs': [
                    "basé sur", "en partant de", "selon l'information initiale",
                    "comme mentionné au début"
                ],
                'corrections': [
                    "Réévaluer sans référence aux informations initiales",
                    "Considérer des points de départ alternatifs",
                    "Questionner la pertinence de l'ancre"
                ]
            },
            TypeBiais.DISPONIBILITE: {
                'indicateurs': [
                    "récemment", "souvent", "couramment", "typiquement",
                    "habituellement", "généralement"
                ],
                'corrections': [
                    "Rechercher des statistiques objectives",
                    "Considérer des cas moins visibles",
                    "Évaluer la représentativité des exemples"
                ]
            },
            TypeBiais.SURCONFIANCE: {
                'indicateurs': [
                    "certainement", "sans aucun doute", "absolument sûr",
                    "impossible que", "évidemment", "clairement"
                ],
                'corrections': [
                    "Identifier les sources d'incertitude",
                    "Chercher des scénarios d'échec",
                    "Calibrer le niveau de confiance"
                ]
            }
        }
        self.seuil_detection = 0.6
    
    def detecter_biais(self, texte: str, contexte: Dict[str, Any] = None) -> List[BiaisCognitif]:
        """Détecte les biais cognitifs dans un texte."""
        if contexte is None:
            contexte = {}
        
        biais_detectes = []
        texte_lower = texte.lower()
        
        for type_biais, patterns in self.patterns_biais.items():
            score_biais = self._calculer_score_biais(texte_lower, patterns['indicateurs'])
            
            if score_biais >= self.seuil_detection:
                biais = BiaisCognitif(
                    type_biais=type_biais,
                    intensite=score_biais,
                    confiance_detection=min(score_biais * 1.2, 1.0),
                    contexte=str(contexte),
                    strategies_correction=patterns['corrections'].copy()
                )
                biais_detectes.append(biais)
        
        # Enregistrer dans l'historique
        self.historique_biais.extend(biais_detectes)
        
        return biais_detectes
    
    def _calculer_score_biais(self, texte: str, indicateurs: List[str]) -> float:
        """Calcule le score de présence d'un biais."""
        score = 0.0
        total_mots = len(texte.split())
        
        for indicateur in indicateurs:
            occurrences = texte.count(indicateur)
            if occurrences > 0:
                # Score basé sur la fréquence relative
                score += (occurrences / max(total_mots, 1)) * 10
        
        return min(score, 1.0)
    
    def corriger_biais(self, texte: str, biais: BiaisCognitif) -> str:
        """Propose une correction pour un biais détecté."""
        correction = f"\n\n🔍 **Auto-correction cognitive détectée**\n"
        correction += f"Biais identifié: {biais.type_biais.value}\n"
        correction += f"Stratégies de correction appliquées:\n"
        
        for i, strategie in enumerate(biais.strategies_correction, 1):
            correction += f"{i}. {strategie}\n"
        
        correction += "\n**Révision du raisonnement:**\n"
        correction += self._generer_revision(texte, biais)
        
        return texte + correction
    
    def _generer_revision(self, texte: str, biais: BiaisCognitif) -> str:
        """Génère une révision pour corriger un biais."""
        revisions = {
            TypeBiais.CONFIRMATION: "Examinons des perspectives alternatives et des contre-arguments...",
            TypeBiais.ANCRAGE: "Reconsidérons cette question sans référence aux informations initiales...",
            TypeBiais.DISPONIBILITE: "Cherchons des données plus représentatives et objectives...",
            TypeBiais.SURCONFIANCE: "Identifions les sources d'incertitude et les limites de cette analyse..."
        }
        
        return revisions.get(biais.type_biais, "Révision du raisonnement nécessaire.")
    
    def obtenir_statistiques_biais(self) -> Dict[str, Any]:
        """Retourne des statistiques sur les biais détectés."""
        if not self.historique_biais:
            return {"total": 0}
        
        stats = {"total": len(self.historique_biais)}
        
        # Compter par type
        compteur_types = defaultdict(int)
        for biais in self.historique_biais:
            compteur_types[biais.type_biais.value] += 1
        
        stats["par_type"] = dict(compteur_types)
        
        # Biais les plus fréquents
        if compteur_types:
            stats["plus_frequent"] = max(compteur_types.items(), key=lambda x: x[1])
        
        # Tendance récente
        recent = [b for b in self.historique_biais if 
                 (datetime.now() - b.timestamp).days <= 7]
        stats["recent_7_jours"] = len(recent)
        
        return stats

class MoniteurProcessus:
    """Système de monitoring des processus de résolution avec adaptation dynamique."""
    
    def __init__(self):
        self.processus_actifs: Dict[str, ProcessusRaisonnement] = {}
        self.historique_processus: List[ProcessusRaisonnement] = []
        self.strategies_adaptation = {
            'ralentissement': self._adapter_pour_ralentissement,
            'erreur_repetee': self._adapter_pour_erreurs,
            'qualite_faible': self._adapter_pour_qualite,
            'biais_frequent': self._adapter_pour_biais
        }
        self.seuils_adaptation = {
            'duree_max': 300,  # 5 minutes
            'qualite_min': 0.6,
            'erreurs_max': 3
        }
    
    def demarrer_monitoring(self, processus_id: str, type_processus: str) -> ProcessusRaisonnement:
        """Démarre le monitoring d'un processus."""
        processus = ProcessusRaisonnement(
            id=processus_id,
            type_processus=type_processus,
            statut=StatutProcessus.INITIALISATION,
            timestamp_debut=datetime.now()
        )
        
        self.processus_actifs[processus_id] = processus
        logger.info(f"Monitoring démarré pour le processus {processus_id}")
        
        return processus
    
    def mettre_a_jour_processus(self, processus_id: str, etape: str, donnees: Dict[str, Any] = None) -> None:
        """Met à jour un processus en cours."""
        if processus_id in self.processus_actifs:
            processus = self.processus_actifs[processus_id]
            processus.ajouter_etape(etape, donnees)
            processus.statut = StatutProcessus.EN_COURS
            
            # Vérifier si des adaptations sont nécessaires
            self._verifier_besoins_adaptation(processus)
    
    def terminer_processus(self, processus_id: str, metriques: MetriqueQualite = None) -> None:
        """Termine le monitoring d'un processus."""
        if processus_id in self.processus_actifs:
            processus = self.processus_actifs[processus_id]
            processus.timestamp_fin = datetime.now()
            processus.statut = StatutProcessus.TERMINE
            
            if metriques:
                processus.metriques = metriques
            
            # Archiver le processus
            self.historique_processus.append(processus)
            del self.processus_actifs[processus_id]
            
            logger.info(f"Processus {processus_id} terminé en {processus.calculer_duree():.2f}s")
    
    def _verifier_besoins_adaptation(self, processus: ProcessusRaisonnement) -> None:
        """Vérifie si le processus nécessite des adaptations."""
        duree_actuelle = processus.calculer_duree()
        
        # Adaptation pour ralentissement
        if duree_actuelle > self.seuils_adaptation['duree_max']:
            self._appliquer_adaptation(processus, 'ralentissement')
        
        # Adaptation pour qualité faible
        if processus.metriques.calculer_score_global() < self.seuils_adaptation['qualite_min']:
            self._appliquer_adaptation(processus, 'qualite_faible')
        
        # Adaptation pour biais fréquents
        if len(processus.biais_detectes) > 2:
            self._appliquer_adaptation(processus, 'biais_frequent')
    
    def _appliquer_adaptation(self, processus: ProcessusRaisonnement, type_adaptation: str) -> None:
        """Applique une stratégie d'adaptation."""
        if type_adaptation in self.strategies_adaptation:
            adaptation = self.strategies_adaptation[type_adaptation](processus)
            processus.adaptations.append(adaptation)
            logger.info(f"Adaptation appliquée au processus {processus.id}: {adaptation}")
    
    def _adapter_pour_ralentissement(self, processus: ProcessusRaisonnement) -> str:
        """Stratégie d'adaptation pour les ralentissements."""
        strategies = [
            "Simplification du raisonnement en cours",
            "Priorisation des éléments essentiels",
            "Division en sous-tâches plus petites",
            "Réduction de la profondeur d'analyse"
        ]
        strategie = random.choice(strategies)
        processus.statut = StatutProcessus.OPTIMISATION
        return f"Ralentissement détecté - {strategie}"
    
    def _adapter_pour_erreurs(self, processus: ProcessusRaisonnement) -> str:
        """Stratégie d'adaptation pour les erreurs répétées."""
        return "Erreurs répétées détectées - Changement d'approche méthodologique"
    
    def _adapter_pour_qualite(self, processus: ProcessusRaisonnement) -> str:
        """Stratégie d'adaptation pour la qualité faible."""
        return "Qualité insuffisante - Renforcement des vérifications et révisions"
    
    def _adapter_pour_biais(self, processus: ProcessusRaisonnement) -> str:
        """Stratégie d'adaptation pour les biais fréquents."""
        return "Biais fréquents détectés - Activation des contre-mesures cognitives"
    
    def obtenir_rapport_performance(self) -> Dict[str, Any]:
        """Génère un rapport de performance des processus."""
        if not self.historique_processus:
            return {"message": "Aucun processus terminé"}
        
        durees = [p.calculer_duree() for p in self.historique_processus]
        qualites = [p.metriques.calculer_score_global() for p in self.historique_processus]
        
        rapport = {
            "processus_total": len(self.historique_processus),
            "duree_moyenne": statistics.mean(durees),
            "duree_mediane": statistics.median(durees),
            "qualite_moyenne": statistics.mean(qualites),
            "processus_adaptes": len([p for p in self.historique_processus if p.adaptations]),
            "types_processus": list(set(p.type_processus for p in self.historique_processus))
        }
        
        return rapport

class ConscienceLimites:
    """Système de conscience des propres limites et zones d'incertitude."""
    
    def __init__(self):
        self.zones_incertitude: List[ZoneIncertitude] = []
        self.limites_connues = {
            "temporel": "Connaissance limitée aux données d'entraînement",
            "factuel": "Possible obsolescence des informations",
            "culturel": "Biais culturels potentiels dans la formation",
            "technique": "Limites des modèles de langage actuels",
            "creatif": "Contraintes dans la génération vraiment originale"
        }
        self.domaines_expertise = {
            "fort": ["analyse textuelle", "logique", "mathématiques de base"],
            "moyen": ["sciences générales", "histoire", "programmation"],
            "faible": ["prédictions futures", "conseils médicaux", "conseils juridiques"]
        }
    
    def identifier_incertitudes(self, contexte: str, domaine: str = "") -> List[ZoneIncertitude]:
        """Identifie les zones d'incertitude pour un contexte donné."""
        incertitudes = []
        
        # Analyse du domaine
        niveau_expertise = self._evaluer_niveau_expertise(domaine)
        
        if niveau_expertise == "faible":
            incertitude = ZoneIncertitude(
                domaine=domaine,
                type_incertitude=TypeIncertitude.EPISTEMIQUE,
                niveau=0.8,
                description=f"Expertise limitée dans le domaine {domaine}",
                impact_potentiel=0.7,
                strategies_mitigation=[
                    "Recommander de consulter un expert",
                    "Préciser les limites de l'analyse",
                    "Fournir des sources pour vérification"
                ]
            )
            incertitudes.append(incertitude)
        
        # Détection d'incertitudes temporelles
        if any(mot in contexte.lower() for mot in ["futur", "prédiction", "prévision", "demain"]):
            incertitude = ZoneIncertitude(
                domaine="prédiction",
                type_incertitude=TypeIncertitude.TEMPORELLE,
                niveau=0.9,
                description="Prédictions futures intrinsèquement incertaines",
                impact_potentiel=0.8,
                strategies_mitigation=[
                    "Présenter plusieurs scénarios",
                    "Indiquer les facteurs d'incertitude",
                    "Éviter les prédictions absolues"
                ]
            )
            incertitudes.append(incertitude)
        
        # Détection d'incertitudes sur les données
        if "récent" in contexte.lower() or "actualité" in contexte.lower():
            incertitude = ZoneIncertitude(
                domaine="actualité",
                type_incertitude=TypeIncertitude.DONNEES,
                niveau=0.7,
                description="Informations récentes possiblement non incluses",
                impact_potentiel=0.6,
                strategies_mitigation=[
                    "Préciser la date limite des connaissances",
                    "Recommander de vérifier les sources récentes",
                    "Indiquer le caractère potentiellement daté"
                ]
            )
            incertitudes.append(incertitude)
        
        self.zones_incertitude.extend(incertitudes)
        return incertitudes
    
    def _evaluer_niveau_expertise(self, domaine: str) -> str:
        """Évalue le niveau d'expertise dans un domaine."""
        domaine_lower = domaine.lower()
        
        for niveau, domaines in self.domaines_expertise.items():
            if any(d in domaine_lower for d in domaines):
                return niveau
        
        return "moyen"  # Par défaut
    
    def generer_avertissement_limites(self, incertitudes: List[ZoneIncertitude]) -> str:
        """Génère un avertissement sur les limites identifiées."""
        if not incertitudes:
            return ""
        
        avertissement = "\n\n⚠️ **Conscience des limites**\n"
        
        for incertitude in incertitudes:
            avertissement += f"• **{incertitude.domaine}**: {incertitude.description}\n"
            
            if incertitude.niveau > 0.7:
                avertissement += f"  - Niveau d'incertitude élevé ({incertitude.niveau:.1%})\n"
            
            if incertitude.strategies_mitigation:
                avertissement += f"  - Recommandation: {incertitudes[0].strategies_mitigation[0]}\n"
        
        return avertissement
    
    def evaluer_confiance_globale(self, contexte: str, domaine: str = "") -> float:
        """Évalue la confiance globale pour une réponse."""
        # Confiance de base selon l'expertise
        niveau_expertise = self._evaluer_niveau_expertise(domaine)
        confiance_base = {"fort": 0.9, "moyen": 0.7, "faible": 0.4}[niveau_expertise]
        
        # Ajustements selon les incertitudes
        incertitudes = self.identifier_incertitudes(contexte, domaine)
        
        for incertitude in incertitudes:
            confiance_base *= (1 - incertitude.niveau * 0.3)
        
        return max(confiance_base, 0.1)
    
    def obtenir_cartographie_limites(self) -> Dict[str, Any]:
        """Retourne une cartographie complète des limites."""
        return {
            "limites_connues": self.limites_connues,
            "domaines_expertise": self.domaines_expertise,
            "zones_incertitude_actives": len(self.zones_incertitude),
            "types_incertitude_detectes": list(set(z.type_incertitude.value for z in self.zones_incertitude))
        }

class StrategieMetacognitive:
    """Système de stratégies métacognitives pour optimiser l'apprentissage."""
    
    def __init__(self):
        self.strategies_disponibles = {
            "planification": {
                "description": "Planification du processus de raisonnement",
                "techniques": [
                    "Décomposition du problème",
                    "Définition d'objectifs intermédiaires",
                    "Allocation des ressources cognitives"
                ]
            },
            "monitoring": {
                "description": "Surveillance continue du processus",
                "techniques": [
                    "Vérifications régulières de progression",
                    "Détection d'erreurs en cours",
                    "Ajustement de la stratégie si nécessaire"
                ]
            },
            "evaluation": {
                "description": "Évaluation des résultats et du processus",
                "techniques": [
                    "Analyse de la qualité du résultat",
                    "Identification des points d'amélioration",
                    "Calibration de la confiance"
                ]
            },
            "regulation": {
                "description": "Régulation et optimisation",
                "techniques": [
                    "Correction des erreurs détectées",
                    "Optimisation des stratégies",
                    "Adaptation aux contraintes"
                ]
            }
        }
        self.historique_optimisations: List[Dict[str, Any]] = []
        
    def optimiser_processus_raisonnement(self, processus: ProcessusRaisonnement) -> Dict[str, Any]:
        """Optimise un processus de raisonnement."""
        optimisations = {}
        
        # Étape 1: Planification
        plan = self._planifier_ameliorations(processus)
        optimisations["planification"] = plan
        
        # Étape 2: Identification des goulots d'étranglement
        goulots = self._identifier_goulots_etranglement(processus)
        optimisations["goulots_detectes"] = goulots
        
        # Étape 3: Stratégies d'amélioration
        strategies = self._generer_strategies_amelioration(processus, goulots)
        optimisations["strategies"] = strategies
        
        # Étape 4: Mesures préventives
        preventions = self._definir_mesures_preventives(processus)
        optimisations["preventions"] = preventions
        
        # Enregistrer l'optimisation
        self.historique_optimisations.append({
            "timestamp": datetime.now(),
            "processus_id": processus.id,
            "optimisations": optimisations
        })
        
        return optimisations
    
    def _planifier_ameliorations(self, processus: ProcessusRaisonnement) -> List[str]:
        """Planifie les améliorations possibles."""
        ameliorations = []
        
        # Analyse de la durée
        duree = processus.calculer_duree()
        if duree > 60:  # Plus d'une minute
            ameliorations.append("Optimiser la vitesse de traitement")
        
        # Analyse de la qualité
        score_qualite = processus.metriques.calculer_score_global()
        if score_qualite < 0.7:
            ameliorations.append("Améliorer la qualité du raisonnement")
        
        # Analyse des biais
        if len(processus.biais_detectes) > 1:
            ameliorations.append("Renforcer la détection de biais")
        
        return ameliorations
    
    def _identifier_goulots_etranglement(self, processus: ProcessusRaisonnement) -> List[str]:
        """Identifie les goulots d'étranglement."""
        goulots = []
        
        # Analyser les étapes les plus longues (simulation)
        if len(processus.etapes) > 5:
            goulots.append("Trop d'étapes intermédiaires")
        
        # Analyser les biais répétitifs
        types_biais = [b.type_biais for b in processus.biais_detectes]
        if len(set(types_biais)) != len(types_biais):
            goulots.append("Biais cognitifs répétitifs")
        
        # Analyser les métriques de qualité
        if processus.metriques.coherence < 0.6:
            goulots.append("Problèmes de cohérence")
        
        return goulots
    
    def _generer_strategies_amelioration(self, processus: ProcessusRaisonnement, goulots: List[str]) -> List[str]:
        """Génère des stratégies d'amélioration spécifiques."""
        strategies = []
        
        for goulot in goulots:
            if "étapes" in goulot:
                strategies.append("Regrouper les étapes similaires")
                strategies.append("Paralléliser les traitements indépendants")
            
            elif "biais" in goulot:
                strategies.append("Implémenter des contre-vérifications systématiques")
                strategies.append("Diversifier les perspectives d'analyse")
            
            elif "cohérence" in goulot:
                strategies.append("Renforcer la validation logique")
                strategies.append("Améliorer l'enchaînement des idées")
        
        return strategies
    
    def _definir_mesures_preventives(self, processus: ProcessusRaisonnement) -> List[str]:
        """Définit des mesures préventives."""
        mesures = [
            "Checkpoint de qualité tous les 2 minutes",
            "Vérification automatique des biais toutes les 5 étapes",
            "Calibration de confiance avant conclusion",
            "Révision par perspective alternative"
        ]
        
        return mesures
    
    def apprentissage_adaptatif(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Implémente un apprentissage adaptatif basé sur le feedback."""
        adaptations = {
            "strategies_ajustees": [],
            "seuils_modifies": {},
            "nouvelles_regles": []
        }
        
        # Ajuster les stratégies selon le feedback
        if feedback.get("qualite_insuffisante"):
            adaptations["strategies_ajustees"].append("Renforcement des vérifications de qualité")
        
        if feedback.get("trop_lent"):
            adaptations["strategies_ajustees"].append("Optimisation du processus de génération")
        
        if feedback.get("biais_non_detecte"):
            adaptations["strategies_ajustees"].append("Amélioration de la détection de biais")
        
        # Modifier les seuils si nécessaire
        if feedback.get("false_positive_biais"):
            adaptations["seuils_modifies"]["seuil_biais"] = "Augmentation pour réduire les faux positifs"
        
        return adaptations
    
    def obtenir_rapport_apprentissage(self) -> Dict[str, Any]:
        """Génère un rapport sur l'apprentissage et l'optimisation."""
        return {
            "optimisations_effectuees": len(self.historique_optimisations),
            "strategies_utilisees": len(self.strategies_disponibles),
            "tendances_amelioration": "Analyse des tendances d'amélioration",
            "efficacite_strategies": "Évaluation de l'efficacité des stratégies"
        }

class MetacognitionProfonde:
    """Classe principale orchestrant tous les composants de métacognition."""
    
    def __init__(self):
        self.auto_evaluateur = AutoEvaluateur()
        self.detecteur_biais = DetecteurBiais()
        self.moniteur_processus = MoniteurProcessus()
        self.conscience_limites = ConscienceLimites()
        self.strategie_metacognitive = StrategieMetacognitive()
        
        self.historique_sessions: List[Dict[str, Any]] = []
        self.mode_debug = False
        
        logger.info("Système de métacognition profonde initialisé")
    
    def processer_requete(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Traite une requête avec métacognition complète."""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Démarrer le monitoring
        processus = self.moniteur_processus.demarrer_monitoring(session_id, "traitement_requete")
        
        try:
            # Identifier les incertitudes
            contexte = data.get('text', '')
            domaine = data.get('domain', '')
            incertitudes = self.conscience_limites.identifier_incertitudes(contexte, domaine)
            
            # Évaluer la confiance a priori
            confiance_initiale = self.conscience_limites.evaluer_confiance_globale(contexte, domaine)
            
            data['metacognition'] = {
                'session_id': session_id,
                'incertitudes_detectees': incertitudes,
                'confiance_initiale': confiance_initiale,
                'processus_actif': processus
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Erreur dans le traitement métacognitif: {str(e)}")
            self.moniteur_processus.terminer_processus(session_id)
            return data
    
    def processer_reponse(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Traite une réponse avec analyse métacognitive complète."""
        try:
            if 'metacognition' not in data:
                return data
            
            metacognition = data['metacognition']
            session_id = metacognition['session_id']
            texte = data.get('text', '')
            
            # Auto-évaluation de la qualité
            metriques = self.auto_evaluateur.evaluer_reponse(texte, data)
            
            # Détection de biais
            biais_detectes = self.detecteur_biais.detecter_biais(texte, data)
            
            # Mise à jour du processus
            if session_id in self.moniteur_processus.processus_actifs:
                processus = self.moniteur_processus.processus_actifs[session_id]
                processus.metriques = metriques
                processus.biais_detectes = biais_detectes
                
                # Optimisation si nécessaire
                if metriques.calculer_score_global() < 0.6 or len(biais_detectes) > 1:
                    optimisations = self.strategie_metacognitive.optimiser_processus_raisonnement(processus)
                    data['optimisations_suggerees'] = optimisations
                
                self.moniteur_processus.terminer_processus(session_id, metriques)
            
            # Génération des améliorations de la réponse
            texte_ameliore = self._ameliorer_reponse(texte, metriques, biais_detectes, metacognition.get('incertitudes_detectees', []))
            
            # Mise à jour des données
            data['text'] = texte_ameliore
            data['metriques_qualite'] = metriques
            data['biais_detectes'] = biais_detectes
            
            # Enregistrer la session
            self._enregistrer_session(session_id, data, metriques, biais_detectes)
            
            return data
            
        except Exception as e:
            logger.error(f"Erreur dans l'analyse métacognitive de la réponse: {str(e)}")
            return data
    
    def _ameliorer_reponse(self, texte: str, metriques: MetriqueQualite, 
                          biais: List[BiaisCognitif], incertitudes: List[ZoneIncertitude]) -> str:
        """Améliore une réponse basée sur l'analyse métacognitive."""
        texte_ameliore = texte
        
        # Ajouter des corrections de biais
        for biais_detecte in biais:
            if biais_detecte.intensite > 0.7:
                texte_ameliore = self.detecteur_biais.corriger_biais(texte_ameliore, biais_detecte)
        
        # Ajouter des avertissements sur les limites
        if incertitudes:
            avertissement = self.conscience_limites.generer_avertissement_limites(incertitudes)
            texte_ameliore += avertissement
        
        # Ajouter une réflexion métacognitive si la qualité est faible
        if metriques.calculer_score_global() < 0.7:
            reflexion = self._generer_reflexion_metacognitive(metriques)
            texte_ameliore += f"\n\n{reflexion}"
        
        return texte_ameliore
    
    def _generer_reflexion_metacognitive(self, metriques: MetriqueQualite) -> str:
        """Génère une réflexion métacognitive sur la qualité."""
        problemes = self.auto_evaluateur.identifier_problemes_qualite(metriques)
        
        if not problemes:
            return ""
        
        reflexion = "🤔 **Réflexion métacognitive**\n"
        reflexion += "J'identifie les axes d'amélioration suivants dans ma réponse:\n"
        
        for i, probleme in enumerate(problemes, 1):
            reflexion += f"{i}. {probleme}\n"
        
        reflexion += "\nJe m'engage à améliorer ces aspects dans mes prochaines réponses."
        
        return reflexion
    
    def _enregistrer_session(self, session_id: str, data: Dict[str, Any], 
                           metriques: MetriqueQualite, biais: List[BiaisCognitif]) -> None:
        """Enregistre une session pour analyse future."""
        session = {
            'id': session_id,
            'timestamp': datetime.now(),
            'score_qualite': metriques.calculer_score_global(),
            'nb_biais_detectes': len(biais),
            'domaine': data.get('domain', 'general'),
            'longueur_reponse': len(data.get('text', '')),
            'optimisations_appliquees': bool(data.get('optimisations_suggerees'))
        }
        
        self.historique_sessions.append(session)
        
        # Limiter la taille de l'historique
        if len(self.historique_sessions) > 1000:
            self.historique_sessions = self.historique_sessions[-500:]
    
    def generer_rapport_complet(self) -> Dict[str, Any]:
        """Génère un rapport complet sur la métacognition."""
        return {
            "auto_evaluation": {
                "evaluations_effectuees": len(self.auto_evaluateur.historique_evaluations),
                "tendances_qualite": self.auto_evaluateur.obtenir_tendances_qualite()
            },
            "detection_biais": self.detecteur_biais.obtenir_statistiques_biais(),
            "monitoring_processus": self.moniteur_processus.obtenir_rapport_performance(),
            "conscience_limites": self.conscience_limites.obtenir_cartographie_limites(),
            "apprentissage": self.strategie_metacognitive.obtenir_rapport_apprentissage(),
            "sessions_total": len(self.historique_sessions),
            "performance_globale": self._calculer_performance_globale()
        }
    
    def _calculer_performance_globale(self) -> Dict[str, float]:
        """Calcule la performance globale du système."""
        if not self.historique_sessions:
            return {}
        
        scores = [s['score_qualite'] for s in self.historique_sessions]
        nb_biais = [s['nb_biais_detectes'] for s in self.historique_sessions]
        
        return {
            "qualite_moyenne": statistics.mean(scores),
            "qualite_tendance": scores[-1] - scores[0] if len(scores) > 1 else 0,
            "biais_moyen": statistics.mean(nb_biais),
            "taux_optimisation": sum(1 for s in self.historique_sessions if s['optimisations_appliquees']) / len(self.historique_sessions)
        }

# Instance globale
metacognition_system = MetacognitionProfonde()

def process(data: Dict[str, Any], hook: str) -> Dict[str, Any]:
    """
    Point d'entrée principal pour le traitement métacognitif.
    
    Args:
        data: Les données à traiter
        hook: Le type de hook (process_request, process_response, etc.)
        
    Returns:
        Les données modifiées avec métacognition
    """
    try:
        if hook == "process_request" or hook == "pre_reasoning":
            return metacognition_system.processer_requete(data)
        
        elif hook == "process_response" or hook == "post_reasoning":
            return metacognition_system.processer_reponse(data)
        
        return data
    
    except Exception as e:
        logger.error(f"Erreur dans le traitement métacognitif global: {str(e)}")
        return data

# Fonctions utilitaires pour l'interface externe
def obtenir_rapport_metacognition() -> Dict[str, Any]:
    """Retourne un rapport complet sur l'état de la métacognition."""
    return metacognition_system.generer_rapport_complet()

def activer_mode_debug(actif: bool = True) -> None:
    """Active ou désactive le mode debug."""
    metacognition_system.mode_debug = actif
    logger.info(f"Mode debug métacognition: {'activé' if actif else 'désactivé'}")

def obtenir_historique_reflexions(limite: int = 50) -> List[Dict[str, Any]]:
    """Retourne l'historique des réflexions métacognitives."""
    return metacognition_system.historique_sessions[-limite:]

# Test et exemple d'utilisation
if __name__ == "__main__":
    # Test du système de métacognition
    def test_metacognition():
        # Simulation d'une requête
        requete = {
            'text': 'Expliquez-moi comment résoudre le problème du réchauffement climatique',
            'domain': 'environnement'
        }
        
        # Traitement de la requête
        requete_traitee = process(requete, 'process_request')
        print("Requête traitée:", requete_traitee.get('metacognition', {}).get('session_id'))
        
        # Simulation d'une réponse
        reponse = {
            **requete_traitee,
            'text': 'Le réchauffement climatique est un problème complexe qui nécessite des solutions à plusieurs niveaux. Évidemment, la transition énergétique est la solution principale. Il faut absolument passer aux énergies renouvelables sans aucun doute.'
        }
        
        # Traitement de la réponse
        reponse_traitee = process(reponse, 'process_response')
        
        print("\n=== RÉSULTAT DE L'ANALYSE MÉTACOGNITIVE ===")
        print(f"Texte final: {reponse_traitee['text'][:200]}...")
        
        if 'metriques_qualite' in reponse_traitee:
            metriques = reponse_traitee['metriques_qualite']
            print(f"Score de qualité: {metriques.calculer_score_global():.2f}")
        
        if 'biais_detectes' in reponse_traitee:
            biais = reponse_traitee['biais_detectes']
            print(f"Biais détectés: {len(biais)}")
            for b in biais:
                print(f"  - {b.type_biais.value}: {b.intensite:.2f}")
        
        # Rapport final
        rapport = obtenir_rapport_metacognition()
        print(f"\nSessions traitées: {rapport['sessions_total']}")
    
    test_metacognition()
