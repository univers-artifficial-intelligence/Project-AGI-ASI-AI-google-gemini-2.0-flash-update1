"""
Système de Raisonnement Probabiliste et Gestion de l'Incertitude
================================================================

Un système complet pour l'intégration de l'incertitude dans les processus de raisonnement,
incluant l'inférence bayésienne, la gestion des probabilités conditionnelles,
la calibration de confiance et la prise de décision sous incertitude.


"""

import numpy as np
import scipy.stats as stats
from scipy.special import logsumexp, gammaln
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass
import warnings
from abc import ABC, abstractmethod
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UncertaintyQuantification:
    """Structure pour quantifier l'incertitude"""
    epistemic: float  # Incertitude due au manque de connaissance
    aleatoric: float  # Incertitude inhérente au système
    total: float      # Incertitude totale
    confidence: float # Niveau de confiance

class BaseUncertaintyModel(ABC):
    """Classe de base pour tous les modèles d'incertitude"""
    
    @abstractmethod
    def compute_uncertainty(self, *args, **kwargs) -> UncertaintyQuantification:
        """Calcule l'incertitude pour les données fournies"""
        pass
    
    @abstractmethod
    def update_beliefs(self, evidence: Any) -> None:
        """Met à jour les croyances basées sur nouvelles preuves"""
        pass

class ProbabilisticDistribution:
    """Classe pour représenter et manipuler les distributions probabilistes"""
    
    def __init__(self, distribution_type: str, parameters: Dict[str, float]):
        self.distribution_type = distribution_type
        self.parameters = parameters
        self._distribution = self._create_distribution()
    
    def _create_distribution(self):
        """Crée l'objet distribution scipy correspondant"""
        if self.distribution_type == 'normal':
            return stats.norm(loc=self.parameters['mean'], 
                            scale=self.parameters['std'])
        elif self.distribution_type == 'beta':
            return stats.beta(a=self.parameters['alpha'], 
                            b=self.parameters['beta'])
        elif self.distribution_type == 'gamma':
            return stats.gamma(a=self.parameters['shape'], 
                             scale=self.parameters['scale'])
        elif self.distribution_type == 'dirichlet':
            return stats.dirichlet(alpha=self.parameters['alpha'])
        elif self.distribution_type == 'uniform':
            return stats.uniform(loc=self.parameters['low'], 
                               scale=self.parameters['high'] - self.parameters['low'])
        else:
            raise ValueError(f"Distribution type {self.distribution_type} not supported")
    
    def pdf(self, x):
        """Densité de probabilité"""
        return self._distribution.pdf(x)
    
    def logpdf(self, x):
        """Log-densité de probabilité"""
        return self._distribution.logpdf(x)
    
    def cdf(self, x):
        """Fonction de répartition cumulative"""
        return self._distribution.cdf(x)
    
    def sample(self, size: int = 1):
        """Échantillonnage de la distribution"""
        return self._distribution.rvs(size=size)
    
    def mean(self):
        """Moyenne de la distribution"""
        return self._distribution.mean()
    
    def var(self):
        """Variance de la distribution"""
        return self._distribution.var()
    
    def entropy(self):
        """Entropie de la distribution"""
        return self._distribution.entropy()

class BayesianNetwork:
    """Réseau bayésien pour modéliser les dépendances probabilistes"""
    
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.conditional_distributions = {}
        self.evidence = {}
    
    def add_node(self, name: str, distribution: ProbabilisticDistribution):
        """Ajoute un nœud au réseau"""
        self.nodes[name] = distribution
        self.edges[name] = []
    
    def add_edge(self, parent: str, child: str):
        """Ajoute une arête dirigée parent -> child"""
        if parent not in self.edges:
            self.edges[parent] = []
        self.edges[parent].append(child)
    
    def set_conditional_distribution(self, node: str, parents: List[str], 
                                   conditional_func: Callable):
        """Définit la distribution conditionnelle P(node|parents)"""
        self.conditional_distributions[node] = {
            'parents': parents,
            'function': conditional_func
        }
    
    def set_evidence(self, node: str, value: Any):
        """Définit une évidence pour un nœud"""
        self.evidence[node] = value
    
    def forward_sampling(self, num_samples: int = 1000) -> Dict[str, np.ndarray]:
        """Échantillonnage forward pour générer des échantillons du réseau"""
        samples = {node: [] for node in self.nodes}
        
        # Tri topologique des nœuds
        ordered_nodes = self._topological_sort()
        
        for _ in range(num_samples):
            sample = {}
            for node in ordered_nodes:
                if node in self.evidence:
                    sample[node] = self.evidence[node]
                elif node in self.conditional_distributions:
                    parents = self.conditional_distributions[node]['parents']
                    parent_values = [sample[p] for p in parents]
                    func = self.conditional_distributions[node]['function']
                    sample[node] = func(*parent_values)
                else:
                    sample[node] = self.nodes[node].sample()
                
                samples[node].append(sample[node])
        
        return {node: np.array(values) for node, values in samples.items()}
    
    def _topological_sort(self) -> List[str]:
        """Tri topologique des nœuds"""
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(node):
            if node in temp_visited:
                raise ValueError("Cycle détecté dans le réseau bayésien")
            if node not in visited:
                temp_visited.add(node)
                for child in self.edges.get(node, []):
                    visit(child)
                temp_visited.remove(node)
                visited.add(node)
                result.insert(0, node)
        
        for node in self.nodes:
            if node not in visited:
                visit(node)
        
        return result

class BayesianInference:
    """Classe pour l'inférence bayésienne et la mise à jour des croyances"""
    
    def __init__(self):
        self.priors = {}
        self.posteriors = {}
        self.evidence_history = []
        self.belief_history = []
    
    def set_prior(self, parameter: str, distribution: ProbabilisticDistribution):
        """Définit une distribution a priori pour un paramètre"""
        self.priors[parameter] = distribution
        self.posteriors[parameter] = distribution
    
    def bayesian_update(self, parameter: str, likelihood_func: Callable, 
                       evidence: Any) -> ProbabilisticDistribution:
        """Met à jour les croyances bayésiennes basées sur de nouvelles preuves"""
        if parameter not in self.posteriors:
            raise ValueError(f"Paramètre {parameter} non trouvé")
        
        prior = self.posteriors[parameter]
        
        # Calcul de la vraisemblance
        def unnormalized_posterior(x):
            prior_prob = prior.pdf(x)
            likelihood = likelihood_func(x, evidence)
            return prior_prob * likelihood
        
        # Normalisation par intégration numérique
        from scipy.integrate import quad
        
        # Pour simplicité, supposons une distribution normale mise à jour
        # Dans une implémentation complète, utilisez MCMC ou VI
        if prior.distribution_type == 'normal':
            # Mise à jour conjuguée pour normale-normale
            if hasattr(evidence, '__len__'):
                n = len(evidence)
                sample_mean = np.mean(evidence)
                sample_var = np.var(evidence)
                
                # Paramètres du prior
                prior_mean = prior.parameters['mean']
                prior_var = prior.parameters['std']**2
                
                # Mise à jour bayésienne
                posterior_var = 1 / (1/prior_var + n/sample_var)
                posterior_mean = posterior_var * (prior_mean/prior_var + n*sample_mean/sample_var)
                
                posterior = ProbabilisticDistribution(
                    'normal', 
                    {'mean': posterior_mean, 'std': np.sqrt(posterior_var)}
                )
            else:
                # Single observation
                observation = evidence
                prior_mean = prior.parameters['mean']
                prior_var = prior.parameters['std']**2
                obs_var = 1.0  # Supposé
                
                posterior_var = 1 / (1/prior_var + 1/obs_var)
                posterior_mean = posterior_var * (prior_mean/prior_var + observation/obs_var)
                
                posterior = ProbabilisticDistribution(
                    'normal',
                    {'mean': posterior_mean, 'std': np.sqrt(posterior_var)}
                )
        else:
            # Pour d'autres distributions, utiliser MCMC ou approximation
            posterior = self._approximate_posterior(prior, likelihood_func, evidence)
        
        self.posteriors[parameter] = posterior
        self.evidence_history.append(evidence)
        self.belief_history.append(posterior)
        
        return posterior
    
    def _approximate_posterior(self, prior: ProbabilisticDistribution, 
                             likelihood_func: Callable, evidence: Any) -> ProbabilisticDistribution:
        """Approximation de la distribution a posteriori"""
        # Implémentation simplifiée - dans la pratique, utiliser MCMC/VI
        samples = prior.sample(10000)
        log_weights = []
        
        for sample in samples:
            log_prior = prior.logpdf(sample)
            log_likelihood = np.log(likelihood_func(sample, evidence))
            log_weights.append(log_prior + log_likelihood)
        
        log_weights = np.array(log_weights)
        weights = np.exp(log_weights - logsumexp(log_weights))
        
        # Approximation par une distribution normale
        weighted_mean = np.average(samples, weights=weights)
        weighted_var = np.average((samples - weighted_mean)**2, weights=weights)
        
        return ProbabilisticDistribution(
            'normal',
            {'mean': weighted_mean, 'std': np.sqrt(weighted_var)}
        )
    
    def compute_marginal_likelihood(self, parameter: str, likelihood_func: Callable, 
                                  evidence: Any) -> float:
        """Calcule la vraisemblance marginale (évidence)"""
        prior = self.priors[parameter]
        
        def integrand(x):
            return prior.pdf(x) * likelihood_func(x, evidence)
        
        from scipy.integrate import quad
        result, _ = quad(integrand, -10, 10)  # Limites ajustables
        return result
    
    def compute_bayes_factor(self, param1: str, param2: str, likelihood_func: Callable, 
                           evidence: Any) -> float:
        """Calcule le facteur de Bayes entre deux modèles"""
        ml1 = self.compute_marginal_likelihood(param1, likelihood_func, evidence)
        ml2 = self.compute_marginal_likelihood(param2, likelihood_func, evidence)
        return ml1 / ml2

class ConditionalProbabilityManager:
    """Gestionnaire pour les probabilités conditionnelles complexes"""
    
    def __init__(self):
        self.conditional_tables = {}
        self.independence_assumptions = {}
        self.causal_graph = {}
    
    def add_conditional_probability(self, event: str, given: List[str], 
                                  probability_table: Dict[Tuple, float]):
        """Ajoute une table de probabilité conditionnelle P(event|given)"""
        self.conditional_tables[event] = {
            'given': given,
            'table': probability_table
        }
    
    def compute_conditional_probability(self, event: str, given_values: Dict[str, Any]) -> float:
        """Calcule P(event|given_values)"""
        if event not in self.conditional_tables:
            raise ValueError(f"Probabilité conditionnelle pour {event} non définie")
        
        table_info = self.conditional_tables[event]
        given_vars = table_info['given']
        table = table_info['table']
        
        # Construire la clé pour la table
        key = tuple(given_values[var] for var in given_vars)
        
        if key in table:
            return table[key]
        else:
            raise ValueError(f"Combinaison de valeurs {key} non trouvée dans la table")
    
    def compute_joint_probability(self, events: Dict[str, Any]) -> float:
        """Calcule la probabilité jointe en utilisant la règle de la chaîne"""
        # P(A,B,C) = P(A) * P(B|A) * P(C|A,B)
        probability = 1.0
        
        # Ordre topologique des événements
        ordered_events = list(events.keys())  # Simplification
        
        for i, event in enumerate(ordered_events):
            if i == 0:
                # Probabilité marginale du premier événement
                prob = self._get_marginal_probability(event, events[event])
            else:
                # Probabilité conditionnelle
                given_events = {e: events[e] for e in ordered_events[:i]}
                prob = self.compute_conditional_probability(event, given_events)
            
            probability *= prob
        
        return probability
    
    def _get_marginal_probability(self, event: str, value: Any) -> float:
        """Obtient la probabilité marginale d'un événement"""
        # Implémentation simplifiée - dans la pratique, calculer à partir des tables
        return 0.5  # Placeholder
    
    def check_conditional_independence(self, event_a: str, event_b: str, 
                                     given: List[str]) -> bool:
        """Vérifie l'indépendance conditionnelle P(A|C) = P(A|B,C)"""
        # Implémentation basée sur la structure du graphe causal
        return self._d_separation(event_a, event_b, given)
    
    def _d_separation(self, node_a: str, node_b: str, conditioning_set: List[str]) -> bool:
        """Algorithme de d-séparation pour vérifier l'indépendance conditionnelle"""
        # Implémentation simplifiée du test de d-séparation
        # Dans une implémentation complète, implémenter l'algorithme complet
        return False  # Placeholder

class ConfidenceCalibrator:
    """Classe pour la calibration de la confiance et quantification de l'incertitude épistémique"""
    
    def __init__(self):
        self.calibration_data = []
        self.calibration_function = None
        self.reliability_diagram_data = None
    
    def add_prediction(self, confidence: float, correct: bool):
        """Ajoute une prédiction avec son niveau de confiance et sa véracité"""
        self.calibration_data.append((confidence, correct))
    
    def compute_calibration_error(self, num_bins: int = 10) -> Tuple[float, Dict]:
        """Calcule l'erreur de calibration (ECE - Expected Calibration Error)"""
        if not self.calibration_data:
            return 0.0, {}
        
        confidences = np.array([c for c, _ in self.calibration_data])
        corrects = np.array([c for _, c in self.calibration_data])
        
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        bin_data = {}
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = corrects[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                bin_data[f'bin_{bin_lower:.1f}_{bin_upper:.1f}'] = {
                    'confidence': avg_confidence_in_bin,
                    'accuracy': accuracy_in_bin,
                    'proportion': prop_in_bin
                }
        
        self.reliability_diagram_data = bin_data
        return ece, bin_data
    
    def temperature_scaling(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Calibration par temperature scaling"""
        from scipy.optimize import minimize_scalar
        
        def loss(temperature):
            scaled_logits = logits / temperature
            # Calcul de la log-vraisemblance
            exp_logits = np.exp(scaled_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Cross-entropy loss
            return -np.mean(np.log(probs[np.arange(len(labels)), labels]))
        
        result = minimize_scalar(loss, bounds=(0.1, 10.0), method='bounded')
        optimal_temperature = result.x
        
        return optimal_temperature
    
    def platt_scaling(self, scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        """Calibration par Platt scaling (régression logistique)"""
        from sklearn.linear_model import LogisticRegression
        
        lr = LogisticRegression()
        lr.fit(scores.reshape(-1, 1), labels)
        
        # Paramètres A et B pour P(y=1|f) = 1/(1 + exp(A*f + B))
        A = lr.coef_[0][0]
        B = lr.intercept_[0]
        
        return A, B
    
    def compute_epistemic_uncertainty(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Calcule l'incertitude épistémique à partir de prédictions multiples"""
        # Variance des prédictions = incertitude épistémique
        predictions_array = np.array(predictions)
        epistemic_uncertainty = np.var(predictions_array, axis=0)
        return epistemic_uncertainty
    
    def compute_aleatoric_uncertainty(self, prediction_variances: List[np.ndarray]) -> np.ndarray:
        """Calcule l'incertitude aléatoire moyenne"""
        # Moyenne des variances = incertitude aléatoire
        variances_array = np.array(prediction_variances)
        aleatoric_uncertainty = np.mean(variances_array, axis=0)
        return aleatoric_uncertainty
    
    def decompose_uncertainty(self, predictions: List[np.ndarray], 
                            variances: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Décompose l'incertitude totale en épistémique et aléatoire"""
        epistemic = self.compute_epistemic_uncertainty(predictions)
        aleatoric = self.compute_aleatoric_uncertainty(variances)
        total = epistemic + aleatoric
        
        return {
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'total': total
        }

class GameTheoryDecisionMaker:
    """Prise de décision sous incertitude avec théorie des jeux"""
    
    def __init__(self):
        self.payoff_matrices = {}
        self.strategies = {}
        self.uncertainty_models = {}
    
    def add_game(self, game_name: str, players: List[str], 
                strategies: Dict[str, List[str]], 
                payoff_matrix: np.ndarray):
        """Ajoute un jeu à la base de données"""
        self.strategies[game_name] = strategies
        self.payoff_matrices[game_name] = payoff_matrix
    
    def nash_equilibrium(self, game_name: str) -> List[Tuple]:
        """Calcule les équilibres de Nash"""
        # Implémentation simplifiée pour jeux 2x2
        payoff_matrix = self.payoff_matrices[game_name]
        
        if payoff_matrix.shape == (2, 2, 2):  # Jeu 2x2 à 2 joueurs
            equilibria = []
            
            # Vérifier les équilibres en stratégies pures
            for i in range(2):
                for j in range(2):
                    if self._is_nash_equilibrium(payoff_matrix, i, j):
                        equilibria.append((i, j))
            
            # Calculer l'équilibre en stratégies mixtes s'il n'y a pas d'équilibre pur
            if not equilibria:
                mixed_eq = self._compute_mixed_nash(payoff_matrix)
                if mixed_eq:
                    equilibria.append(mixed_eq)
            
            return equilibria
        else:
            raise NotImplementedError("Seulement les jeux 2x2 sont supportés pour l'instant")
    
    def _is_nash_equilibrium(self, payoff_matrix: np.ndarray, i: int, j: int) -> bool:
        """Vérifie si (i,j) est un équilibre de Nash"""
        # Joueur 1: vérifier si i est la meilleure réponse à j
        player1_payoffs = payoff_matrix[:, j, 0]
        best_response_1 = np.argmax(player1_payoffs)
        
        # Joueur 2: vérifier si j est la meilleure réponse à i
        player2_payoffs = payoff_matrix[i, :, 1]
        best_response_2 = np.argmax(player2_payoffs)
        
        return best_response_1 == i and best_response_2 == j
    
    def _compute_mixed_nash(self, payoff_matrix: np.ndarray) -> Optional[Tuple]:
        """Calcule l'équilibre de Nash en stratégies mixtes pour un jeu 2x2"""
        # Pour un jeu 2x2, l'équilibre mixte existe si aucun équilibre pur n'existe
        
        # Payoffs pour chaque joueur
        A = payoff_matrix[:, :, 0]  # Payoffs joueur 1
        B = payoff_matrix[:, :, 1]  # Payoffs joueur 2
        
        # Probabilité que joueur 1 joue stratégie 0
        if A[1, 0] - A[1, 1] - A[0, 0] + A[0, 1] != 0:
            p = (B[1, 1] - B[0, 1]) / (B[1, 0] - B[1, 1] - B[0, 0] + B[0, 1])
        else:
            return None
        
        # Probabilité que joueur 2 joue stratégie 0
        if B[0, 1] - B[1, 1] - B[0, 0] + B[1, 0] != 0:
            q = (A[1, 1] - A[1, 0]) / (A[0, 0] - A[0, 1] - A[1, 0] + A[1, 1])
        else:
            return None
        
        if 0 <= p <= 1 and 0 <= q <= 1:
            return ((p, 1-p), (q, 1-q))
        else:
            return None
    
    def maximin_strategy(self, game_name: str, player: int) -> Tuple[int, float]:
        """Stratégie maximin (maximiser le gain minimum)"""
        payoff_matrix = self.payoff_matrices[game_name]
        
        if player == 0:
            # Joueur 1: maximiser le minimum sur les colonnes
            min_payoffs = np.min(payoff_matrix[:, :, 0], axis=1)
            best_strategy = np.argmax(min_payoffs)
            best_payoff = min_payoffs[best_strategy]
        else:
            # Joueur 2: maximiser le minimum sur les lignes
            min_payoffs = np.min(payoff_matrix[:, :, 1], axis=0)
            best_strategy = np.argmax(min_payoffs)
            best_payoff = min_payoffs[best_strategy]
        
        return best_strategy, best_payoff
    
    def minimax_regret(self, game_name: str, player: int) -> Tuple[int, float]:
        """Stratégie minimax regret"""
        payoff_matrix = self.payoff_matrices[game_name]
        
        if player == 0:
            payoffs = payoff_matrix[:, :, 0]
            # Calcul de la matrice de regret
            max_per_column = np.max(payoffs, axis=0)
            regret_matrix = max_per_column - payoffs
            
            # Minimax regret
            max_regret_per_row = np.max(regret_matrix, axis=1)
            best_strategy = np.argmin(max_regret_per_row)
            min_regret = max_regret_per_row[best_strategy]
        else:
            payoffs = payoff_matrix[:, :, 1]
            max_per_row = np.max(payoffs, axis=1)
            regret_matrix = max_per_row.reshape(-1, 1) - payoffs
            
            max_regret_per_column = np.max(regret_matrix, axis=0)
            best_strategy = np.argmin(max_regret_per_column)
            min_regret = max_regret_per_column[best_strategy]
        
        return best_strategy, min_regret
    
    def expected_utility_maximization(self, game_name: str, player: int, 
                                    opponent_strategy_distribution: np.ndarray) -> Tuple[int, float]:
        """Maximisation de l'utilité espérée given une distribution sur les stratégies adverses"""
        payoff_matrix = self.payoff_matrices[game_name]
        
        if player == 0:
            payoffs = payoff_matrix[:, :, 0]
            expected_payoffs = np.dot(payoffs, opponent_strategy_distribution)
        else:
            payoffs = payoff_matrix[:, :, 1]
            expected_payoffs = np.dot(payoffs.T, opponent_strategy_distribution)
        
        best_strategy = np.argmax(expected_payoffs)
        best_expected_payoff = expected_payoffs[best_strategy]
        
        return best_strategy, best_expected_payoff
    
    def robust_decision_making(self, game_name: str, player: int, 
                             uncertainty_set: List[np.ndarray]) -> Tuple[int, float]:
        """Prise de décision robuste sous incertitude d'ensemble"""
        payoff_matrix = self.payoff_matrices[game_name]
        
        worst_case_payoffs = []
        
        for strategy_idx in range(payoff_matrix.shape[player]):
            worst_payoff = float('inf')
            
            for uncertain_payoff in uncertainty_set:
                if player == 0:
                    payoff = uncertain_payoff[strategy_idx, :, 0].min()
                else:
                    payoff = uncertain_payoff[:, strategy_idx, 1].min()
                
                worst_payoff = min(worst_payoff, payoff)
            
            worst_case_payoffs.append(worst_payoff)
        
        best_strategy = np.argmax(worst_case_payoffs)
        best_worst_case = worst_case_payoffs[best_strategy]
        
        return best_strategy, best_worst_case

class IntegratedUncertaintySystem:
    """Système intégré combinant tous les composants de gestion de l'incertitude"""
    
    def __init__(self):
        self.bayesian_inference = BayesianInference()
        self.bayesian_network = BayesianNetwork()
        self.conditional_manager = ConditionalProbabilityManager()
        self.confidence_calibrator = ConfidenceCalibrator()
        self.game_theory_dm = GameTheoryDecisionMaker()
        self.uncertainty_history = []
    
    def process_uncertain_reasoning(self, evidence: Dict[str, Any], 
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Processus de raisonnement intégrant l'incertitude"""
        results = {}
        
        # 1. Mise à jour bayésienne des croyances
        for parameter, value in evidence.items():
            if parameter in self.bayesian_inference.posteriors:
                likelihood_func = context.get(f'{parameter}_likelihood', 
                                            lambda x, e: stats.norm.pdf(e, loc=x, scale=1))
                posterior = self.bayesian_inference.bayesian_update(
                    parameter, likelihood_func, value
                )
                results[f'{parameter}_posterior'] = posterior
        
        # 2. Inference dans le réseau bayésien
        if hasattr(context, 'network_evidence'):
            for node, value in context['network_evidence'].items():
                self.bayesian_network.set_evidence(node, value)
            
            network_samples = self.bayesian_network.forward_sampling(1000)
            results['network_inference'] = network_samples
        
        # 3. Quantification de l'incertitude
        epistemic_uncertainty = self._compute_epistemic_uncertainty(evidence)
        aleatoric_uncertainty = self._compute_aleatoric_uncertainty(evidence)
        
        uncertainty_quantification = UncertaintyQuantification(
            epistemic=epistemic_uncertainty,
            aleatoric=aleatoric_uncertainty,
            total=epistemic_uncertainty + aleatoric_uncertainty,
            confidence=self._compute_confidence_level(epistemic_uncertainty, aleatoric_uncertainty)
        )
        
        results['uncertainty'] = uncertainty_quantification
        
        # 4. Prise de décision sous incertitude
        if 'decision_context' in context:
            decision_context = context['decision_context']
            if 'game_name' in decision_context:
                game_name = decision_context['game_name']
                player = decision_context['player']
                
                # Différentes stratégies de décision
                nash_eq = self.game_theory_dm.nash_equilibrium(game_name)
                maximin = self.game_theory_dm.maximin_strategy(game_name, player)
                
                results['decision_analysis'] = {
                    'nash_equilibria': nash_eq,
                    'maximin_strategy': maximin,
                    'uncertainty_level': uncertainty_quantification.total
                }
        
        # Enregistrer l'historique
        self.uncertainty_history.append({
            'timestamp': np.datetime64('now'),
            'evidence': evidence,
            'results': results
        })
        
        return results
    
    def _compute_epistemic_uncertainty(self, evidence: Dict[str, Any]) -> float:
        """Calcule l'incertitude épistémique basée sur l'évidence"""
        # Simplification: basé sur la variance des posteriors
        total_epistemic = 0.0
        count = 0
        
        for param, posterior in self.bayesian_inference.posteriors.items():
            if hasattr(posterior, 'var'):
                total_epistemic += posterior.var()
                count += 1
        
        return total_epistemic / max(count, 1)
    
    def _compute_aleatoric_uncertainty(self, evidence: Dict[str, Any]) -> float:
        """Calcule l'incertitude aléatoire intrinsèque"""
        # Simplification: basé sur la variabilité des observations
        if isinstance(evidence, dict) and len(evidence) > 0:
            values = [v for v in evidence.values() if isinstance(v, (int, float))]
            if values:
                return np.var(values)
        return 0.1  # Valeur par défaut
    
    def _compute_confidence_level(self, epistemic: float, aleatoric: float) -> float:
        """Calcule le niveau de confiance basé sur les incertitudes"""
        total_uncertainty = epistemic + aleatoric
        # Transformation logistique pour obtenir une valeur entre 0 et 1
        confidence = 1 / (1 + np.exp(total_uncertainty))
        return confidence
    
    def visualize_uncertainty(self, parameter: str = None):
        """Visualise l'évolution de l'incertitude"""
        if not self.uncertainty_history:
            print("Aucune donnée d'historique disponible")
            return
        
        timestamps = [entry['timestamp'] for entry in self.uncertainty_history]
        uncertainties = [entry['results']['uncertainty'].total for entry in self.uncertainty_history]
        confidences = [entry['results']['uncertainty'].confidence for entry in self.uncertainty_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Graphique de l'incertitude totale
        ax1.plot(timestamps, uncertainties, 'b-', label='Incertitude totale')
        ax1.set_ylabel('Incertitude')
        ax1.set_title('Évolution de l\'incertitude dans le temps')
        ax1.legend()
        ax1.grid(True)
        
        # Graphique de la confiance
        ax2.plot(timestamps, confidences, 'r-', label='Niveau de confiance')
        ax2.set_ylabel('Confiance')
        ax2.set_xlabel('Temps')
        ax2.set_title('Évolution de la confiance dans le temps')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def generate_uncertainty_report(self) -> Dict[str, Any]:
        """Génère un rapport détaillé sur l'état de l'incertitude"""
        report = {
            'timestamp': np.datetime64('now'),
            'system_state': {
                'num_parameters_tracked': len(self.bayesian_inference.posteriors),
                'num_network_nodes': len(self.bayesian_network.nodes),
                'num_conditional_tables': len(self.conditional_manager.conditional_tables),
                'history_length': len(self.uncertainty_history)
            },
            'current_uncertainties': {},
            'calibration_metrics': {},
            'recommendations': []
        }
        
        # État actuel des incertitudes par paramètre
        for param, posterior in self.bayesian_inference.posteriors.items():
            report['current_uncertainties'][param] = {
                'mean': posterior.mean(),
                'variance': posterior.var(),
                'entropy': posterior.entropy()
            }
        
        # Métriques de calibration
        if self.confidence_calibrator.calibration_data:
            ece, bin_data = self.confidence_calibrator.compute_calibration_error()
            report['calibration_metrics'] = {
                'expected_calibration_error': ece,
                'reliability_diagram_data': bin_data
            }
        
        # Recommandations basées sur l'état du système
        if len(self.uncertainty_history) > 10:
            recent_uncertainties = [
                entry['results']['uncertainty'].total 
                for entry in self.uncertainty_history[-10:]
            ]
            if np.std(recent_uncertainties) > 0.1:
                report['recommendations'].append(
                    "L'incertitude montre une forte variabilité récente. "
                    "Considérer l'acquisition de données supplémentaires."
                )
        
        return report

# Fonctions utilitaires pour l'utilisation du système

def create_simple_uncertainty_model(prior_params: Dict[str, Dict[str, float]]) -> IntegratedUncertaintySystem:
    """Crée un modèle d'incertitude simple avec des priors spécifiés"""
    system = IntegratedUncertaintySystem()
    
    for param_name, param_config in prior_params.items():
        dist_type = param_config.get('type', 'normal')
        dist_params = {k: v for k, v in param_config.items() if k != 'type'}
        
        prior = ProbabilisticDistribution(dist_type, dist_params)
        system.bayesian_inference.set_prior(param_name, prior)
    
    return system

def demo_uncertainty_system():
    """Démonstration du système de gestion de l'incertitude"""
    print("=== Démonstration du Système de Raisonnement Probabiliste ===\n")
    
    # Créer un système avec des priors
    prior_params = {
        'temperature': {'type': 'normal', 'mean': 20.0, 'std': 5.0},
        'humidity': {'type': 'beta', 'alpha': 2.0, 'beta': 3.0},
        'pressure': {'type': 'gamma', 'shape': 2.0, 'scale': 1.0}
    }
    
    system = create_simple_uncertainty_model(prior_params)
    
    # Simuler des observations
    observations = {
        'temperature': [22.1, 21.8, 23.2, 20.9, 22.5],
        'humidity': 0.65,
        'pressure': 2.3
    }
    
    print("1. Priors initiaux:")
    for param, prior in system.bayesian_inference.priors.items():
        print(f"   {param}: moyenne={prior.mean():.2f}, variance={prior.var():.4f}")
    
    print("\n2. Mise à jour bayésienne avec observations:")
    
    # Fonction de vraisemblance simple
    def simple_likelihood(param_value, observation):
        if isinstance(observation, list):
            return np.prod([stats.norm.pdf(obs, loc=param_value, scale=1.0) for obs in observation])
        else:
            return stats.norm.pdf(observation, loc=param_value, scale=1.0)
    
    # Traitement des observations
    context = {
        'temperature_likelihood': simple_likelihood,
        'humidity_likelihood': simple_likelihood,
        'pressure_likelihood': simple_likelihood
    }
    
    results = system.process_uncertain_reasoning(observations, context)
    
    print("\n3. Posteriors mis à jour:")
    for param, posterior in system.bayesian_inference.posteriors.items():
        print(f"   {param}: moyenne={posterior.mean():.2f}, variance={posterior.var():.4f}")
    
    print(f"\n4. Quantification de l'incertitude:")
    uncertainty = results['uncertainty']
    print(f"   Incertitude épistémique: {uncertainty.epistemic:.4f}")
    print(f"   Incertitude aléatoire: {uncertainty.aleatoric:.4f}")
    print(f"   Incertitude totale: {uncertainty.total:.4f}")
    print(f"   Niveau de confiance: {uncertainty.confidence:.4f}")
    
    # Démonstration de la théorie des jeux
    print("\n5. Exemple de prise de décision avec théorie des jeux:")
    
    # Jeu simple: Dilemme du prisonnier modifié
    payoff_matrix = np.array([
        [[3, 3], [0, 5]],   # Coopérer
        [[5, 0], [1, 1]]    # Trahir
    ])
    
    system.game_theory_dm.add_game(
        'prisoner_dilemma',
        ['Player1', 'Player2'],
        {'Player1': ['Cooperate', 'Defect'], 'Player2': ['Cooperate', 'Defect']},
        payoff_matrix
    )
    
    nash_equilibria = system.game_theory_dm.nash_equilibrium('prisoner_dilemma')
    maximin_p1 = system.game_theory_dm.maximin_strategy('prisoner_dilemma', 0)
    
    print(f"   Équilibres de Nash: {nash_equilibria}")
    print(f"   Stratégie maximin joueur 1: stratégie {maximin_p1[0]}, gain {maximin_p1[1]}")
    
    print("\n6. Rapport de l'état du système:")
    report = system.generate_uncertainty_report()
    print(f"   Nombre de paramètres suivis: {report['system_state']['num_parameters_tracked']}")
    print(f"   Longueur de l'historique: {report['system_state']['history_length']}")
    
    if report['recommendations']:
        print("   Recommandations:")
        for rec in report['recommendations']:
            print(f"   - {rec}")

if __name__ == "__main__":
    # Exécuter la démonstration
    demo_uncertainty_system()
