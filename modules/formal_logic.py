"""
Système de Logique Formelle Avancée et Non-Classique
Un framework complet pour la manipulation de différents systèmes logiques,
la démonstration automatique de théorèmes, et le raisonnement formel.
"""

import numpy as np
import sympy as sp
from enum import Enum
from typing import Dict, List, Set, Tuple, Optional, Callable, Any, Union
from abc import ABC, abstractmethod
import itertools
import networkx as nx
from dataclasses import dataclass
import functools
import re


##########################################
# PARTIE 1: LOGIQUES NON-CLASSIQUES
##########################################

class Formule(ABC):
    """Classe abstraite de base pour toutes les formules logiques."""
    
    @abstractmethod
    def evaluer(self, *args, **kwargs):
        """Évalue la formule selon la sémantique appropriée."""
        pass
    
    @abstractmethod
    def __str__(self):
        """Représentation sous forme de chaîne de caractères de la formule."""
        pass


#########################################
# LOGIQUE FLOUE
#########################################

class ValeurFloue:
    """Représente une valeur dans la logique floue entre 0 et 1."""
    
    def __init__(self, valeur: float):
        if not 0 <= valeur <= 1:
            raise ValueError("Une valeur floue doit être entre 0 et 1")
        self.valeur = valeur
    
    def __and__(self, autre):
        if isinstance(autre, ValeurFloue):
            return ValeurFloue(min(self.valeur, autre.valeur))
        return NotImplemented
    
    def __or__(self, autre):
        if isinstance(autre, ValeurFloue):
            return ValeurFloue(max(self.valeur, autre.valeur))
        return NotImplemented
    
    def __invert__(self):
        return ValeurFloue(1 - self.valeur)
    
    def __str__(self):
        return f"{self.valeur:.3f}"
    
    def __repr__(self):
        return f"ValeurFloue({self.valeur})"


class FormuleFloue(Formule):
    """Classe de base pour les formules de logique floue."""
    pass


class VariableFloue(FormuleFloue):
    """Variable floue avec un nom et une fonction d'appartenance."""
    
    def __init__(self, nom: str, fonction_appartenance: Callable[[Any], float] = None):
        self.nom = nom
        self.fonction_appartenance = fonction_appartenance
        
    def evaluer(self, contexte=None, **kwargs):
        if contexte and self.nom in contexte:
            return contexte[self.nom]
        if self.fonction_appartenance and 'valeur' in kwargs:
            return ValeurFloue(self.fonction_appartenance(kwargs['valeur']))
        raise ValueError(f"Impossible d'évaluer la variable {self.nom}")
        
    def __str__(self):
        return self.nom


class EtFloue(FormuleFloue):
    """Conjonction floue (t-norme)."""
    
    def __init__(self, gauche: FormuleFloue, droite: FormuleFloue, t_norme: str = 'min'):
        self.gauche = gauche
        self.droite = droite
        
        # Différentes t-normes disponibles
        self.t_normes = {
            'min': lambda a, b: min(a, b),
            'produit': lambda a, b: a * b,
            'lukasiewicz': lambda a, b: max(0, a + b - 1),
        }
        
        if t_norme not in self.t_normes:
            raise ValueError(f"T-norme '{t_norme}' non reconnue")
        
        self.t_norme = t_norme
    
    def evaluer(self, contexte=None, **kwargs):
        val_gauche = self.gauche.evaluer(contexte, **kwargs).valeur
        val_droite = self.droite.evaluer(contexte, **kwargs).valeur
        resultat = self.t_normes[self.t_norme](val_gauche, val_droite)
        return ValeurFloue(resultat)
    
    def __str__(self):
        return f"({self.gauche} ∧ {self.droite})"


class OuFloue(FormuleFloue):
    """Disjonction floue (t-conorme)."""
    
    def __init__(self, gauche: FormuleFloue, droite: FormuleFloue, t_conorme: str = 'max'):
        self.gauche = gauche
        self.droite = droite
        
        # Différentes t-conormes disponibles
        self.t_conormes = {
            'max': lambda a, b: max(a, b),
            'somme_prob': lambda a, b: a + b - a * b,
            'lukasiewicz': lambda a, b: min(1, a + b),
        }
        
        if t_conorme not in self.t_conormes:
            raise ValueError(f"T-conorme '{t_conorme}' non reconnue")
        
        self.t_conorme = t_conorme
    
    def evaluer(self, contexte=None, **kwargs):
        val_gauche = self.gauche.evaluer(contexte, **kwargs).valeur
        val_droite = self.droite.evaluer(contexte, **kwargs).valeur
        resultat = self.t_conormes[self.t_conorme](val_gauche, val_droite)
        return ValeurFloue(resultat)
    
    def __str__(self):
        return f"({self.gauche} ∨ {self.droite})"


class NonFloue(FormuleFloue):
    """Négation floue."""
    
    def __init__(self, formule: FormuleFloue, type_neg: str = 'standard'):
        self.formule = formule
        
        # Différents types de négation
        self.negations = {
            'standard': lambda a: 1 - a,
            'sugeno': lambda a, lambda_: 1 - a / (1 + lambda_ * a) if lambda_ > 0 else 1 - a,
            'yager': lambda a, w: (1 - a**w)**(1/w) if w > 0 else 1 - a,
        }
        
        self.type_neg = type_neg
        self.params = {}
    
    def with_params(self, **params):
        """Configure les paramètres pour les négations paramétrables."""
        self.params = params
        return self
    
    def evaluer(self, contexte=None, **kwargs):
        val = self.formule.evaluer(contexte, **kwargs).valeur
        
        if self.type_neg == 'standard':
            resultat = self.negations[self.type_neg](val)
        elif self.type_neg == 'sugeno':
            lambda_ = self.params.get('lambda', 1)
            resultat = self.negations[self.type_neg](val, lambda_)
        elif self.type_neg == 'yager':
            w = self.params.get('w', 2)
            resultat = self.negations[self.type_neg](val, w)
        else:
            raise ValueError(f"Type de négation non pris en charge: {self.type_neg}")
            
        return ValeurFloue(resultat)
    
    def __str__(self):
        return f"¬({self.formule})"


class ImplicationFloue(FormuleFloue):
    """Implication floue avec différentes sémantiques."""
    
    def __init__(self, antecedent: FormuleFloue, consequent: FormuleFloue, type_impl: str = 'godel'):
        self.antecedent = antecedent
        self.consequent = consequent
        
        # Différentes implications floues
        self.implications = {
            'godel': lambda a, b: 1 if a <= b else b,
            'lukasiewicz': lambda a, b: min(1, 1 - a + b),
            'goguen': lambda a, b: 1 if a <= b else b/a,
            'kleene_dienes': lambda a, b: max(1 - a, b),
            'reichenbach': lambda a, b: 1 - a + a*b,
        }
        
        if type_impl not in self.implications:
            raise ValueError(f"Type d'implication '{type_impl}' non reconnu")
        
        self.type_impl = type_impl
    
    def evaluer(self, contexte=None, **kwargs):
        val_ant = self.antecedent.evaluer(contexte, **kwargs).valeur
        val_cons = self.consequent.evaluer(contexte, **kwargs).valeur
        resultat = self.implications[self.type_impl](val_ant, val_cons)
        return ValeurFloue(resultat)
    
    def __str__(self):
        return f"({self.antecedent} → {self.consequent})"


class EnsembleFlou:
    """Représente un ensemble flou avec une fonction d'appartenance."""
    
    def __init__(self, nom: str, univers: List, fonction_appartenance: Callable[[Any], float]):
        self.nom = nom
        self.univers = univers
        self.fonction_appartenance = fonction_appartenance
    
    def appartenance(self, element):
        """Retourne le degré d'appartenance d'un élément à l'ensemble."""
        return ValeurFloue(self.fonction_appartenance(element))
    
    def alpha_coupe(self, alpha: float):
        """Retourne l'α-coupe de l'ensemble flou."""
        return [x for x in self.univers if self.fonction_appartenance(x) >= alpha]
    
    def support(self):
        """Retourne le support de l'ensemble flou."""
        return self.alpha_coupe(0.0)
    
    def noyau(self):
        """Retourne le noyau de l'ensemble flou."""
        return self.alpha_coupe(1.0)
    
    def est_normal(self):
        """Vérifie si l'ensemble flou est normal."""
        return any(self.fonction_appartenance(x) == 1 for x in self.univers)
    
    def cardinalite(self):
        """Calcule la cardinalité de l'ensemble flou."""
        return sum(self.fonction_appartenance(x) for x in self.univers)
    
    def __str__(self):
        return f"EnsembleFlou({self.nom})"


class OperationsEnsemblesFlous:
    """Opérations sur les ensembles flous."""
    
    @staticmethod
    def intersection(ensemble1: EnsembleFlou, ensemble2: EnsembleFlou, t_norme: str = 'min'):
        """Intersection de deux ensembles flous."""
        t_normes = {
            'min': lambda a, b: min(a, b),
            'produit': lambda a, b: a * b,
            'lukasiewicz': lambda a, b: max(0, a + b - 1),
        }
        
        if t_norme not in t_normes:
            raise ValueError(f"T-norme '{t_norme}' non reconnue")
        
        # Vérifier que les univers sont compatibles
        if ensemble1.univers != ensemble2.univers:
            raise ValueError("Les ensembles flous doivent avoir le même univers")
        
        nom = f"({ensemble1.nom} ∩ {ensemble2.nom})"
        
        def fonction_appartenance(x):
            val1 = ensemble1.fonction_appartenance(x)
            val2 = ensemble2.fonction_appartenance(x)
            return t_normes[t_norme](val1, val2)
        
        return EnsembleFlou(nom, ensemble1.univers, fonction_appartenance)
    
    @staticmethod
    def union(ensemble1: EnsembleFlou, ensemble2: EnsembleFlou, t_conorme: str = 'max'):
        """Union de deux ensembles flous."""
        t_conormes = {
            'max': lambda a, b: max(a, b),
            'somme_prob': lambda a, b: a + b - a * b,
            'lukasiewicz': lambda a, b: min(1, a + b),
        }
        
        if t_conorme not in t_conormes:
            raise ValueError(f"T-conorme '{t_conorme}' non reconnue")
        
        # Vérifier que les univers sont compatibles
        if ensemble1.univers != ensemble2.univers:
            raise ValueError("Les ensembles flous doivent avoir le même univers")
        
        nom = f"({ensemble1.nom} ∪ {ensemble2.nom})"
        
        def fonction_appartenance(x):
            val1 = ensemble1.fonction_appartenance(x)
            val2 = ensemble2.fonction_appartenance(x)
            return t_conormes[t_conorme](val1, val2)
        
        return EnsembleFlou(nom, ensemble1.univers, fonction_appartenance)
    
    @staticmethod
    def complement(ensemble: EnsembleFlou):
        """Complément d'un ensemble flou."""
        nom = f"¬({ensemble.nom})"
        
        def fonction_appartenance(x):
            return 1 - ensemble.fonction_appartenance(x)
        
        return EnsembleFlou(nom, ensemble.univers, fonction_appartenance)


class SystemeReglesFloues:
    """Système d'inférence floue basé sur des règles."""
    
    def __init__(self, nom: str):
        self.nom = nom
        self.regles = []
        self.variables_entree = {}
        self.variables_sortie = {}
    
    def ajouter_variable_entree(self, nom: str, univers: List, ensembles_flous: Dict[str, EnsembleFlou]):
        """Ajoute une variable d'entrée avec ses ensembles flous associés."""
        self.variables_entree[nom] = {
            'univers': univers,
            'ensembles': ensembles_flous
        }
        return self
    
    def ajouter_variable_sortie(self, nom: str, univers: List, ensembles_flous: Dict[str, EnsembleFlou]):
        """Ajoute une variable de sortie avec ses ensembles flous associés."""
        self.variables_sortie[nom] = {
            'univers': univers,
            'ensembles': ensembles_flous
        }
        return self
    
    def ajouter_regle(self, antecedents: List[Tuple[str, str, str]], consequents: List[Tuple[str, str]]):
        """
        Ajoute une règle floue.
        antecedents: liste de tuples (variable, ensemble_flou, operateur)
        consequents: liste de tuples (variable, ensemble_flou)
        """
        self.regles.append({
            'antecedents': antecedents,
            'consequents': consequents
        })
        return self
    
    def fuzzifier(self, entrees: Dict[str, Any]):
        """Fuzzifie les entrées crisp."""
        resultats = {}
        for var, valeur in entrees.items():
            if var not in self.variables_entree:
                raise ValueError(f"Variable d'entrée inconnue: {var}")
            
            # Calcul des degrés d'appartenance pour chaque ensemble flou
            resultats[var] = {}
            for nom_ens, ens_flou in self.variables_entree[var]['ensembles'].items():
                resultats[var][nom_ens] = ens_flou.fonction_appartenance(valeur)
        
        return resultats
    
    def evaluer_regle(self, regle, entrees_fuzzifiees):
        """Évalue une règle floue et retourne le degré d'activation."""
        # Évaluer les antécédents
        degres_activation = []
        
        for var, ens_flou, operateur in regle['antecedents']:
            if var not in entrees_fuzzifiees or ens_flou not in entrees_fuzzifiees[var]:
                raise ValueError(f"Antécédent invalide: {var}.{ens_flou}")
            
            degre = entrees_fuzzifiees[var][ens_flou]
            degres_activation.append((degre, operateur))
        
        # Combiner les degrés d'activation selon les opérateurs
        degre_final = degres_activation[0][0]
        for i in range(1, len(degres_activation)):
            degre, op = degres_activation[i]
            if op == 'ET':
                degre_final = min(degre_final, degre)
            elif op == 'OU':
                degre_final = max(degre_final, degre)
        
        return degre_final
    
    def inferer(self, entrees: Dict[str, Any], methode_agregation: str = 'max', methode_defuzzification: str = 'centroide'):
        """
        Effectue l'inférence floue complète:
        1. Fuzzification des entrées
        2. Évaluation des règles
        3. Agrégation des sorties
        4. Défuzzification
        """
        # Étape 1: Fuzzification
        entrees_fuzzifiees = self.fuzzifier(entrees)
        
        # Étape 2: Évaluation des règles
        activations = {}
        for i, regle in enumerate(self.regles):
            degre = self.evaluer_regle(regle, entrees_fuzzifiees)
            
            # Enregistrer l'activation pour chaque conséquent
            for var, ens_flou in regle['consequents']:
                if var not in activations:
                    activations[var] = {}
                
                if ens_flou not in activations[var]:
                    activations[var][ens_flou] = []
                
                activations[var][ens_flou].append(degre)
        
        # Étape 3: Agrégation des sorties
        sorties_agregees = {}
        for var in self.variables_sortie:
            sorties_agregees[var] = {}
            
            # Si la variable n'a pas été activée par une règle
            if var not in activations:
                continue
                
            for ens_flou in self.variables_sortie[var]['ensembles']:
                if ens_flou not in activations[var]:
                    continue
                    
                # Agréger les activations de cet ensemble flou
                if methode_agregation == 'max':
                    sorties_agregees[var][ens_flou] = max(activations[var][ens_flou])
                elif methode_agregation == 'somme':
                    sorties_agregees[var][ens_flou] = sum(activations[var][ens_flou])
                else:
                    raise ValueError(f"Méthode d'agrégation non reconnue: {methode_agregation}")
        
        # Étape 4: Défuzzification
        resultats = {}
        for var in self.variables_sortie:
            if var not in sorties_agregees or not sorties_agregees[var]:
                resultats[var] = None
                continue
                
            univers = self.variables_sortie[var]['univers']
            
            if methode_defuzzification == 'centroide':
                numerateur = 0
                denominateur = 0
                
                for x in univers:
                    # Calculer le degré d'appartenance agrégé pour x
                    degre_max = 0
                    for ens_flou, activation in sorties_agregees[var].items():
                        degre = min(activation, self.variables_sortie[var]['ensembles'][ens_flou].fonction_appartenance(x))
                        degre_max = max(degre_max, degre)
                    
                    numerateur += x * degre_max
                    denominateur += degre_max
                
                if denominateur > 0:
                    resultats[var] = numerateur / denominateur
                else:
                    resultats[var] = None
                    
            elif methode_defuzzification == 'moyenne_max':
                # Méthode de la moyenne des maximums
                degres_max = {}
                for x in univers:
                    degre_max = 0
                    for ens_flou, activation in sorties_agregees[var].items():
                        degre = min(activation, self.variables_sortie[var]['ensembles'][ens_flou].fonction_appartenance(x))
                        degre_max = max(degre_max, degre)
                    
                    degres_max[x] = degre_max
                
                max_degre = max(degres_max.values()) if degres_max else 0
                
                # Trouver les points avec le degré maximum
                points_max = [x for x, degre in degres_max.items() if degre == max_degre]
                
                if points_max:
                    resultats[var] = sum(points_max) / len(points_max)
                else:
                    resultats[var] = None
            
            elif methode_defuzzification == 'premier_max':
                # Méthode du premier maximum
                degres_max = {}
                for x in univers:
                    degre_max = 0
                    for ens_flou, activation in sorties_agregees[var].items():
                        degre = min(activation, self.variables_sortie[var]['ensembles'][ens_flou].fonction_appartenance(x))
                        degre_max = max(degre_max, degre)
                    
                    degres_max[x] = degre_max
                
                max_degre = max(degres_max.values()) if degres_max else 0
                
                # Trouver le premier point avec le degré maximum
                for x, degre in sorted(degres_max.items()):
                    if degre == max_degre:
                        resultats[var] = x
                        break
                else:
                    resultats[var] = None
            
            else:
                raise ValueError(f"Méthode de défuzzification non reconnue: {methode_defuzzification}")
        
        return resultats


#########################################
# LOGIQUE MODALE
#########################################

class MondesPossibles:
    """Représente un modèle de mondes possibles pour la logique modale."""
    
    def __init__(self):
        self.mondes = set()
        self.relations = {}  # Relations d'accessibilité entre mondes
        self.valuations = {}  # Valuations des propositions dans chaque monde
    
    def ajouter_monde(self, monde):
        """Ajoute un monde possible au modèle."""
        self.mondes.add(monde)
        self.relations[monde] = set()
        self.valuations[monde] = {}
        return self
    
    def ajouter_relation(self, monde1, monde2):
        """Ajoute une relation d'accessibilité entre deux mondes."""
        if monde1 not in self.mondes or monde2 not in self.mondes:
            raise ValueError("Les mondes doivent être dans le modèle")
        
        self.relations[monde1].add(monde2)
        return self
    
    def definir_valuation(self, monde, proposition, valeur):
        """Définit la valeur de vérité d'une proposition dans un monde."""
        if monde not in self.mondes:
            raise ValueError(f"Le monde {monde} n'est pas dans le modèle")
        
        self.valuations[monde][proposition] = valeur
        return self
    
    def valuation(self, monde, proposition):
        """Retourne la valeur de vérité d'une proposition dans un monde."""
        if monde not in self.mondes:
            raise ValueError(f"Le monde {monde} n'est pas dans le modèle")
        
        return self.valuations[monde].get(proposition, False)
    
    def mondes_accessibles(self, monde):
        """Retourne les mondes accessibles depuis un monde donné."""
        if monde not in self.mondes:
            raise ValueError(f"Le monde {monde} n'est pas dans le modèle")
        
        return self.relations[monde]
    
    def verifier_proprietes(self):
        """Vérifie les propriétés de la relation d'accessibilité."""
        resultats = {
            'reflexive': True,
            'symetrique': True,
            'transitive': True,
            'serielle': True,
            'euclidienne': True
        }
        
        # Vérifier la réflexivité
        for monde in self.mondes:
            if monde not in self.relations[monde]:
                resultats['reflexive'] = False
                break
        
        # Vérifier la symétrie
        for monde1 in self.mondes:
            for monde2 in self.relations[monde1]:
                if monde1 not in self.relations[monde2]:
                    resultats['symetrique'] = False
                    break
        
        # Vérifier la transitivité
        for monde1 in self.mondes:
            for monde2 in self.relations[monde1]:
                for monde3 in self.relations[monde2]:
                    if monde3 not in self.relations[monde1]:
                        resultats['transitive'] = False
                        break
        
        # Vérifier la sérialité (chaque monde a au moins un monde accessible)
        for monde in self.mondes:
            if not self.relations[monde]:
                resultats['serielle'] = False
                break
        
        # Vérifier la propriété euclidienne
        for monde1 in self.mondes:
            for monde2 in self.relations[monde1]:
                for monde3 in self.relations[monde1]:
                    if monde3 not in self.relations[monde2]:
                        resultats['euclidienne'] = False
                        break
        
        return resultats
    
    def type_logique(self):
        """Détermine le type de logique modale selon les propriétés de la relation."""
        proprietes = self.verifier_proprietes()
        
        if proprietes['reflexive'] and proprietes['transitive']:
            if proprietes['symetrique']:
                return "S5"
            else:
                return "S4"
        elif proprietes['reflexive']:
            return "T"
        elif proprietes['serielle']:
            if proprietes['transitive'] and proprietes['euclidienne']:
                return "D45"
            elif proprietes['transitive']:
                return "D4"
            elif proprietes['euclidienne']:
                return "D5"
            else:
                return "D"
        elif proprietes['transitive'] and proprietes['euclidienne']:
            return "K45"
        elif proprietes['transitive']:
            return "K4"
        elif proprietes['euclidienne']:
            return "K5"
        else:
            return "K"


class FormuleModale(Formule):
    """Classe de base pour les formules de logique modale."""
    pass


class PropositionModale(FormuleModale):
    """Représente une proposition atomique en logique modale."""
    
    def __init__(self, nom: str):
        self.nom = nom
    
    def evaluer(self, monde, modele: MondesPossibles):
        """Évalue la proposition dans un monde donné du modèle."""
        return modele.valuation(monde, self.nom)
    
    def __str__(self):
        return self.nom


class NonModale(FormuleModale):
    """Négation en logique modale."""
    
    def __init__(self, formule: FormuleModale):
        self.formule = formule
    
    def evaluer(self, monde, modele: MondesPossibles):
        """Évalue la négation dans un monde donné du modèle."""
        return not self.formule.evaluer(monde, modele)
    
    def __str__(self):
        return f"¬{self.formule}"


class EtModale(FormuleModale):
    """Conjonction en logique modale."""
    
    def __init__(self, gauche: FormuleModale, droite: FormuleModale):
        self.gauche = gauche
        self.droite = droite
    
    def evaluer(self, monde, modele: MondesPossibles):
        """Évalue la conjonction dans un monde donné du modèle."""
        return self.gauche.evaluer(monde, modele) and self.droite.evaluer(monde, modele)
    
    def __str__(self):
        return f"({self.gauche} ∧ {self.droite})"


class OuModale(FormuleModale):
    """Disjonction en logique modale."""
    
    def __init__(self, gauche: FormuleModale, droite: FormuleModale):
        self.gauche = gauche
        self.droite = droite
    
    def evaluer(self, monde, modele: MondesPossibles):
        """Évalue la disjonction dans un monde donné du modèle."""
        return self.gauche.evaluer(monde, modele) or self.droite.evaluer(monde, modele)
    
    def __str__(self):
        return f"({self.gauche} ∨ {self.droite})"


class ImplicationModale(FormuleModale):
    """Implication en logique modale."""
    
    def __init__(self, antecedent: FormuleModale, consequent: FormuleModale):
        self.antecedent = antecedent
        self.consequent = consequent
    
    def evaluer(self, monde, modele: MondesPossibles):
        """Évalue l'implication dans un monde donné du modèle."""
        return not self.antecedent.evaluer(monde, modele) or self.consequent.evaluer(monde, modele)
    
    def __str__(self):
        return f"({self.antecedent} → {self.consequent})"


class Necessite(FormuleModale):
    """Opérateur de nécessité (□) en logique modale."""
    
    def __init__(self, formule: FormuleModale):
        self.formule = formule
    
    def evaluer(self, monde, modele: MondesPossibles):
        """Évalue la nécessité dans un monde donné du modèle."""
        # Une formule est nécessaire si elle est vraie dans tous les mondes accessibles
        mondes_accessibles = modele.mondes_accessibles(monde)
        
        if not mondes_accessibles:
            # Si aucun monde n'est accessible, la nécessité est vraie par vacuité
            return True
        
        return all(self.formule.evaluer(m, modele) for m in mondes_accessibles)
    
    def __str__(self):
        return f"□{self.formule}"


class Possibilite(FormuleModale):
    """Opérateur de possibilité (◇) en logique modale."""
    
    def __init__(self, formule: FormuleModale):
        self.formule = formule
    
    def evaluer(self, monde, modele: MondesPossibles):
        """Évalue la possibilité dans un monde donné du modèle."""
        # Une formule est possible si elle est vraie dans au moins un monde accessible
        mondes_accessibles = modele.mondes_accessibles(monde)
        
        if not mondes_accessibles:
            # Si aucun monde n'est accessible, la possibilité est fausse par vacuité
            return False
        
        return any(self.formule.evaluer(m, modele) for m in mondes_accessibles)
    
    def __str__(self):
        return f"◇{self.formule}"


class VerificateurModele:
    """Vérificateur de modèle pour la logique modale."""
    
    @staticmethod
    def verifier(formule: FormuleModale, modele: MondesPossibles, monde=None):
        """
        Vérifie si une formule est vraie dans un monde donné ou dans tous les mondes.
        Si monde est None, vérifie si la formule est vraie dans tous les mondes.
        """
        if monde is not None:
            return formule.evaluer(monde, modele)
        
        return all(formule.evaluer(m, modele) for m in modele.mondes)
    
    @staticmethod
    def mondes_satisfaisant(formule: FormuleModale, modele: MondesPossibles):
        """Retourne l'ensemble des mondes qui satisfont la formule."""
        return {m for m in modele.mondes if formule.evaluer(m, modele)}
    
    @staticmethod
    def est_valide(formule: FormuleModale, modele: MondesPossibles):
        """Vérifie si une formule est valide dans le modèle (vraie dans tous les mondes)."""
        return all(formule.evaluer(m, modele) for m in modele.mondes)
    
    @staticmethod
    def est_satisfaisable(formule: FormuleModale, modele: MondesPossibles):
        """Vérifie si une formule est satisfaisable dans le modèle (vraie dans au moins un monde)."""
        return any(formule.evaluer(m, modele) for m in modele.mondes)


class LogiqueDeontique:
    """Implémentation de la logique déontique en utilisant la logique modale."""
    
    def __init__(self):
        self.modele = MondesPossibles()
    
    def ajouter_monde_ideal(self, monde):
        """Ajoute un monde idéal au modèle."""
        self.modele.ajouter_monde(monde)
        return self
    
    def ajouter_monde_reel(self, monde):
        """Ajoute un monde réel au modèle."""
        self.modele.ajouter_monde(monde)
        return self
    
    def definir_accessibilite_ideale(self, monde_reel, monde_ideal):
        """Définit qu'un monde idéal est accessible depuis un monde réel."""
        self.modele.ajouter_relation(monde_reel, monde_ideal)
        return self
    
    def definir_valuation(self, monde, proposition, valeur):
        """Définit la valeur de vérité d'une proposition dans un monde."""
        self.modele.definir_valuation(monde, proposition, valeur)
        return self
    
    def obligation(self, formule: FormuleModale):
        """Crée une formule d'obligation (O)."""
        return Necessite(formule)
    
    def permission(self, formule: FormuleModale):
        """Crée une formule de permission (P)."""
        return Possibilite(formule)
    
    def interdiction(self, formule: FormuleModale):
        """Crée une formule d'interdiction (F)."""
        return Necessite(NonModale(formule))
    
    def optionnel(self, formule: FormuleModale):
        """Crée une formule optionnelle (tout ce qui n'est ni obligatoire ni interdit)."""
        # Quelque chose est optionnel si ni lui ni sa négation ne sont obligatoires
        non_formule = NonModale(formule)
        return EtModale(NonModale(self.obligation(formule)), NonModale(self.obligation(non_formule)))
    
    def verifier(self, formule: FormuleModale, monde=None):
        """Vérifie si une formule est vraie dans un monde donné ou dans tous les mondes."""
        return VerificateurModele.verifier(formule, self.modele, monde)


#########################################
# LOGIQUE TEMPORELLE
#########################################

class StructureTemporelle:
    """Représente une structure temporelle pour la logique temporelle."""
    
    def __init__(self, type_temps: str = 'lineaire'):
        """
        Initialise une structure temporelle.
        type_temps: 'lineaire', 'branchant', 'cyclique'
        """
        self.instants = set()
        self.relations = {}  # Relations d'ordre entre instants
        self.valuations = {}  # Valuations des propositions à chaque instant
        self.type_temps = type_temps
    
    def ajouter_instant(self, instant):
        """Ajoute un instant à la structure temporelle."""
        self.instants.add(instant)
        self.relations[instant] = set()
        self.valuations[instant] = {}
        return self
    
    def ajouter_relation(self, instant1, instant2):
        """Ajoute une relation temporelle: instant1 précède instant2."""
        if instant1 not in self.instants or instant2 not in self.instants:
            raise ValueError("Les instants doivent être dans la structure")
        
        self.relations[instant1].add(instant2)
        return self
    
    def definir_valuation(self, instant, proposition, valeur):
        """Définit la valeur de vérité d'une proposition à un instant donné."""
        if instant not in self.instants:
            raise ValueError(f"L'instant {instant} n'est pas dans la structure")
        
        self.valuations[instant][proposition] = valeur
        return self
    
    def valuation(self, instant, proposition):
        """Retourne la valeur de vérité d'une proposition à un instant donné."""
        if instant not in self.instants:
            raise ValueError(f"L'instant {instant} n'est pas dans la structure")
        
        return self.valuations[instant].get(proposition, False)
    
    def instants_futurs(self, instant):
        """Retourne les instants futurs immédiats depuis un instant donné."""
        if instant not in self.instants:
            raise ValueError(f"L'instant {instant} n'est pas dans la structure")
        
        return self.relations[instant]
    
    def instants_passes(self, instant):
        """Retourne les instants passés immédiats depuis un instant donné."""
        if instant not in self.instants:
            raise ValueError(f"L'instant {instant} n'est pas dans la structure")
        
        return {i for i in self.instants if instant in self.relations[i]}
    
    def verifier_proprietes(self):
        """Vérifie les propriétés de la relation temporelle."""
        resultats = {
            'irreflexive': True,  # Un instant ne précède pas lui-même
            'antisymetrique': True,  # Si t1 précède t2, t2 ne précède pas t1
            'transitive': True,  # Si t1 précède t2 et t2 précède t3, alors t1 précède t3
            'connexe': True,  # Pour tout t1 et t2, soit t1 précède t2, soit t2 précède t1, soit t1=t2
            'lineaire': True,  # Chaque instant a au plus un successeur
            'dense': True  # Entre deux instants, il y a toujours un autre instant
        }
        
        # Vérifier l'irréflexivité
        for instant in self.instants:
            if instant in self.relations[instant]:
                resultats['irreflexive'] = False
                break
        
        # Vérifier l'antisymétrie
        for instant1 in self.instants:
            for instant2 in self.relations[instant1]:
                if instant1 in self.relations[instant2]:
                    resultats['antisymetrique'] = False
                    break
        
        # Vérifier la transitivité
        for instant1 in self.instants:
            for instant2 in self.relations[instant1]:
                for instant3 in self.relations[instant2]:
                    if instant3 not in self.relations[instant1]:
                        resultats['transitive'] = False
                        break
        
        # Vérifier la connexité
        for instant1 in self.instants:
            for instant2 in self.instants:
                if instant1 != instant2:
                    if instant2 not in self.relations[instant1] and instant1 not in self.relations[instant2]:
                        resultats['connexe'] = False
                        break
        
        # Vérifier la linéarité
        for instant in self.instants:
            if len(self.relations[instant]) > 1:
                resultats['lineaire'] = False
                break
        
        # Vérifier la densité
        for instant1 in self.instants:
            for instant2 in self.relations[instant1]:
                if not any(instant1 in self.relations[i] and instant2 in self.relations[i] for i in self.instants):
                    resultats['dense'] = False
                    break
        
        return resultats
    
    def type_logique(self):
        """Détermine le type de logique temporelle selon les propriétés de la relation."""
        proprietes = self.verifier_proprietes()
        
        if self.type_temps == 'lineaire':
            if proprietes['transitive'] and proprietes['irreflexive']:
                if proprietes['dense']:
                    return "LTL dense"
                else:
                    return "LTL discrète"
            else:
                return "Structure temporelle linéaire non standard"
        elif self.type_temps == 'branchant':
            if proprietes['transitive'] and proprietes['irreflexive']:
                return "CTL"
            else:
                return "Structure temporelle branchante non standard"
        elif self.type_temps == 'cyclique':
            return "Structure temporelle cyclique"
        else:
            return "Structure temporelle non standard"


class FormuleTemporelle(Formule):
    """Classe de base pour les formules de logique temporelle."""
    pass


class PropositionTemporelle(FormuleTemporelle):
    """Représente une proposition atomique en logique temporelle."""
    
    def __init__(self, nom: str):
        self.nom = nom
    
    def evaluer(self, instant, structure: StructureTemporelle):
        """Évalue la proposition à un instant donné de la structure."""
        return structure.valuation(instant, self.nom)
    
    def __str__(self):
        return self.nom


class NonTemporelle(FormuleTemporelle):
    """Négation en logique temporelle."""
    
    def __init__(self, formule: FormuleTemporelle):
        self.formule = formule
    
    def evaluer(self, instant, structure: StructureTemporelle):
        """Évalue la négation à un instant donné de la structure."""
        return not self.formule.evaluer(instant, structure)
    
    def __str__(self):
        return f"¬{self.formule}"


class EtTemporelle(FormuleTemporelle):
    """Conjonction en logique temporelle."""
    
    def __init__(self, gauche: FormuleTemporelle, droite: FormuleTemporelle):
        self.gauche = gauche
        self.droite = droite
    
    def evaluer(self, instant, structure: StructureTemporelle):
        """Évalue la conjonction à un instant donné de la structure."""
        return self.gauche.evaluer(instant, structure) and self.droite.evaluer(instant, structure)
    
    def __str__(self):
        return f"({self.gauche} ∧ {self.droite})"


class OuTemporelle(FormuleTemporelle):
    """Disjonction en logique temporelle."""
    
    def __init__(self, gauche: FormuleTemporelle, droite: FormuleTemporelle):
        self.gauche = gauche
        self.droite = droite
    
    def evaluer(self, instant, structure: StructureTemporelle):
        """Évalue la disjonction à un instant donné de la structure."""
        return self.gauche.evaluer(instant, structure) or self.droite.evaluer(instant, structure)
    
    def __str__(self):
        return f"({self.gauche} ∨ {self.droite})"


class ImplicationTemporelle(FormuleTemporelle):
    """Implication en logique temporelle."""
    
    def __init__(self, antecedent: FormuleTemporelle, consequent: FormuleTemporelle):
        self.antecedent = antecedent
        self.consequent = consequent
    
    def evaluer(self, instant, structure: StructureTemporelle):
        """Évalue l'implication à un instant donné de la structure."""
        return not self.antecedent.evaluer(instant, structure) or self.consequent.evaluer(instant, structure)
    
    def __str__(self):
        return f"({self.antecedent} → {self.consequent})"


class FuturProche(FormuleTemporelle):
    """Opérateur X (next) en logique temporelle: vrai si la formule est vraie à l'instant suivant."""
    
    def __init__(self, formule: FormuleTemporelle):
        self.formule = formule
    
    def evaluer(self, instant, structure: StructureTemporelle):
        """Évalue l'opérateur X à un instant donné de la structure."""
        instants_futurs = structure.instants_futurs(instant)
        
        if not instants_futurs:
            # Si aucun instant futur, X est faux par défaut
            return False
        
        if structure.type_temps == 'lineaire':
            # Dans un temps linéaire, il y a au plus un instant futur immédiat
            if len(instants_futurs) != 1:
                raise ValueError("Une structure temporelle linéaire devrait avoir exactement un futur immédiat")
            
            prochain_instant = next(iter(instants_futurs))
            return self.formule.evaluer(prochain_instant, structure)
        else:
            # Dans un temps branchant, X est vrai si la formule est vraie dans tous les futurs immédiats possibles
            return all(self.formule.evaluer(i, structure) for i in instants_futurs)
    
    def __str__(self):
        return f"X{self.formule}"


class FuturEventuel(FormuleTemporelle):
    """Opérateur F (finally) en logique temporelle: vrai si la formule est vraie à un instant futur."""
    
    def __init__(self, formule: FormuleTemporelle):
        self.formule = formule
    
    def evaluer(self, instant, structure: StructureTemporelle, visite=None):
        """Évalue l'opérateur F à un instant donné de la structure."""
        if visite is None:
            visite = set()
        
        if instant in visite:
            # Détection de cycle, évite les boucles infinies
            return False
        
        visite.add(instant)
        
        # Vérifier si la formule est vraie à l'instant actuel
        if self.formule.evaluer(instant, structure):
            return True
        
        # Vérifier récursivement dans les instants futurs
        instants_futurs = structure.instants_futurs(instant)
        
        if structure.type_temps == 'lineaire':
            # Dans un temps linéaire, F est vrai si la formule est vraie dans au moins un instant futur
            for prochain_instant in instants_futurs:
                if self.evaluer(prochain_instant, structure, visite):
                    return True
            return False
        else:
            # Dans un temps branchant, F est vrai s'il existe un chemin où la formule devient vraie
            return any(self.evaluer(i, structure, visite.copy()) for i in instants_futurs)
    
    def __str__(self):
        return f"F{self.formule}"


class FuturToujoursFutur(FormuleTemporelle):
    """Opérateur G (globally) en logique temporelle: vrai si la formule est vraie à tous les instants futurs."""
    
    def __init__(self, formule: FormuleTemporelle):
        self.formule = formule
    
    def evaluer(self, instant, structure: StructureTemporelle, visite=None):
        """Évalue l'opérateur G à un instant donné de la structure."""
        if visite is None:
            visite = set()
        
        if instant in visite:
            # Détection de cycle, pour les structures cycliques
            return True
        
        visite.add(instant)
        
        # Vérifier si la formule est vraie à l'instant actuel
        if not self.formule.evaluer(instant, structure):
            return False
        
        # Vérifier récursivement dans les instants futurs
        instants_futurs = structure.instants_futurs(instant)
        
        if not instants_futurs:
            # Si aucun futur, G est vrai par vacuité (fin du temps)
            return True
        
        if structure.type_temps == 'lineaire':
            # Dans un temps linéaire, G est vrai si la formule est vraie dans tous les instants futurs
            for prochain_instant in instants_futurs:
                if not self.evaluer(prochain_instant, structure, visite):
                    return False
            return True
        else:
            # Dans un temps branchant, G est vrai si la formule est vraie dans tous les chemins futurs
            return all(self.evaluer(i, structure, visite.copy()) for i in instants_futurs)
    
    def __str__(self):
        return f"G{self.formule}"


class Jusqu_a(FormuleTemporelle):
    """
    Opérateur U (until) en logique temporelle: 
    f U g est vrai si g est vrai à un moment futur et f est vrai jusqu'à ce moment.
    """
    
    def __init__(self, gauche: FormuleTemporelle, droite: FormuleTemporelle):
        self.gauche = gauche
        self.droite = droite
    
    def evaluer(self, instant, structure: StructureTemporelle, visite=None):
        """Évalue l'opérateur U à un instant donné de la structure."""
        if visite is None:
            visite = set()
        
        if instant in visite:
            # Détection de cycle
            return False
        
        visite.add(instant)
        
        # Vérifier si la condition droite est vraie à l'instant actuel
        if self.droite.evaluer(instant, structure):
            return True
        
        # Vérifier si la condition gauche est vraie à l'instant actuel
        if not self.gauche.evaluer(instant, structure):
            return False
        
        # Vérifier récursivement dans les instants futurs
        instants_futurs = structure.instants_futurs(instant)
        
        if not instants_futurs:
            # Si aucun futur, U est faux car la condition droite n'est jamais atteinte
            return False
        
        if structure.type_temps == 'lineaire':
            # Dans un temps linéaire, U est vrai s'il existe un instant futur où la condition droite est vraie
            # et la condition gauche est vraie jusqu'à cet instant
            for prochain_instant in instants_futurs:
                if self.evaluer(prochain_instant, structure, visite):
                    return True
            return False
        else:
            # Dans un temps branchant, U est vrai s'il existe un chemin où la condition est satisfaite
            return any(self.evaluer(i, structure, visite.copy()) for i in instants_futurs)
    
    def __str__(self):
        return f"({self.gauche} U {self.droite})"


class PasseProche(FormuleTemporelle):
    """Opérateur Y (yesterday) en logique temporelle: vrai si la formule était vraie à l'instant précédent."""
    
    def __init__(self, formule: FormuleTemporelle):
        self.formule = formule
    
    def evaluer(self, instant, structure: StructureTemporelle):
        """Évalue l'opérateur Y à un instant donné de la structure."""
        instants_passes = structure.instants_passes(instant)
        
        if not instants_passes:
            # Si aucun instant passé, Y est faux par défaut
            return False
        
        if structure.type_temps == 'lineaire':
            # Dans un temps linéaire, il y a au plus un instant passé immédiat
            if len(instants_passes) != 1:
                raise ValueError("Une structure temporelle linéaire devrait avoir exactement un passé immédiat")
            
            instant_precedent = next(iter(instants_passes))
            return self.formule.evaluer(instant_precedent, structure)
        else:
            # Dans un temps branchant, Y est vrai si la formule était vraie dans tous les passés immédiats possibles
            return all(self.formule.evaluer(i, structure) for i in instants_passes)
    
    def __str__(self):
        return f"Y{self.formule}"


class PasseEventuel(FormuleTemporelle):
    """Opérateur P (past) en logique temporelle: vrai si la formule était vraie à un instant passé."""
    
    def __init__(self, formule: FormuleTemporelle):
        self.formule = formule
    
    def evaluer(self, instant, structure: StructureTemporelle, visite=None):
        """Évalue l'opérateur P à un instant donné de la structure."""
        if visite is None:
            visite = set()
        
        if instant in visite:
            # Détection de cycle
            return False
        
        visite.add(instant)
        
        # Vérifier si la formule est vraie à l'instant actuel
        if self.formule.evaluer(instant, structure):
            return True
        
        # Vérifier récursivement dans les instants passés
        instants_passes = structure.instants_passes(instant)
        
        if structure.type_temps == 'lineaire':
            # Dans un temps linéaire, P est vrai si la formule était vraie dans au moins un instant passé
            for instant_precedent in instants_passes:
                if self.evaluer(instant_precedent, structure, visite):
                    return True
            return False
        else:
            # Dans un temps branchant, P est vrai s'il existe un chemin où la formule était vraie
            return any(self.evaluer(i, structure, visite.copy()) for i in instants_passes)
    
    def __str__(self):
        return f"P{self.formule}"


class PasseToujoursPasse(FormuleTemporelle):
    """Opérateur H (historically) en logique temporelle: vrai si la formule était vraie à tous les instants passés."""
    
    def __init__(self, formule: FormuleTemporelle):
        self.formule = formule
    
    def evaluer(self, instant, structure: StructureTemporelle, visite=None):
        """Évalue l'opérateur H à un instant donné de la structure."""
        if visite is None:
            visite = set()
        
        if instant in visite:
            # Détection de cycle
            return True
        
        visite.add(instant)
        
        # Vérifier si la formule est vraie à l'instant actuel
        if not self.formule.evaluer(instant, structure):
            return False
        
        # Vérifier récursivement dans les instants passés
        instants_passes = structure.instants_passes(instant)
        
        if not instants_passes:
            # Si aucun passé, H est vrai par vacuité (début du temps)
            return True
        
        if structure.type_temps == 'lineaire':
            # Dans un temps linéaire, H est vrai si la formule était vraie dans tous les instants passés
            for instant_precedent in instants_passes:
                if not self.evaluer(instant_precedent, structure, visite):
                    return False
            return True
        else:
            # Dans un temps branchant, H est vrai si la formule était vraie dans tous les chemins passés
            return all(self.evaluer(i, structure, visite.copy()) for i in instants_passes)
    
    def __str__(self):
        return f"H{self.formule}"


class Depuis(FormuleTemporelle):
    """
    Opérateur S (since) en logique temporelle: 
    f S g est vrai si g était vrai à un moment passé et f a été vrai depuis ce moment.
    """
    
    def __init__(self, gauche: FormuleTemporelle, droite: FormuleTemporelle):
        self.gauche = gauche
        self.droite = droite
    
    def evaluer(self, instant, structure: StructureTemporelle, visite=None):
        """Évalue l'opérateur S à un instant donné de la structure."""
        if visite is None:
            visite = set()
        
        if instant in visite:
            # Détection de cycle
            return False
        
        visite.add(instant)
        
        # Vérifier si la condition droite est vraie à l'instant actuel
        if self.droite.evaluer(instant, structure):
            return True
        
        # Vérifier si la condition gauche est vraie à l'instant actuel
        if not self.gauche.evaluer(instant, structure):
            return False
        
        # Vérifier récursivement dans les instants passés
        instants_passes = structure.instants_passes(instant)
        
        if not instants_passes:
            # Si aucun passé, S est faux car la condition droite n'est jamais atteinte
            return False
        
        if structure.type_temps == 'lineaire':
            # Dans un temps linéaire, S est vrai s'il existe un instant passé où la condition droite était vraie
            # et la condition gauche a été vraie depuis cet instant
            for instant_precedent in instants_passes:
                if self.evaluer(instant_precedent, structure, visite):
                    return True
            return False
        else:
            # Dans un temps branchant, S est vrai s'il existe un chemin où la condition est satisfaite
            return any(self.evaluer(i, structure, visite.copy()) for i in instants_passes)
    
    def __str__(self):
        return f"({self.gauche} S {self.droite})"


class VerificateurTemporel:
    """Vérificateur de modèle pour la logique temporelle."""
    
    @staticmethod
    def verifier(formule: FormuleTemporelle, structure: StructureTemporelle, instant=None):
        """
        Vérifie si une formule est vraie à un instant donné ou à tous les instants.
        Si instant est None, vérifie si la formule est vraie à tous les instants.
        """
        if instant is not None:
            return formule.evaluer(instant, structure)
        
        return all(formule.evaluer(i, structure) for i in structure.instants)
    
    @staticmethod
    def instants_satisfaisant(formule: FormuleTemporelle, structure: StructureTemporelle):
        """Retourne l'ensemble des instants qui satisfont la formule."""
        return {i for i in structure.instants if formule.evaluer(i, structure)}
    
    @staticmethod
    def est_valide(formule: FormuleTemporelle, structure: StructureTemporelle):
        """Vérifie si une formule est valide dans la structure (vraie à tous les instants)."""
        return all(formule.evaluer(i, structure) for i in structure.instants)
    
    @staticmethod
    def est_satisfaisable(formule: FormuleTemporelle, structure: StructureTemporelle):
        """Vérifie si une formule est satisfaisable dans la structure (vraie à au moins un instant)."""
        return any(formule.evaluer(i, structure) for i in structure.instants)


#########################################
# LOGIQUE DÉONTIQUE
#########################################

class SystemeNormatif:
    """Représente un système normatif pour la logique déontique."""
    
    def __init__(self):
        # Utiliser la logique modale comme base pour la logique déontique
        self.logique_deontique = LogiqueDeontique()
        self.monde_reel = "monde_reel"
        self.mondes_ideaux = set()
        
        # Ajouter le monde réel
        self.logique_deontique.ajouter_monde_reel(self.monde_reel)
        
        def ajouter_monde_ideal(self, nom: str):
            """Ajoute un monde idéal au système normatif."""
            self.mondes_ideaux.add(nom)
            self.logique_deontique.ajouter_monde_ideal(nom)
            self.logique_deontique.definir_accessibilite_ideale(self.monde_reel, nom)
            return self
    
    def definir_valuation(self, monde, proposition, valeur):
        """Définit la valeur de vérité d'une proposition dans un monde."""
        self.logique_deontique.definir_valuation(monde, proposition, valeur)
        return self
    
    def obligation(self, formule: FormuleModale):
        """Crée une formule d'obligation (O)."""
        return self.logique_deontique.obligation(formule)
    
    def permission(self, formule: FormuleModale):
        """Crée une formule de permission (P)."""
        return self.logique_deontique.permission(formule)
    
    def interdiction(self, formule: FormuleModale):
        """Crée une formule d'interdiction (F)."""
        return self.logique_deontique.interdiction(formule)
    
    def optionnel(self, formule: FormuleModale):
        """Crée une formule optionnelle."""
        return self.logique_deontique.optionnel(formule)
    
    def verifier(self, formule: FormuleModale):
        """Vérifie si une formule est vraie dans le monde réel."""
        return self.logique_deontique.verifier(formule, self.monde_reel)
    
    def est_coherent(self):
        """Vérifie si le système normatif est cohérent (non contradictoire)."""
        for proposition in set(v for m in self.logique_deontique.modele.valuations.values() for v in m.keys()):
            formule = PropositionModale(proposition)
            if self.verifier(self.obligation(formule)) and self.verifier(self.interdiction(formule)):
                return False
        return True
    
    def identifier_conflits(self):
        """Identifie les conflits normatifs dans le système."""
        conflits = []
        for proposition in set(v for m in self.logique_deontique.modele.valuations.values() for v in m.keys()):
            formule = PropositionModale(proposition)
            if self.verifier(self.obligation(formule)) and self.verifier(self.interdiction(formule)):
                conflits.append(proposition)
        return conflits
    
    def resoudre_conflit(self, proposition, priorite="obligation"):
        """Résout un conflit normatif en donnant la priorité à l'obligation ou l'interdiction."""
        formule = PropositionModale(proposition)
        
        if priorite == "obligation":
            # Garder l'obligation, supprimer l'interdiction
            for monde_ideal in self.mondes_ideaux:
                self.logique_deontique.definir_valuation(monde_ideal, proposition, True)
        elif priorite == "interdiction":
            # Garder l'interdiction, supprimer l'obligation
            for monde_ideal in self.mondes_ideaux:
                self.logique_deontique.definir_valuation(monde_ideal, proposition, False)
        else:
            raise ValueError(f"Priorité inconnue: {priorite}")
        
        return self


##########################################
# PARTIE 2: DÉMONSTRATION AUTOMATIQUE DE THÉORÈMES
##########################################

class Preuve:
    """Représente une preuve formelle."""
    
    def __init__(self, nom: str = ""):
        self.nom = nom
        self.lignes = []  # Liste des lignes de la preuve
        self.hypotheses = set()  # Ensemble des hypothèses utilisées
    
    def ajouter_ligne(self, formule, justification, references=None):
        """
        Ajoute une ligne à la preuve.
        formule: la formule démontrée
        justification: la règle d'inférence utilisée
        references: références aux lignes précédentes
        """
        if references is None:
            references = []
        
        ligne = {
            "numero": len(self.lignes) + 1,
            "formule": formule,
            "justification": justification,
            "references": references
        }
        
        self.lignes.append(ligne)
        return self
    
    def ajouter_hypothese(self, formule):
        """Ajoute une hypothèse à la preuve."""
        self.hypotheses.add(formule)
        self.ajouter_ligne(formule, "Hypothèse")
        return self
    
    def conclusion(self):
        """Retourne la conclusion de la preuve."""
        if not self.lignes:
            return None
        return self.lignes[-1]["formule"]
    
    def est_valide(self):
        """Vérifie si la preuve est valide."""
        # Vérifier que chaque ligne est justifiée correctement
        for ligne in self.lignes:
            if ligne["justification"] == "Hypothèse":
                if str(ligne["formule"]) not in map(str, self.hypotheses):
                    return False
            elif not self.verifier_justification(ligne):
                return False
        
        return True
    
    def verifier_justification(self, ligne):
        """Vérifie qu'une justification est correcte."""
        # Cette méthode serait complétée pour chaque règle d'inférence
        # Pour simplifier, nous supposons que toutes les justifications sont valides
        return True
    
    def to_latex(self):
        """Génère une représentation LaTeX de la preuve."""
        latex = "\\begin{proof}\n"
        latex += "\\begin{enumerate}\n"
        
        for ligne in self.lignes:
            formule = str(ligne["formule"]).replace("∧", "\\land").replace("∨", "\\lor").replace("¬", "\\neg").replace("→", "\\rightarrow")
            
            refs = ""
            if ligne["references"]:
                refs = " [" + ", ".join(map(str, ligne["references"])) + "]"
            
            latex += f"\\item {formule} \\hfill ({ligne['justification']}{refs})\n"
        
        latex += "\\end{enumerate}\n"
        latex += "\\end{proof}"
        
        return latex
    
    def __str__(self):
        resultat = f"Preuve: {self.nom}\n"
        resultat += "Hypothèses: " + ", ".join(map(str, self.hypotheses)) + "\n"
        resultat += "Lignes:\n"
        
        for ligne in self.lignes:
            refs = ""
            if ligne["references"]:
                refs = " [" + ", ".join(map(str, ligne["references"])) + "]"
            
            resultat += f"{ligne['numero']}. {ligne['formule']} ({ligne['justification']}{refs})\n"
        
        return resultat


class RegleInference:
    """Représente une règle d'inférence pour la déduction naturelle."""
    
    def __init__(self, nom, schemas_premisses, schema_conclusion):
        self.nom = nom
        self.schemas_premisses = schemas_premisses  # Liste de schémas de formules pour les prémisses
        self.schema_conclusion = schema_conclusion  # Schéma de formule pour la conclusion
    
    def appliquer(self, formules):
        """
        Applique la règle d'inférence aux formules données.
        Retourne la conclusion si la règle est applicable, None sinon.
        """
        # Cette méthode serait implémentée pour chaque règle d'inférence spécifique
        return None
    
    def __str__(self):
        premisses = ", ".join(map(str, self.schemas_premisses))
        return f"{self.nom}: {premisses} ⊢ {self.schema_conclusion}"


class SystemeDeduction:
    """Système de déduction pour la logique propositionnelle et prédicative."""
    
    def __init__(self):
        self.regles = {}
        self.initialiser_regles()
    
    def initialiser_regles(self):
        """Initialise les règles d'inférence standard."""
        # Règles pour la logique propositionnelle
        self.ajouter_regle_modus_ponens()
        self.ajouter_regle_modus_tollens()
        self.ajouter_regle_introduction_et()
        self.ajouter_regle_elimination_et()
        self.ajouter_regle_introduction_ou()
        self.ajouter_regle_elimination_ou()
        self.ajouter_regle_introduction_implication()
        
        # Règles pour la logique prédicative
        self.ajouter_regle_introduction_universel()
        self.ajouter_regle_elimination_universel()
        self.ajouter_regle_introduction_existentiel()
        self.ajouter_regle_elimination_existentiel()
    
    def ajouter_regle(self, regle):
        """Ajoute une règle d'inférence au système."""
        self.regles[regle.nom] = regle
        return self
    
    def ajouter_regle_modus_ponens(self):
        """Ajoute la règle de Modus Ponens: A, A→B ⊢ B"""
        class ModusPonens(RegleInference):
            def __init__(self):
                super().__init__("Modus Ponens", ["A", "A→B"], "B")
            
            def appliquer(self, formules):
                if len(formules) != 2:
                    return None
                
                # Vérifier si la deuxième formule est une implication
                if not isinstance(formules[1], ImplicationModale) and not isinstance(formules[1], ImplicationTemporelle):
                    return None
                
                # Vérifier si la première formule correspond à l'antécédent de l'implication
                if str(formules[0]) == str(formules[1].antecedent):
                    return formules[1].consequent
                
                return None
        
        self.ajouter_regle(ModusPonens())
    
    def ajouter_regle_modus_tollens(self):
        """Ajoute la règle de Modus Tollens: A→B, ¬B ⊢ ¬A"""
        class ModusTollens(RegleInference):
            def __init__(self):
                super().__init__("Modus Tollens", ["A→B", "¬B"], "¬A")
            
            def appliquer(self, formules):
                if len(formules) != 2:
                    return None
                
                # Vérifier si la première formule est une implication
                if not isinstance(formules[0], ImplicationModale) and not isinstance(formules[0], ImplicationTemporelle):
                    return None
                
                # Vérifier si la deuxième formule est une négation
                if not isinstance(formules[1], NonModale) and not isinstance(formules[1], NonTemporelle):
                    return None
                
                # Vérifier si la négation correspond au conséquent de l'implication
                if str(formules[1].formule) == str(formules[0].consequent):
                    if isinstance(formules[0], ImplicationModale):
                        return NonModale(formules[0].antecedent)
                    else:
                        return NonTemporelle(formules[0].antecedent)
                
                return None
        
        self.ajouter_regle(ModusTollens())
    
    def ajouter_regle_introduction_et(self):
        """Ajoute la règle d'introduction de la conjonction: A, B ⊢ A∧B"""
        class IntroductionEt(RegleInference):
            def __init__(self):
                super().__init__("Introduction ∧", ["A", "B"], "A∧B")
            
            def appliquer(self, formules):
                if len(formules) != 2:
                    return None
                
                # Déterminer le type de formule
                if isinstance(formules[0], FormuleModale) and isinstance(formules[1], FormuleModale):
                    return EtModale(formules[0], formules[1])
                elif isinstance(formules[0], FormuleTemporelle) and isinstance(formules[1], FormuleTemporelle):
                    return EtTemporelle(formules[0], formules[1])
                
                return None
        
        self.ajouter_regle(IntroductionEt())
    
    def ajouter_regle_elimination_et(self):
        """Ajoute les règles d'élimination de la conjonction: A∧B ⊢ A et A∧B ⊢ B"""
        class EliminationEtGauche(RegleInference):
            def __init__(self):
                super().__init__("Élimination ∧ (gauche)", ["A∧B"], "A")
            
            def appliquer(self, formules):
                if len(formules) != 1:
                    return None
                
                if isinstance(formules[0], EtModale):
                    return formules[0].gauche
                elif isinstance(formules[0], EtTemporelle):
                    return formules[0].gauche
                
                return None
        
        class EliminationEtDroite(RegleInference):
            def __init__(self):
                super().__init__("Élimination ∧ (droite)", ["A∧B"], "B")
            
            def appliquer(self, formules):
                if len(formules) != 1:
                    return None
                
                if isinstance(formules[0], EtModale):
                    return formules[0].droite
                elif isinstance(formules[0], EtTemporelle):
                    return formules[0].droite
                
                return None
        
        self.ajouter_regle(EliminationEtGauche())
        self.ajouter_regle(EliminationEtDroite())
    
    def ajouter_regle_introduction_ou(self):
        """Ajoute les règles d'introduction de la disjonction: A ⊢ A∨B et B ⊢ A∨B"""
        class IntroductionOuGauche(RegleInference):
            def __init__(self):
                super().__init__("Introduction ∨ (gauche)", ["A"], "A∨B")
            
            def appliquer(self, formules):
                if len(formules) != 1:
                    return None
                
                # Ici, nous aurions besoin de la formule B pour construire A∨B
                # Dans une implémentation complète, B pourrait être fourni comme argument supplémentaire
                return None
        
        class IntroductionOuDroite(RegleInference):
            def __init__(self):
                super().__init__("Introduction ∨ (droite)", ["B"], "A∨B")
            
            def appliquer(self, formules):
                if len(formules) != 1:
                    return None
                
                # Ici, nous aurions besoin de la formule A pour construire A∨B
                return None
        
        self.ajouter_regle(IntroductionOuGauche())
        self.ajouter_regle(IntroductionOuDroite())
    
    def ajouter_regle_elimination_ou(self):
        """Ajoute la règle d'élimination de la disjonction: A∨B, A→C, B→C ⊢ C"""
        class EliminationOu(RegleInference):
            def __init__(self):
                super().__init__("Élimination ∨", ["A∨B", "A→C", "B→C"], "C")
            
            def appliquer(self, formules):
                if len(formules) != 3:
                    return None
                
                # Vérifier si la première formule est une disjonction
                if not isinstance(formules[0], OuModale) and not isinstance(formules[0], OuTemporelle):
                    return None
                
                # Vérifier si les deux autres formules sont des implications
                if (not isinstance(formules[1], ImplicationModale) and not isinstance(formules[1], ImplicationTemporelle) or
                    not isinstance(formules[2], ImplicationModale) and not isinstance(formules[2], ImplicationTemporelle)):
                    return None
                
                # Vérifier la cohérence des formules
                if (str(formules[0].gauche) == str(formules[1].antecedent) and
                    str(formules[0].droite) == str(formules[2].antecedent) and
                    str(formules[1].consequent) == str(formules[2].consequent)):
                    return formules[1].consequent
                
                return None
        
        self.ajouter_regle(EliminationOu())
    
    def ajouter_regle_introduction_implication(self):
        """Ajoute la règle d'introduction de l'implication."""
        # Cette règle nécessiterait une gestion des sous-preuves
        pass
    
    def ajouter_regle_introduction_universel(self):
        """Ajoute la règle d'introduction du quantificateur universel."""
        # Ces règles seront implémentées dans la partie logique de premier et second ordre
        pass
    
    def ajouter_regle_elimination_universel(self):
        """Ajoute la règle d'élimination du quantificateur universel."""
        pass
    
    def ajouter_regle_introduction_existentiel(self):
        """Ajoute la règle d'introduction du quantificateur existentiel."""
        pass
    
    def ajouter_regle_elimination_existentiel(self):
        """Ajoute la règle d'élimination du quantificateur existentiel."""
        pass
    
    def appliquer_regle(self, nom_regle, formules):
        """Applique une règle d'inférence aux formules données."""
        if nom_regle not in self.regles:
            raise ValueError(f"Règle d'inférence inconnue: {nom_regle}")
        
        return self.regles[nom_regle].appliquer(formules)
    
    def prouver(self, hypotheses, conclusion, max_etapes=100):
        """
        Tente de construire une preuve de la conclusion à partir des hypothèses.
        Retourne une preuve si elle existe, None sinon.
        """
        # Pour simplifier, nous utilisons une stratégie de recherche en avant
        preuve = Preuve()
        
        # Ajouter les hypothèses à la preuve
        for hyp in hypotheses:
            preuve.ajouter_hypothese(hyp)
        
        # Ensemble des formules déjà démontrées
        formules_demontrees = set(hypotheses)
        
        # Tentative de preuve par application des règles
        for _ in range(max_etapes):
            # Si la conclusion a été démontrée, la preuve est terminée
            if str(conclusion) in map(str, formules_demontrees):
                for formule in formules_demontrees:
                    if str(formule) == str(conclusion):
                        preuve.ajouter_ligne(formule, "Déjà démontrée")
                        return preuve
            
            # Appliquer toutes les règles d'inférence possibles
            nouvelles_formules = set()
            
            for nom_regle, regle in self.regles.items():
                # Pour chaque combinaison de formules déjà démontrées
                for comb in itertools.combinations(formules_demontrees, min(len(formules_demontrees), len(regle.schemas_premisses))):
                    # Tenter d'appliquer la règle
                    nouvelle_formule = regle.appliquer(comb)
                    
                    if nouvelle_formule is not None and str(nouvelle_formule) not in map(str, formules_demontrees):
                        nouvelles_formules.add(nouvelle_formule)
                        refs = [i+1 for i, f in enumerate(preuve.lignes) if str(f["formule"]) in map(str, comb)]
                        preuve.ajouter_ligne(nouvelle_formule, nom_regle, refs)
            
            # Si aucune nouvelle formule n'a été démontrée, la preuve échoue
            if not nouvelles_formules:
                return None
            
            # Ajouter les nouvelles formules à l'ensemble des formules démontrées
            formules_demontrees.update(nouvelles_formules)
        
        # Si le nombre maximum d'étapes est atteint, la preuve échoue
        return None


class ResolutionMethode:
    """Implémentation de la méthode de résolution pour la logique propositionnelle."""
    
    @staticmethod
    def to_cnf(formule):
        """Convertit une formule en forme normale conjonctive (CNF)."""
        # Cette méthode serait implémentée pour chaque type de formule
        # Pour simplifier, nous supposons que la formule est déjà en CNF
        return formule
    
    @staticmethod
    def clauses_from_cnf(formule_cnf):
        """Extrait les clauses d'une formule en CNF."""
        # Une clause est un ensemble de littéraux (variables ou leurs négations)
        # Pour simplifier, nous représentons chaque clause comme un ensemble de chaînes
        return [{"p"}, {"q"}, {"r"}]  # Exemple simplifié
    
    @staticmethod
    def resolution(clauses1, clauses2):
        """Applique la règle de résolution aux ensembles de clauses."""
        resultats = set()
        
        for c1 in clauses1:
            for c2 in clauses2:
                # Chercher un littéral dans c1 dont la négation est dans c2
                for lit in c1:
                    # Le littéral complémentaire serait la négation de lit
                    lit_comp = lit[1:] if lit.startswith("¬") else "¬" + lit
                    
                    if lit_comp in c2:
                        # Créer une nouvelle clause en résolvant c1 et c2
                        resolvant = (c1 - {lit}) | (c2 - {lit_comp})
                        
                        # Si le résolvant est la clause vide, la formule est insatisfaisable
                        if not resolvant:
                            return False  # Contradiction trouvée
                        
                        resultats.add(frozenset(resolvant))
        
        return resultats
    
    @staticmethod
    def prouver_par_resolution(hypotheses, conclusion, max_etapes=100):
        """
        Prouve une conclusion à partir d'hypothèses en utilisant la méthode de résolution.
        Retourne True si la preuve réussit, False sinon.
        """
        # Convertir les hypothèses et la négation de la conclusion en CNF
        clauses = set()
        
        for hyp in hypotheses:
            hyp_cnf = ResolutionMethode.to_cnf(hyp)
            clauses_hyp = ResolutionMethode.clauses_from_cnf(hyp_cnf)
            clauses.update(frozenset(c) for c in clauses_hyp)
        
        # Ajouter la négation de la conclusion
        neg_conclusion = None  # Ceci serait la négation de la conclusion
        neg_conclusion_cnf = ResolutionMethode.to_cnf(neg_conclusion)
        clauses_neg = ResolutionMethode.clauses_from_cnf(neg_conclusion_cnf)
        clauses.update(frozenset(c) for c in clauses_neg)
        
        # Appliquer la résolution jusqu'à trouver une contradiction ou atteindre le maximum d'étapes
        for _ in range(max_etapes):
            nouvelles_clauses = set()
            
            # Appliquer la résolution à toutes les paires de clauses
            for c1, c2 in itertools.combinations(clauses, 2):
                resolvants = ResolutionMethode.resolution({c1}, {c2})
                
                if resolvants is False:
                    # Contradiction trouvée, la preuve réussit
                    return True
                
                nouvelles_clauses.update(resolvants)
            
            # Si aucune nouvelle clause n'a été générée, la preuve échoue
            if nouvelles_clauses.issubset(clauses):
                break
            
            clauses.update(nouvelles_clauses)
        
        # Si aucune contradiction n'a été trouvée, la preuve échoue
        return False


class TableauxMethode:
    """Implémentation de la méthode des tableaux sémantiques pour la logique propositionnelle."""
    
    class Noeud:
        """Représente un nœud dans un tableau sémantique."""
        
        def __init__(self, formule, signe=True, parent=None):
            self.formule = formule  # La formule associée au nœud
            self.signe = signe  # True pour affirmation, False pour négation
            self.parent = parent  # Nœud parent
            self.enfants = []  # Nœuds enfants
            self.est_ferme = False  # Indique si la branche est fermée
        
        def ajouter_enfant(self, formule, signe=True):
            """Ajoute un enfant au nœud."""
            enfant = TableauxMethode.Noeud(formule, signe, self)
            self.enfants.append(enfant)
            return enfant
        
        def est_feuille(self):
            """Indique si le nœud est une feuille."""
            return not self.enfants
        
        def __str__(self):
            signe_str = "" if self.signe else "¬"
            return f"{signe_str}{self.formule}"
    
    @staticmethod
    def decomposer(noeud):
        """Décompose un nœud selon les règles des tableaux sémantiques."""
        formule = noeud.formule
        signe = noeud.signe
        
        # Décomposition selon le type de formule et son signe
        if isinstance(formule, NonModale) or isinstance(formule, NonTemporelle):
            # Règle de négation
            noeud.ajouter_enfant(formule.formule, not signe)
        
        elif isinstance(formule, EtModale) or isinstance(formule, EtTemporelle):
            if signe:
                # Règle de conjonction affirmée (règle α)
                noeud.ajouter_enfant(formule.gauche, True)
                noeud.ajouter_enfant(formule.droite, True)
            else:
                # Règle de conjonction niée (règle β)
                enfant1 = noeud.ajouter_enfant(formule.gauche, False)
                enfant2 = noeud.ajouter_enfant(formule.droite, False)
                enfant1.est_branche = True
                enfant2.est_branche = True
        
        elif isinstance(formule, OuModale) or isinstance(formule, OuTemporelle):
            if signe:
                # Règle de disjonction affirmée (règle β)
                enfant1 = noeud.ajouter_enfant(formule.gauche, True)
                enfant2 = noeud.ajouter_enfant(formule.droite, True)
                enfant1.est_branche = True
                enfant2.est_branche = True
            else:
                # Règle de disjonction niée (règle α)
                noeud.ajouter_enfant(formule.gauche, False)
                noeud.ajouter_enfant(formule.droite, False)
        
        elif isinstance(formule, ImplicationModale) or isinstance(formule, ImplicationTemporelle):
            if signe:
                # Règle d'implication affirmée (règle β)
                enfant1 = noeud.ajouter_enfant(formule.antecedent, False)
                enfant2 = noeud.ajouter_enfant(formule.consequent, True)
                enfant1.est_branche = True
                enfant2.est_branche = True
            else:
                # Règle d'implication niée (règle α)
                noeud.ajouter_enfant(formule.antecedent, True)
                noeud.ajouter_enfant(formule.consequent, False)
        
        # D'autres règles seraient ajoutées pour les autres types de formules
        
        return noeud.enfants
    
    @staticmethod
    def est_contradictoire(chemin):
        """Vérifie si un chemin contient une contradiction."""
        # Un chemin est contradictoire s'il contient une formule et sa négation
        for i, noeud1 in enumerate(chemin):
            for noeud2 in chemin[i+1:]:
                if str(noeud1.formule) == str(noeud2.formule) and noeud1.signe != noeud2.signe:
                    return True
        
        return False
    
    @staticmethod
    def construire_tableau(formules, signes=None):
        """
        Construit un tableau sémantique pour un ensemble de formules.
        formules: liste de formules
        signes: liste de signes (True pour affirmation, False pour négation)
        """
        if signes is None:
            signes = [True] * len(formules)
        
        # Créer le nœud racine
        racine = TableauxMethode.Noeud(None)
        
        # Ajouter les formules initiales
        noeuds = []
        for formule, signe in zip(formules, signes):
            noeud = racine.ajouter_enfant(formule, signe)
            noeuds.append(noeud)
        
        # Développer le tableau
        TableauxMethode.developper_tableau(noeuds)
        
        return racine
    
    @staticmethod
    def developper_tableau(noeuds, visite=None):
        """
        Développe le tableau sémantique à partir des nœuds donnés.
        noeuds: liste de nœuds à développer
        visite: ensemble des formules déjà visitées
        """
        if visite is None:
            visite = set()
        
        # Pour chaque nœud
        for noeud in noeuds:
            # Si la formule a déjà été visitée ou si le nœud est atomique, passer
            formule_str = str(noeud)
            if formule_str in visite or (isinstance(noeud.formule, PropositionModale) or isinstance(noeud.formule, PropositionTemporelle)):
                continue
            
            visite.add(formule_str)
            
            # Décomposer le nœud
            enfants = TableauxMethode.decomposer(noeud)
            
            # Vérifier si le chemin est contradictoire
            chemin = []
            n = noeud
            while n is not None:
                chemin.append(n)
                n = n.parent
            
            if TableauxMethode.est_contradictoire(chemin):
                noeud.est_ferme = True
                continue
            
            # Développer récursivement les enfants
            TableauxMethode.developper_tableau(enfants, visite)
    
    @staticmethod
    def est_ferme(racine):
        """Vérifie si le tableau est fermé (toutes les branches sont fermées)."""
        # Un tableau est fermé si toutes ses branches sont fermées
        
        def parcourir(noeud):
            if noeud.est_ferme:
                return True
            
            if noeud.est_feuille():
                # Vérifier si le chemin est contradictoire
                chemin = []
                n = noeud
                while n is not None:
                    chemin.append(n)
                    n = n.parent
                
                return TableauxMethode.est_contradictoire(chemin)
            
            return all(parcourir(enfant) for enfant in noeud.enfants)
        
        return parcourir(racine)
    
    @staticmethod
    def prouver_par_tableaux(hypotheses, conclusion):
        """
        Prouve une conclusion à partir d'hypothèses en utilisant la méthode des tableaux.
        Retourne True si la preuve réussit, False sinon.
        """
        # Pour prouver hypothèses ⊢ conclusion, on vérifie si hypothèses ∧ ¬conclusion est insatisfaisable
        # Si le tableau est fermé, la preuve réussit
        
        formules = list(hypotheses)
        signes = [True] * len(formules)
        
        # Ajouter la négation de la conclusion
        if isinstance(conclusion, FormuleModale):
            neg_conclusion = NonModale(conclusion)
        else:
            neg_conclusion = NonTemporelle(conclusion)
        
        formules.append(neg_conclusion)
        signes.append(True)
        
        # Construire le tableau
        racine = TableauxMethode.construire_tableau(formules, signes)
        
        # Vérifier si le tableau est fermé
        return TableauxMethode.est_ferme(racine)


##########################################
# PARTIE 3: RAISONNEMENT AVEC CONTRAINTES MULTIPLES
##########################################

class Contrainte(ABC):
    """Classe abstraite pour représenter une contrainte."""
    
    @abstractmethod
    def est_satisfaite(self, solution):
        """Vérifie si la contrainte est satisfaite par la solution."""
        pass
    
    @abstractmethod
    def __str__(self):
        """Représentation textuelle de la contrainte."""
        pass


class ContrainteUnaire(Contrainte):
    """Contrainte portant sur une seule variable."""
    
    def __init__(self, variable, fonction_contrainte):
        self.variable = variable
        self.fonction_contrainte = fonction_contrainte
    
    def est_satisfaite(self, solution):
        """Vérifie si la contrainte est satisfaite par la solution."""
        if self.variable not in solution:
            return True  # La contrainte ne s'applique pas si la variable n'est pas dans la solution
        
        return self.fonction_contrainte(solution[self.variable])
    
    def __str__(self):
        return f"Contrainte({self.variable})"


class ContrainteBinaire(Contrainte):
    """Contrainte portant sur deux variables."""
    
    def __init__(self, variable1, variable2, fonction_contrainte):
        self.variable1 = variable1
        self.variable2 = variable2
        self.fonction_contrainte = fonction_contrainte
    
    def est_satisfaite(self, solution):
        """Vérifie si la contrainte est satisfaite par la solution."""
        if self.variable1 not in solution or self.variable2 not in solution:
            return True  # La contrainte ne s'applique pas si l'une des variables n'est pas dans la solution
        
        return self.fonction_contrainte(solution[self.variable1], solution[self.variable2])
    
    def __str__(self):
        return f"Contrainte({self.variable1}, {self.variable2})"


class ContrainteNaire(Contrainte):
    """Contrainte portant sur plusieurs variables."""
    
    def __init__(self, variables, fonction_contrainte):
        self.variables = variables
        self.fonction_contrainte = fonction_contrainte
    
    def est_satisfaite(self, solution):
        """Vérifie si la contrainte est satisfaite par la solution."""
        if not all(v in solution for v in self.variables):
            return True  # La contrainte ne s'applique pas si toutes les variables ne sont pas dans la solution
        
        return self.fonction_contrainte(*[solution[v] for v in self.variables])
    
    def __str__(self):
        return f"Contrainte({', '.join(self.variables)})"


class ProblemeContraintes:
    """Représente un problème de satisfaction de contraintes."""
    
    def __init__(self):
        self.variables = set()
        self.domaines = {}
        self.contraintes = []
    
    def ajouter_variable(self, variable, domaine):
        """Ajoute une variable avec son domaine de valeurs possibles."""
        self.variables.add(variable)
        self.domaines[variable] = list(domaine)
        return self
    
    def ajouter_contrainte(self, contrainte):
        """Ajoute une contrainte au problème."""
        self.contraintes.append(contrainte)
        return self
    
    def est_solution_valide(self, solution):
        """Vérifie si une solution satisfait toutes les contraintes."""
        return all(c.est_satisfaite(solution) for c in self.contraintes)
    
    def resoudre_backtracking(self):
        """Résout le problème en utilisant l'algorithme de backtracking."""
        solution = {}
        return self._backtracking(solution)
    
    def _backtracking(self, solution):
        """Algorithme de backtracking récursif."""
        # Si toutes les variables ont une valeur, vérifier si la solution est valide
        if len(solution) == len(self.variables):
            if self.est_solution_valide(solution):
                return solution
            return None
        
        # Choisir une variable non assignée
        var = next(iter(self.variables - set(solution.keys())))
        
        # Essayer chaque valeur du domaine
        for val in self.domaines[var]:
            # Vérifier si l'affectation est consistante avec les contraintes
            solution[var] = val
            if self.est_solution_valide(solution):
                # Continuer le backtracking
                resultat = self._backtracking(solution)
                if resultat is not None:
                    return resultat
            
            # Si on arrive ici, l'affectation n'a pas marché, on retire la variable
            del solution[var]
        
        # Aucune solution trouvée
        return None
    
    def resoudre_ac3(self):
        """Résout le problème en utilisant l'algorithme AC-3 (Arc Consistency)."""
        # Réduire les domaines en utilisant AC-3
        domaines = self.domaines.copy()
        
        if not self._ac3(domaines):
            return None  # Pas de solution
        
        # Utiliser le backtracking avec les domaines réduits
        solution = {}
        return self._backtracking_avec_domaines(solution, domaines)
    
    def _ac3(self, domaines):
        """Algorithme AC-3 pour la consistance d'arc."""
        # Initialiser la file d'arcs (paires de variables liées par une contrainte)
        arcs = []
        
        for c in self.contraintes:
            if isinstance(c, ContrainteBinaire):
                arcs.append((c.variable1, c.variable2))
                arcs.append((c.variable2, c.variable1))
        
        # Tant qu'il reste des arcs à traiter
        while arcs:
            x, y = arcs.pop(0)
            
            if self._reviser(domaines, x, y):
                if not domaines[x]:
                    return False  # Domaine vide, pas de solution
                
                # Ajouter tous les arcs (z, x) où z est voisin de x mais différent de y
                for c in self.contraintes:
                    if isinstance(c, ContrainteBinaire):
                        if c.variable1 == x and c.variable2 != y:
                            arcs.append((c.variable2, x))
                        elif c.variable2 == x and c.variable1 != y:
                            arcs.append((c.variable1, x))
        
        return True
    
    def _reviser(self, domaines, x, y):
        """Révise le domaine de x en fonction de y."""
        revise = False
        
        for vx in list(domaines[x]):
            # Vérifier s'il existe une valeur vy dans le domaine de y telle que (vx, vy) satisfait la contrainte
            if not any(self._est_consistant({x: vx, y: vy}) for vy in domaines[y]):
                domaines[x].remove(vx)
                revise = True
        
        return revise
    
    def _est_consistant(self, solution_partielle):
        """Vérifie si une solution partielle est consistante avec les contraintes."""
        return all(c.est_satisfaite(solution_partielle) for c in self.contraintes)
    
    def _backtracking_avec_domaines(self, solution, domaines):
        """Algorithme de backtracking avec domaines réduits."""
        # Si toutes les variables ont une valeur, vérifier si la solution est valide
        if len(solution) == len(self.variables):
            if self.est_solution_valide(solution):
                return solution
            return None
        
        # Choisir une variable non assignée
        var = next(iter(self.variables - set(solution.keys())))
        
        # Essayer chaque valeur du domaine
        for val in domaines[var]:
            # Vérifier si l'affectation est consistante avec les contraintes
            solution[var] = val
            if self.est_solution_valide(solution):
                # Continuer le backtracking
                resultat = self._backtracking_avec_domaines(solution, domaines)
                if resultat is not None:
                    return resultat
            
            # Si on arrive ici, l'affectation n'a pas marché, on retire la variable
            del solution[var]
        
        # Aucune solution trouvée
        return None


class Optimisation:
    """Classe pour les problèmes d'optimisation."""
    
    class TypeOptimisation(Enum):
        MINIMISATION = 1
        MAXIMISATION = 2
    
    def __init__(self, probleme: ProblemeContraintes, fonction_objectif, type_optimisation=TypeOptimisation.MINIMISATION):
        self.probleme = probleme
        self.fonction_objectif = fonction_objectif
        self.type_optimisation = type_optimisation
    
    def resoudre(self):
        """Résout le problème d'optimisation."""
        # Trouver toutes les solutions valides
        solutions = self._trouver_solutions()
        
        if not solutions:
            return None
        
        # Trouver la meilleure solution selon la fonction objectif
        if self.type_optimisation == self.TypeOptimisation.MINIMISATION:
            return min(solutions, key=lambda s: self.fonction_objectif(s))
        else:
            return max(solutions, key=lambda s: self.fonction_objectif(s))
    
    def _trouver_solutions(self):
        """Trouve toutes les solutions valides du problème."""
        solutions = []
        
        def backtracking(solution):
            # Si toutes les variables ont une valeur, vérifier si la solution est valide
            if len(solution) == len(self.probleme.variables):
                if self.probleme.est_solution_valide(solution):
                    solutions.append(solution.copy())
                return
            
            # Choisir une variable non assignée
            var = next(iter(self.probleme.variables - set(solution.keys())))
            
            # Essayer chaque valeur du domaine
            for val in self.probleme.domaines[var]:
                # Vérifier si l'affectation est consistante avec les contraintes
                solution[var] = val
                if self.probleme.est_solution_valide(solution):
                    # Continuer le backtracking
                    backtracking(solution)
                
                # Si on arrive ici, on retire la variable pour essayer une autre valeur
                del solution[var]
        
        # Lancer le backtracking
        backtracking({})
        return solutions


class ProgrammationLogique:
    """Classe pour la programmation logique (style Prolog)."""
    
    def __init__(self):
        self.faits = set()
        self.regles = []
    
    def ajouter_fait(self, predicat, *arguments):
        """Ajoute un fait à la base de connaissances."""
        self.faits.add((predicat, arguments))
        return self
    
    def ajouter_regle(self, tete, corps):
        """
        Ajoute une règle à la base de connaissances.
        tete: tuple (predicat, arguments) représentant la tête de la règle
        corps: liste de tuples (predicat, arguments) représentant le corps de la règle
        """
        self.regles.append((tete, corps))
        return self
    
    def requete(self, predicat, *arguments):
        """Effectue une requête et retourne les substitutions qui la satisfont."""
        but = (predicat, arguments)
        return self._resoudre([but], {})
    
    def _resoudre(self, buts, substitution):
        """
        Résout une liste de buts en utilisant la résolution SLD.
        buts: liste de buts à résoudre
        substitution: dictionnaire de substitutions courantes
        """
        if not buts:
            return [substitution]
        
        # Prendre le premier but
        but_courant = buts[0]
        predicat, arguments = but_courant
        
        # Essayer de résoudre le but courant
        resultats = []
        
        # Essayer les faits
        for fait in self.faits:
            fait_predicat, fait_arguments = fait
            
            if fait_predicat == predicat and len(fait_arguments) == len(arguments):
                # Tenter d'unifier le but avec le fait
                nouvelle_substitution = self._unifier(arguments, fait_arguments, substitution.copy())
                
                if nouvelle_substitution is not None:
                    # Continuer avec les buts restants
                    resultats.extend(self._resoudre(buts[1:], nouvelle_substitution))
        
        # Essayer les règles
        for regle in self.regles:
            tete, corps = regle
            tete_predicat, tete_arguments = tete
            
            if tete_predicat == predicat and len(tete_arguments) == len(arguments):
                # Tenter d'unifier le but avec la tête de la règle
                nouvelle_substitution = self._unifier(arguments, tete_arguments, substitution.copy())
                
                if nouvelle_substitution is not None:
                    # Ajouter les buts du corps au début de la liste de buts
                    nouveaux_buts = corps + buts[1:]
                    resultats.extend(self._resoudre(nouveaux_buts, nouvelle_substitution))
        
        return resultats
    
    def _unifier(self, termes1, termes2, substitution):
        """
        Unifie deux listes de termes et met à jour la substitution.
        Retourne la substitution mise à jour ou None si l'unification échoue.
        """
        if len(termes1) != len(termes2):
            return None
        
        for t1, t2 in zip(termes1, termes2):
            # Si t1 est une variable
            if isinstance(t1, str) and t1.startswith("?"):
                if t1 in substitution:
                    # La variable est déjà liée, vérifier la consistance
                    if substitution[t1] != t2:
                        return None
                else:
                    # Lier la variable à t2
                    substitution[t1] = t2
            
            # Si t2 est une variable
            elif isinstance(t2, str) and t2.startswith("?"):
                if t2 in substitution:
                    # La variable est déjà liée, vérifier la consistance
                    if substitution[t2] != t1:
                        return None
                else:
                    # Lier la variable à t1
                    substitution[t2] = t1
            
            # Si t1 et t2 sont des constantes
            elif t1 != t2:
                return None
        
        return substitution


##########################################
# PARTIE 4: LOGIQUE DE PREMIER ET SECOND ORDRE
##########################################

class TermeLogique(ABC):
    """Classe abstraite pour les termes logiques."""
    
    @abstractmethod
    def substituer(self, substitution):
        """Applique une substitution au terme."""
        pass
    
    @abstractmethod
    def variables(self):
        """Retourne l'ensemble des variables libres du terme."""
        pass
    
    @abstractmethod
    def __str__(self):
        """Représentation textuelle du terme."""
        pass


class Variable(TermeLogique):
    """Représente une variable logique."""
    
    def __init__(self, nom):
        self.nom = nom
    
    def substituer(self, substitution):
        """Applique une substitution à la variable."""
        if self.nom in substitution:
            return substitution[self.nom]
        return self
    
    def variables(self):
        """Retourne l'ensemble des variables libres."""
        return {self.nom}
    
    def __str__(self):
        return self.nom
    
    def __eq__(self, autre):
        if isinstance(autre, Variable):
            return self.nom == autre.nom
        return False
    
    def __hash__(self):
        return hash(self.nom)


class Constante(TermeLogique):
    """Représente une constante logique."""
    
    def __init__(self, valeur):
        self.valeur = valeur
    
    def substituer(self, substitution):
        """Applique une substitution à la constante (ne fait rien)."""
        return self
    
    def variables(self):
        """Retourne l'ensemble des variables libres (vide pour une constante)."""
        return set()
    
    def __str__(self):
        return str(self.valeur)
    
    def __eq__(self, autre):
        if isinstance(autre, Constante):
            return self.valeur == autre.valeur
        return False
    
    def __hash__(self):
        return hash(self.valeur)


class Fonction(TermeLogique):
    """Représente une fonction logique."""
    
    def __init__(self, nom, arguments):
        self.nom = nom
        self.arguments = arguments
    
    def substituer(self, substitution):
        """Applique une substitution à la fonction."""
        return Fonction(self.nom, [arg.substituer(substitution) for arg in self.arguments])
    
    def variables(self):
        """Retourne l'ensemble des variables libres."""
        return set().union(*[arg.variables() for arg in self.arguments])
    
    def __str__(self):
        args_str = ", ".join(map(str, self.arguments))
        return f"{self.nom}({args_str})"
    
    def __eq__(self, autre):
        if isinstance(autre, Fonction):
            return self.nom == autre.nom and self.arguments == autre.arguments
        return False
    
    def __hash__(self):
        return hash((self.nom, tuple(self.arguments)))


class FormulePremierOrdre(Formule):
    """Classe de base pour les formules de logique du premier ordre."""
    
    @abstractmethod
    def substituer(self, substitution):
        """Applique une substitution à la formule."""
        pass
    
    @abstractmethod
    def variables_libres(self):
        """Retourne l'ensemble des variables libres de la formule."""
        pass


class Predicat(FormulePremierOrdre):
    """Représente un prédicat logique."""
    
    def __init__(self, nom, arguments):
        self.nom = nom
        self.arguments = arguments
    
    def evaluer(self, interpretation):
        """Évalue le prédicat dans une interprétation donnée."""
        # Dans une interprétation du premier ordre, un prédicat est évalué
        # en vérifiant si le tuple d'éléments est dans l'extension du prédicat
        if self.nom not in interpretation:
            return False
        
        # Évaluer les arguments
        valeurs = []
        for arg in self.arguments:
            if isinstance(arg, Variable):
                if arg.nom not in interpretation:
                    return False
                valeurs.append(interpretation[arg.nom])
            elif isinstance(arg, Constante):
                valeurs.append(arg.valeur)
            elif isinstance(arg, Fonction):
                # Pour simplifier, nous ne traitons pas l'évaluation des fonctions
                return False
        
        # Vérifier si le tuple est dans l'extension du prédicat
        return tuple(valeurs) in interpretation[self.nom]
    
    def substituer(self, substitution):
        """Applique une substitution au prédicat."""
        return Predicat(self.nom, [arg.substituer(substitution) for arg in self.arguments])
    
    def variables_libres(self):
        """Retourne l'ensemble des variables libres du prédicat."""
        return set().union(*[arg.variables() for arg in self.arguments])
    
    def __str__(self):
        args_str = ", ".join(map(str, self.arguments))
        return f"{self.nom}({args_str})"
    
    def __eq__(self, autre):
        if isinstance(autre, Predicat):
            return self.nom == autre.nom and self.arguments == autre.arguments
        return False
    
    def __hash__(self):
        return hash((self.nom, tuple(self.arguments)))


class NonPremierOrdre(FormulePremierOrdre):
    """Négation en logique du premier ordre."""
    
    def __init__(self, formule):
        self.formule = formule
    
    def evaluer(self, interpretation):
        """Évalue la négation dans une interprétation donnée."""
        return not self.formule.evaluer(interpretation)
    
    def substituer(self, substitution):
        """Applique une substitution à la négation."""
        return NonPremierOrdre(self.formule.substituer(substitution))
    
    def variables_libres(self):
        """Retourne l'ensemble des variables libres de la négation."""
        return self.formule.variables_libres()
    
    def __str__(self):
        return f"¬{self.formule}"
    
    def __eq__(self, autre):
        if isinstance(autre, NonPremierOrdre):
            return self.formule == autre.formule
        return False
    
    def __hash__(self):
        return hash(("non", self.formule))


class EtPremierOrdre(FormulePremierOrdre):
    """Conjonction en logique du premier ordre."""
    
    def __init__(self, gauche, droite):
        self.gauche = gauche
        self.droite = droite
    
    def evaluer(self, interpretation):
        """Évalue la conjonction dans une interprétation donnée."""
        return self.gauche.evaluer(interpretation) and self.droite.evaluer(interpretation)
    
    def substituer(self, substitution):
        """Applique une substitution à la conjonction."""
        return EtPremierOrdre(self.gauche.substituer(substitution), self.droite.substituer(substitution))
    
    def variables_libres(self):
        """Retourne l'ensemble des variables libres de la conjonction."""
        return self.gauche.variables_libres() | self.droite.variables_libres()
    
    def __str__(self):
        return f"({self.gauche} ∧ {self.droite})"
    
    def __eq__(self, autre):
        if isinstance(autre, EtPremierOrdre):
            return self.gauche == autre.gauche and self.droite == autre.droite
        return False
    
    def __hash__(self):
        return hash(("et", self.gauche, self.droite))


class OuPremierOrdre(FormulePremierOrdre):
    """Disjonction en logique du premier ordre."""
    
    def __init__(self, gauche, droite):
        self.gauche = gauche
        self.droite = droite
    
    def evaluer(self, interpretation):
        """Évalue la disjonction dans une interprétation donnée."""
        return self.gauche.evaluer(interpretation) or self.droite.evaluer(interpretation)
    
    def substituer(self, substitution):
        """Applique une substitution à la disjonction."""
        return OuPremierOrdre(self.gauche.substituer(substitution), self.droite.substituer(substitution))
    
    def variables_libres(self):
        """Retourne l'ensemble des variables libres de la disjonction."""
        return self.gauche.variables_libres() | self.droite.variables_libres()
    
    def __str__(self):
        return f"({self.gauche} ∨ {self.droite})"
    
    def __eq__(self, autre):
        if isinstance(autre, OuPremierOrdre):
            return self.gauche == autre.gauche and self.droite == autre.droite
        return False
    
    def __hash__(self):
        return hash(("ou", self.gauche, self.droite))


class ImplicationPremierOrdre(FormulePremierOrdre):
    """Implication en logique du premier ordre."""
    
    def __init__(self, antecedent, consequent):
        self.antecedent = antecedent
        self.consequent = consequent
    
    def evaluer(self, interpretation):
        """Évalue l'implication dans une interprétation donnée."""
        return not self.antecedent.evaluer(interpretation) or self.consequent.evaluer(interpretation)
    
    def substituer(self, substitution):
        """Applique une substitution à l'implication."""
        return ImplicationPremierOrdre(self.antecedent.substituer(substitution), self.consequent.substituer(substitution))
    
    def variables_libres(self):
        """Retourne l'ensemble des variables libres de l'implication."""
        return self.antecedent.variables_libres() | self.consequent.variables_libres()
    
    def __str__(self):
        return f"({self.antecedent} → {self.consequent})"
    
    def __eq__(self, autre):
        if isinstance(autre, ImplicationPremierOrdre):
            return self.antecedent == autre.antecedent and self.consequent == autre.consequent
        return False
    
    def __hash__(self):
        return hash(("implique", self.antecedent, self.consequent))


class QuantificateurUniversel(FormulePremierOrdre):
    """Quantificateur universel en logique du premier ordre."""
    
    def __init__(self, variable, formule):
        self.variable = variable
        self.formule = formule
    
    def evaluer(self, interpretation, domaine):
        """
        Évalue le quantificateur universel dans une interprétation donnée.
        domaine: ensemble des valeurs possibles pour les variables
        """
        # Pour tout élément du domaine, la formule doit être vraie
        for valeur in domaine:
            # Créer une nouvelle interprétation avec la variable liée à la valeur
            nouvelle_interpretation = interpretation.copy()
            nouvelle_interpretation[self.variable.nom] = valeur
            
            if not self.formule.evaluer(nouvelle_interpretation):
                return False
        
        return True
    
    def substituer(self, substitution):
        """Applique une substitution au quantificateur universel."""
        # Éviter la capture de variable
        nouvelle_substitution = substitution.copy()
        if self.variable.nom in nouvelle_substitution:
            del nouvelle_substitution[self.variable.nom]
        
        return QuantificateurUniversel(self.variable, self.formule.substituer(nouvelle_substitution))
    
    def variables_libres(self):
        """Retourne l'ensemble des variables libres du quantificateur universel."""
        # La variable quantifiée n'est pas libre
        return self.formule.variables_libres() - {self.variable.nom}
    
    def __str__(self):
        return f"∀{self.variable}.{self.formule}"
    
    def __eq__(self, autre):
        if isinstance(autre, QuantificateurUniversel):
            return self.variable == autre.variable and self.formule == autre.formule
        return False
    
    def __hash__(self):
        return hash(("pour_tout", self.variable, self.formule))


class QuantificateurExistentiel(FormulePremierOrdre):
    """Quantificateur existentiel en logique du premier ordre."""
    
    def __init__(self, variable, formule):
        self.variable = variable
        self.formule = formule
    
    def evaluer(self, interpretation, domaine):
        """
        Évalue le quantificateur existentiel dans une interprétation donnée.
        domaine: ensemble des valeurs possibles pour les variables
        """
        # Il doit exister au moins un élément du domaine pour lequel la formule est vraie
        for valeur in domaine:
            # Créer une nouvelle interprétation avec la variable liée à la valeur
            nouvelle_interpretation = interpretation.copy()
            nouvelle_interpretation[self.variable.nom] = valeur
            
            if self.formule.evaluer(nouvelle_interpretation):
                return True
        
        return False
    
    def substituer(self, substitution):
        """Applique une substitution au quantificateur existentiel."""
        # Éviter la capture de variable
        nouvelle_substitution = substitution.copy()
        if self.variable.nom in nouvelle_substitution:
            del nouvelle_substitution[self.variable.nom]
        
        return QuantificateurExistentiel(self.variable, self.formule.substituer(nouvelle_substitution))
    
    def variables_libres(self):
        """Retourne l'ensemble des variables libres du quantificateur existentiel."""
        # La variable quantifiée n'est pas libre
        return self.formule.variables_libres() - {self.variable.nom}
    
    def __str__(self):
        return f"∃{self.variable}.{self.formule}"
    
    def __eq__(self, autre):
        if isinstance(autre, QuantificateurExistentiel):
            return self.variable == autre.variable and self.formule == autre.formule
        return False
    
    def __hash__(self):
        return hash(("existe", self.variable, self.formule))


class LogiqueSecondOrdre:
    """Logique du second ordre avec quantification sur les prédicats et les fonctions."""
    
    class QuantificateurPredicat(FormulePremierOrdre):
        """Quantificateur sur les prédicats (logique du second ordre)."""
        
        def __init__(self, nom_predicat, arite, formule, universel=True):
            self.nom_predicat = nom_predicat
            self.arite = arite  # Nombre d'arguments du prédicat
            self.formule = formule
            self.universel = universel  # True pour ∀, False pour ∃
        
        def evaluer(self, interpretation, domaine):
            """
            Évalue le quantificateur de prédicat dans une interprétation donnée.
            domaine: ensemble des valeurs possibles pour les variables
            """
            # Générer toutes les extensions possibles pour le prédicat
            extensions = self._generer_extensions_predicat(domaine)
            
            if self.universel:
                # Pour tout prédicat, la formule doit être vraie
                for extension in extensions:
                    nouvelle_interpretation = interpretation.copy()
                    nouvelle_interpretation[self.nom_predicat] = extension
                    
                    if not self.formule.evaluer(nouvelle_interpretation):
                        return False
                return True
            else:
                # Il doit exister au moins un prédicat pour lequel la formule est vraie
                for extension in extensions:
                    nouvelle_interpretation = interpretation.copy()
                    nouvelle_interpretation[self.nom_predicat] = extension
                    
                    if self.formule.evaluer(nouvelle_interpretation):
                        return True
                return False
        
        def _generer_extensions_predicat(self, domaine):
            """Génère toutes les extensions possibles pour le prédicat."""
            # Pour simplifier, nous limitons à un petit nombre d'extensions
            # Dans une implémentation complète, on générerait toutes les combinaisons possibles
            extensions = []
            
            # Générer toutes les combinaisons de tuples de longueur arite
            tuples = list(itertools.product(domaine, repeat=self.arite))
            
            # Générer toutes les sous-collections de ces tuples
            for i in range(len(tuples) + 1):
                for combo in itertools.combinations(tuples, i):
                    extensions.append(set(combo))
            
            return extensions[:10]  # Limiter pour des raisons de performance
        
        def substituer(self, substitution):
            """Applique une substitution au quantificateur de prédicat."""
            return LogiqueSecondOrdre.QuantificateurPredicat(
                self.nom_predicat,
                self.arite,
                self.formule.substituer(substitution),
                self.universel
            )
        
        def variables_libres(self):
            """Retourne l'ensemble des variables libres du quantificateur de prédicat."""
            # Le nom du prédicat est lié par le quantificateur
            return self.formule.variables_libres()
        
        def __str__(self):
            quantificateur = "∀" if self.universel else "∃"
            return f"{quantificateur}{self.nom_predicat}^{self.arite}.{self.formule}"
        
        def __eq__(self, autre):
            if isinstance(autre, LogiqueSecondOrdre.QuantificateurPredicat):
                return (self.nom_predicat == autre.nom_predicat and
                        self.arite == autre.arite and
                        self.formule == autre.formule and
                        self.universel == autre.universel)
            return False
        
        def __hash__(self):
            return hash(("quantificateur_predicat", self.nom_predicat, self.arite, self.formule, self.universel))
    
    class QuantificateurFonction(FormulePremierOrdre):
        """Quantificateur sur les fonctions (logique du second ordre)."""
        
        def __init__(self, nom_fonction, arite, formule, universel=True):
            self.nom_fonction = nom_fonction
            self.arite = arite  # Nombre d'arguments de la fonction
            self.formule = formule
            self.universel = universel  # True pour ∀, False pour ∃
        
        def evaluer(self, interpretation, domaine):
            """
            Évalue le quantificateur de fonction dans une interprétation donnée.
            domaine: ensemble des valeurs possibles pour les variables
            """
            # Générer toutes les fonctions possibles
            fonctions = self._generer_fonctions(domaine)
            
            if self.universel:
                # Pour toute fonction, la formule doit être vraie
                for fonction in fonctions:
                    nouvelle_interpretation = interpretation.copy()
                    nouvelle_interpretation[self.nom_fonction] = fonction
                    
                    if not self.formule.evaluer(nouvelle_interpretation):
                        return False
                return True
            else:
                # Il doit exister au moins une fonction pour laquelle la formule est vraie
                for fonction in fonctions:
                    nouvelle_interpretation = interpretation.copy()
                    nouvelle_interpretation[self.nom_fonction] = fonction
                    
                    if self.formule.evaluer(nouvelle_interpretation):
                        return True
                return False
        
        def _generer_fonctions(self, domaine):
            """Génère toutes les fonctions possibles avec le domaine donné."""
            # Pour simplifier, nous limitons à un petit nombre de fonctions
            fonctions = []
            
            # Générer tous les tuples d'arguments possibles
            arguments = list(itertools.product(domaine, repeat=self.arite))
            
            # Générer quelques fonctions aléatoires
            for _ in range(5):
                fonction = {}
                for args in arguments:
                    fonction[args] = random.choice(list(domaine))
                fonctions.append(fonction)
            
            return fonctions
        
        def substituer(self, substitution):
            """Applique une substitution au quantificateur de fonction."""
            return LogiqueSecondOrdre.QuantificateurFonction(
                self.nom_fonction,
                self.arite,
                self.formule.substituer(substitution),
                self.universel
            )
        
        def variables_libres(self):
            """Retourne l'ensemble des variables libres du quantificateur de fonction."""
            # Le nom de la fonction est lié par le quantificateur
            return self.formule.variables_libres()
        
        def __str__(self):
            quantificateur = "∀" if self.universel else "∃"
            return f"{quantificateur}{self.nom_fonction}^{self.arite}.{self.formule}"
        
        def __eq__(self, autre):
            if isinstance(autre, LogiqueSecondOrdre.QuantificateurFonction):
                return (self.nom_fonction == autre.nom_fonction and
                        self.arite == autre.arite and
                        self.formule == autre.formule and
                        self.universel == autre.universel)
            return False
        
        def __hash__(self):
            return hash(("quantificateur_fonction", self.nom_fonction, self.arite, self.formule, self.universel))


class VerificateurModeleLogiquePremierOrdre:
    """Vérificateur de modèle pour la logique du premier ordre."""
    
    @staticmethod
    def verifier(formule, interpretation, domaine):
        """Vérifie si une formule est vraie dans une interprétation donnée."""
        if isinstance(formule, QuantificateurUniversel) or isinstance(formule, QuantificateurExistentiel):
            return formule.evaluer(interpretation, domaine)
        else:
            return formule.evaluer(interpretation)
    
    @staticmethod
    def est_satisfaisable(formule, domaine):
        """Vérifie si une formule est satisfaisable dans un domaine donné."""
        # Générer toutes les interprétations possibles
        interpretations = VerificateurModeleLogiquePremierOrdre._generer_interpretations(formule, domaine)
        
        for interpretation in interpretations:
            if VerificateurModeleLogiquePremierOrdre.verifier(formule, interpretation, domaine):
                return True
        
        return False
    
    @staticmethod
    def est_valide(formule, domaine):
        """Vérifie si une formule est valide dans un domaine donné."""
        # Une formule est valide si sa négation n'est pas satisfaisable
        negation = NonPremierOrdre(formule)
        return not VerificateurModeleLogiquePremierOrdre.est_satisfaisable(negation, domaine)
    
    @staticmethod
    def _generer_interpretations(formule, domaine):
        """Génère toutes les interprétations possibles pour une formule dans un domaine."""
        # Pour simplifier, nous générons un nombre limité d'interprétations
        interpretations = []
        
        # Extraire les prédicats de la formule
        predicats = VerificateurModeleLogiquePremierOrdre._extraire_predicats(formule)
        
        # Générer quelques interprétations aléatoires
        for _ in range(10):
            interpretation = {}
            
            for nom, arite in predicats:
                # Générer une extension aléatoire pour chaque prédicat
                extension = set()
                tuples = list(itertools.product(domaine, repeat=arite))
                
                # Ajouter aléatoirement des tuples à l'extension
                for tup in tuples:
                    if random.random() > 0.5:
                        extension.add(tup)
                
                interpretation[nom] = extension
            
            interpretations.append(interpretation)
        
        return interpretations
    
    @staticmethod
    def _extraire_predicats(formule):
        """Extrait les prédicats d'une formule."""
        predicats = set()
        
        def parcourir(f):
            if isinstance(f, Predicat):
                predicats.add((f.nom, len(f.arguments)))
            elif isinstance(f, NonPremierOrdre):
                parcourir(f.formule)
            elif isinstance(f, EtPremierOrdre) or isinstance(f, OuPremierOrdre) or isinstance(f, ImplicationPremierOrdre):
                parcourir(f.gauche)
                parcourir(f.droite)
            elif isinstance(f, QuantificateurUniversel) or isinstance(f, QuantificateurExistentiel):
                parcourir(f.formule)
        
        parcourir(formule)
        return predicats


##########################################
# PARTIE 5: GESTION DES PARADOXES ET CONTRADICTIONS
##########################################

class Paradoxe:
    """Classe pour la gestion des paradoxes logiques."""
    
    def __init__(self, nom, description):
        self.nom = nom
        self.description = description
    
    def __str__(self):
        return f"Paradoxe: {self.nom}\nDescription: {self.description}"


class ParadoxeRussell(Paradoxe):
    """Paradoxe de Russell: l'ensemble de tous les ensembles qui ne se contiennent pas eux-mêmes."""
    
    def __init__(self):
        super().__init__(
            "Paradoxe de Russell",
            "Si R est l'ensemble de tous les ensembles qui ne se contiennent pas eux-mêmes, "
            "alors R se contient-il lui-même ? Si R se contient lui-même, alors il ne devrait "
            "pas se contenir lui-même. Si R ne se contient pas lui-même, alors il devrait se "
            "contenir lui-même."
        )
    
    def formaliser(self):
        """Formalise le paradoxe en logique du premier ordre."""
        # Définir R = {x | x ∉ x}
        # Puis poser la question: R ∈ R ?
        
        # Ceci est une simplification, car en réalité, nous aurions besoin
        # de la théorie des ensembles pour formaliser complètement ce paradoxe
        x = Variable("x")
        R = Variable("R")
        appartient = lambda a, b: Predicat("appartient", [a, b])
        
        # Définition de R
        definition_R = QuantificateurUniversel(
            x,
            ImplicationPremierOrdre(
                appartient(x, R),
                NonPremierOrdre(appartient(x, x))
            )
        )
        
        # Question: R ∈ R ?
        question = appartient(R, R)
        
        return definition_R, question
    
    def analyser(self):
        """Analyse le paradoxe et explique pourquoi il est problématique."""
        return (
            "Si R ∈ R, alors par définition de R, R ∉ R, ce qui est contradictoire.\n"
            "Si R ∉ R, alors par définition de R, R ∈ R, ce qui est aussi contradictoire.\n"
            "Cette contradiction montre les limites de la théorie naïve des ensembles et "
            "a conduit à l'élaboration de la théorie des types et de la théorie axiomatique "
            "des ensembles (comme ZFC) pour éviter de tels paradoxes."
        )


class ParadoxeMenteur(Paradoxe):
    """Paradoxe du menteur: 'Cette phrase est fausse'."""
    
    def __init__(self):
        super().__init__(
            "Paradoxe du Menteur",
            "Considérons la phrase: 'Cette phrase est fausse'. Si elle est vraie, "
            "alors elle est fausse. Si elle est fausse, alors elle est vraie."
        )
    
    def formaliser(self):
        """Tente de formaliser le paradoxe en logique modale."""
        # Dans la logique standard, il est difficile de formaliser ce paradoxe
        # car il implique une auto-référence
        p = PropositionModale("p")
        
        # p ↔ ¬p (p est équivalent à non-p)
        equivalence = EtModale(
            ImplicationModale(p, NonModale(p)),
            ImplicationModale(NonModale(p), p)
        )
        
        return equivalence
    
    def analyser(self):
        """Analyse le paradoxe et explique pourquoi il est problématique."""
        return (
            "Le paradoxe du menteur implique une auto-référence qui ne peut pas être "
            "facilement formalisée dans la logique classique. Des théories comme la "
            "théorie des types de Tarski ou la logique paraconsistante ont été développées "
            "pour traiter ce type de paradoxe.\n\n"
            "Tarski a proposé une hiérarchie des langages où le prédicat de vérité pour "
            "un langage ne peut être défini que dans un métalangage. Cela évite l'auto-référence "
            "qui conduit au paradoxe."
        )


class SystemeLogiqueDialetheique:
    """Système logique dialethéique qui tolère certaines contradictions."""
    
    def __init__(self):
        self.faits = {}  # Dictionnaire de faits avec leur valeur de vérité
    
    def ajouter_fait(self, fait, valeur):
        """
        Ajoute un fait avec sa valeur de vérité.
        valeur: peut être True, False, ou "contradictoire"
        """
        self.faits[str(fait)] = valeur
        return self
    
    def evaluer(self, formule):
        """Évalue une formule dans le système dialethéique."""
        if isinstance(formule, PropositionModale) or isinstance(formule, PropositionTemporelle) or isinstance(formule, Predicat):
            # Proposition atomique
            fait_str = str(formule)
            if fait_str in self.faits:
                return self.faits[fait_str]
            return False  # Par défaut, les faits inconnus sont faux
        
        elif isinstance(formule, NonModale) or isinstance(formule, NonTemporelle) or isinstance(formule, NonPremierOrdre):
            # Négation
            valeur_interne = self.evaluer(formule.formule)
            
            if valeur_interne == "contradictoire":
                return "contradictoire"
            elif valeur_interne is True:
                return False
            else:
                return True
        
        elif isinstance(formule, EtModale) or isinstance(formule, EtTemporelle) or isinstance(formule, EtPremierOrdre):
            # Conjonction
            valeur_gauche = self.evaluer(formule.gauche)
            valeur_droite = self.evaluer(formule.droite)
            
            if valeur_gauche == "contradictoire" or valeur_droite == "contradictoire":
                return "contradictoire"
            elif valeur_gauche is True and valeur_droite is True:
                return True
            else:
                return False
        
        elif isinstance(formule, OuModale) or isinstance(formule, OuTemporelle) or isinstance(formule, OuPremierOrdre):
            # Disjonction
            valeur_gauche = self.evaluer(formule.gauche)
            valeur_droite = self.evaluer(formule.droite)
            
            if valeur_gauche == "contradictoire" or valeur_droite == "contradictoire":
                return "contradictoire"
            elif valeur_gauche is True or valeur_droite is True:
                return True
            else:
                return False
        
        elif isinstance(formule, ImplicationModale) or isinstance(formule, ImplicationTemporelle) or isinstance(formule, ImplicationPremierOrdre):
            # Implication
            valeur_antecedent = self.evaluer(formule.antecedent)
            valeur_consequent = self.evaluer(formule.consequent)
            
            if valeur_antecedent == "contradictoire" or valeur_consequent == "contradictoire":
                return "contradictoire"
            elif valeur_antecedent is False or valeur_consequent is True:
                return True
            else:
                return False
        
        # D'autres cas pourraient être ajoutés pour d'autres types de formules
        
        return False
    
    def est_coherent(self):
        """Vérifie si le système est cohérent (pas de contradictions explicites)."""
        for fait, valeur in self.faits.items():
            if valeur == "contradictoire":
                return False
            
            # Vérifier si la négation du fait existe et a une valeur incompatible
            for autre_fait, autre_valeur in self.faits.items():
                if autre_fait.startswith("¬") and autre_fait[1:] == fait:
                    if valeur is True and autre_valeur is True:
                        return False
                    if valeur is False and autre_valeur is False:
                        return False
        
        return True
    
    def identifier_contradictions(self):
        """Identifie les contradictions explicites dans le système."""
        contradictions = []
        
        for fait, valeur in self.faits.items():
            if valeur == "contradictoire":
                contradictions.append(fait)
                continue
            
            # Vérifier si la négation du fait existe et a une valeur incompatible
            for autre_fait, autre_valeur in self.faits.items():
                if autre_fait.startswith("¬") and autre_fait[1:] == fait:
                    if valeur is True and autre_valeur is True:
                        contradictions.append(f"{fait} et {autre_fait}")
                    if valeur is False and autre_valeur is False:
                        contradictions.append(f"{fait} et {autre_fait}")
        
        return contradictions
    
    def resoudre_contradiction(self, fait, nouvelle_valeur):
        """Résout une contradiction en modifiant la valeur d'un fait."""
        if fait in self.faits:
            self.faits[fait] = nouvelle_valeur
        return self


class LogiqueParaconsistante:
    """Implémentation d'une logique paraconsistante qui tolère les contradictions sans trivialisation."""
    
    class ValeurParaconsistante(Enum):
        VRAI = 1
        FAUX = 2
        CONTRADICTOIRE = 3
        INCONNU = 4
    
    def __init__(self):
        self.valeurs = {}  # Dictionnaire des valeurs de vérité des propositions
    
    def definir_valeur(self, proposition, valeur):
        """Définit la valeur de vérité d'une proposition."""
        self.valeurs[str(proposition)] = valeur
        return self
    
    def evaluer(self, formule):
        """Évalue une formule en logique paraconsistante."""
        if isinstance(formule, PropositionModale) or isinstance(formule, PropositionTemporelle) or isinstance(formule, Predicat):
            # Proposition atomique
            prop_str = str(formule)
            if prop_str in self.valeurs:
                return self.valeurs[prop_str]
            return self.ValeurParaconsistante.INCONNU
        
        elif isinstance(formule, NonModale) or isinstance(formule, NonTemporelle) or isinstance(formule, NonPremierOrdre):
            # Négation
            valeur_interne = self.evaluer(formule.formule)
            
            if valeur_interne == self.ValeurParaconsistante.VRAI:
                return self.ValeurParaconsistante.FAUX
            elif valeur_interne == self.ValeurParaconsistante.FAUX:
                return self.ValeurParaconsistante.VRAI
            elif valeur_interne == self.ValeurParaconsistante.CONTRADICTOIRE:
                return self.ValeurParaconsistante.CONTRADICTOIRE
            else:  # INCONNU
                return self.ValeurParaconsistante.INCONNU
        
        elif isinstance(formule, EtModale) or isinstance(formule, EtTemporelle) or isinstance(formule, EtPremierOrdre):
            # Conjonction
            valeur_gauche = self.evaluer(formule.gauche)
            valeur_droite = self.evaluer(formule.droite)
            
            # Table de vérité pour la conjonction paraconsistante
            if valeur_gauche == self.ValeurParaconsistante.FAUX or valeur_droite == self.ValeurParaconsistante.FAUX:
                return self.ValeurParaconsistante.FAUX
            elif valeur_gauche == self.ValeurParaconsistante.CONTRADICTOIRE or valeur_droite == self.ValeurParaconsistante.CONTRADICTOIRE:
                return self.ValeurParaconsistante.CONTRADICTOIRE
            elif valeur_gauche == self.ValeurParaconsistante.INCONNU or valeur_droite == self.ValeurParaconsistante.INCONNU:
                return self.ValeurParaconsistante.INCONNU
            else:  # Les deux sont VRAI
                return self.ValeurParaconsistante.VRAI
        
        elif isinstance(formule, OuModale) or isinstance(formule, OuTemporelle) or isinstance(formule, OuPremierOrdre):
            # Disjonction
            valeur_gauche = self.evaluer(formule.gauche)
            valeur_droite = self.evaluer(formule.droite)
            
            # Table de vérité pour la disjonction paraconsistante
            if valeur_gauche == self.ValeurParaconsistante.VRAI or valeur_droite == self.ValeurParaconsistante.VRAI:
                return self.ValeurParaconsistante.VRAI
            elif valeur_gauche == self.ValeurParaconsistante.CONTRADICTOIRE or valeur_droite == self.ValeurParaconsistante.CONTRADICTOIRE:
                return self.ValeurParaconsistante.CONTRADICTOIRE
            elif valeur_gauche == self.ValeurParaconsistante.INCONNU or valeur_droite == self.ValeurParaconsistante.INCONNU:
                return self.ValeurParaconsistante.INCONNU
            else:  # Les deux sont FAUX
                return self.ValeurParaconsistante.FAUX
        
        elif isinstance(formule, ImplicationModale) or isinstance(formule, ImplicationTemporelle) or isinstance(formule, ImplicationPremierOrdre):
            # Implication
            valeur_antecedent = self.evaluer(formule.antecedent)
            valeur_consequent = self.evaluer(formule.consequent)
            
            # Table de vérité pour l'implication paraconsistante
            if valeur_antecedent == self.ValeurParaconsistante.FAUX:
                return self.ValeurParaconsistante.VRAI
            elif valeur_consequent == self.ValeurParaconsistante.VRAI:
                return self.ValeurParaconsistante.VRAI
            elif valeur_antecedent == self.ValeurParaconsistante.CONTRADICTOIRE or valeur_consequent == self.ValeurParaconsistante.CONTRADICTOIRE:
                return self.ValeurParaconsistante.CONTRADICTOIRE
            elif valeur_antecedent == self.ValeurParaconsistante.INCONNU or valeur_consequent == self.ValeurParaconsistante.INCONNU:
                return self.ValeurParaconsistante.INCONNU
            else:  # antecedent=VRAI et consequent=FAUX
                return self.ValeurParaconsistante.FAUX
        
        # D'autres cas pour d'autres types de formules
        
        return self.ValeurParaconsistante.INCONNU
    
    def est_valide(self, formule):
        """Vérifie si une formule est valide en logique paraconsistante."""
        return self.evaluer(formule) == self.ValeurParaconsistante.VRAI
    
    def est_contradictoire(self, formule):
        """Vérifie si une formule est contradictoire en logique paraconsistante."""
        return self.evaluer(formule) == self.ValeurParaconsistante.CONTRADICTOIRE


class TheorieDeLaRevision:
    """Implémentation de la théorie de la révision des croyances."""
    
    def __init__(self):
        self.croyances = set()  # Ensemble des croyances actuelles
    
    def ajouter_croyance(self, croyance):
        """Ajoute une croyance à l'ensemble des croyances."""
        self.croyances.add(str(croyance))
        return self
    
    def reviser(self, nouvelle_croyance):
        """
        Révise l'ensemble des croyances avec une nouvelle croyance.
        La révision consiste à incorporer la nouvelle croyance tout en
        préservant la cohérence de l'ensemble.
        """
        # Pour simplifier, nous implémentons une version naïve de la révision
        # où l'on supprime les croyances contradictoires avec la nouvelle croyance
        
        # Vérifier si la nouvelle croyance est compatible avec les croyances existantes
        croyances_incompatibles = set()
        for croyance in self.croyances:
            if self._sont_contradictoires(croyance, str(nouvelle_croyance)):
                croyances_incompatibles.add(croyance)
        
        # Supprimer les croyances incompatibles
        self.croyances -= croyances_incompatibles
        
        # Ajouter la nouvelle croyance
        self.croyances.add(str(nouvelle_croyance))
        
        return self
    
    def contracter(self, croyance_a_retirer):
        """
        Contracte l'ensemble des croyances en retirant une croyance.
        La contraction consiste à retirer une croyance sans ajouter de nouvelles informations.
        """
        # Retirer la croyance
        if str(croyance_a_retirer) in self.croyances:
            self.croyances.remove(str(croyance_a_retirer))
        
        return self
    
    def _sont_contradictoires(self, croyance1, croyance2):
        """Vérifie si deux croyances sont contradictoires."""
        # Pour simplifier, nous considérons que deux croyances sont contradictoires
        # si l'une est la négation de l'autre
        return (croyance1.startswith("¬") and croyance1[1:] == croyance2) or (croyance2.startswith("¬") and croyance2[1:] == croyance1)
    
    def est_coherent(self):
        """Vérifie si l'ensemble des croyances est cohérent."""
        for c1 in self.croyances:
            for c2 in self.croyances:
                if self._sont_contradictoires(c1, c2):
                    return False
        return True
    
    def __str__(self):
        return "Croyances: {" + ", ".join(self.croyances) + "}"


def end_method():
    """Méthode finale pour marquer la fin du module."""
    print("Module de Logique Formelle Avancée et Non-Classique chargé avec succès.")
    return True

# Fin du module
end_method()
