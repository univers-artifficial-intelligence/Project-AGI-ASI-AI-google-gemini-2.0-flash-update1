# Améliorations pour l'Analyse d'Images Astronomiques

## Problèmes identifiés

1. **Descriptions répétitives** : L'IA commence systématiquement ses analyses d'images astronomiques par "L'image montre une carte du ciel nocturne avec plusieurs constellations..." sans tenir compte des spécificités de chaque image.

2. **Manque de variété** : Les descriptions des cartes célestes sont trop similaires d'une image à l'autre, ce qui rend les conversations monotones.

3. **Ignorance du contexte de la question** : L'IA donne une description générale au lieu de se focaliser sur l'élément spécifique mentionné dans la question de l'utilisateur.

## Solutions implémentées

### 1. Module `gemini_api.py`

- Ajout d'instructions spécifiques pour les images astronomiques dans le prompt système
- Instructions pour éviter les formulations répétitives
- Directives pour se concentrer sur les éléments spécifiques et uniques de chaque image
- Exigence d'adapter la réponse à la question posée

```python
IMAGES ASTRONOMIQUES: Pour les images de cartes célestes, constellations ou ciel nocturne:
1. ÉVITE ABSOLUMENT les formulations répétitives du type "L'image montre une carte du ciel nocturne avec plusieurs constellations..."
2. Concentre-toi sur les ÉLÉMENTS SPÉCIFIQUES de CETTE image particulière (constellations précises, planètes, position de la lune, etc.)
3. Adapte ta réponse à la QUESTION POSÉE plutôt que de faire une description générique
4. Mentionne les caractéristiques uniques ou intéressantes de l'image (alignements particuliers, phénomènes visibles, etc.)
5. Utilise tes connaissances en astronomie pour donner des explications pertinentes et variées
```

### 2. Module `conversation_context_manager.py`

- Ajout de patterns de détection spécifiques pour les réponses d'analyse d'images astronomiques
- Amélioration de la fonction `detect_image_analysis` pour mieux identifier les contenus astronomiques

```python
# Mots-clés spécifiques aux images astronomiques
astro_keywords = [
    r"(?i)(constellation[s]? (de|du|des))",
    r"(?i)(carte (du|céleste|du ciel))",
    r"(?i)(ciel nocturne)",
    r"(?i)(étoile[s]? (visible|brillante|nommée))",
    r"(?i)(position (de la|des) (lune|planète|étoile))",
    r"(?i)(trajectoire (de|des|du))",
]
```

### 3. Module `emotional_engine.py`

- Amélioration de la fonction `is_image_analysis_request` pour mieux détecter les requêtes d'analyse d'images basées sur le contexte
- Ajout de mots-clés pour identifier les demandes d'analyse d'image

```python
# Vérifier les mots-clés dans la requête qui suggèrent l'analyse d'une image
if 'message' in request_data and isinstance(request_data['message'], str):
    image_request_keywords = [
        r"(?i)(analyse[r]? (cette|l'|l'|une|des|la) image)",
        r"(?i)(que (vois|voit|montre|représente)-tu (sur|dans) (cette|l'|l'|une|des|la) image)",
        r"(?i)(que peux-tu (me dire|dire) (sur|à propos de|de) (cette|l'|l'|une|des|la) image)",
        r"(?i)(décri[s|re|vez] (cette|l'|l'|une|des|la) image)",
        r"(?i)(explique[r|z]? (cette|l'|l'|une|des|la) image)",
    ]
```

## Résultats attendus

- Les analyses d'images astronomiques seront désormais plus variées et personnalisées
- L'IA se concentrera sur les éléments spécifiques mentionnés dans la question de l'utilisateur
- Chaque image recevra une analyse unique adaptée à son contenu réel
- Les conversations seront plus naturelles et interactives

## Tests

Pour vérifier que les modifications sont efficaces:
1. Présenter plusieurs images astronomiques différentes à l'IA
2. Vérifier que les descriptions ne sont pas répétitives
3. Poser des questions spécifiques sur certains éléments pour tester si l'IA adapte sa réponse
