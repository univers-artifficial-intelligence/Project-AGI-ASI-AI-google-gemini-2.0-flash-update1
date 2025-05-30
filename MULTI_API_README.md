# Implémentation Multi-API

Cette extension permet d'utiliser différentes API d'intelligence artificielle au sein de GeminiChat.

## Compatibilité API

GeminiChat supporte maintenant plusieurs fournisseurs d'API d'IA :

- **Google Gemini** (par défaut) - API Gemini 2.0 Flash
- **Claude by Anthropic** - Support pour Claude-3 Opus

D'autres fournisseurs peuvent être facilement ajoutés en implémentant l'interface `AIApiInterface`.

## Architecture

Le système utilise une architecture modulaire avec les composants suivants :

- `ai_api_interface.py` - Interface abstraite que toutes les implémentations d'API doivent suivre
- `ai_api_manager.py` - Gestionnaire centralisant l'accès aux différentes API d'IA
- `gemini_api_adapter.py` - Implémentation de l'interface pour Google Gemini
- `claude_api_adapter.py` - Implémentation de l'interface pour Claude by Anthropic

## Configuration

### Interface utilisateur

Une interface utilisateur est disponible pour configurer les API :
1. Connectez-vous à votre compte GeminiChat
2. Cliquez sur "Config API" dans le menu de navigation
3. Pour chaque API :
   - Entrez votre clé API
   - Cliquez sur "Enregistrer la clé"
   - Cliquez sur "Activer cette API" pour l'utiliser

### Configuration par fichier

Vous pouvez également configurer les API via le fichier `ai_api_config.json` :

```json
{
    "default_api": "gemini",
    "apis": {
        "gemini": {
            "api_key": "votre_clé_api_gemini",
            "api_url": null
        },
        "claude": {
            "api_key": "votre_clé_api_claude",
            "api_url": null
        }
    }
}
```

## Ajouter une nouvelle API

Pour ajouter le support d'une nouvelle API d'IA :

1. Créez une nouvelle classe implémentant `AIApiInterface`
2. Enregistrez cette classe auprès du `AIApiManager`
3. Mettez à jour la configuration pour inclure les paramètres de la nouvelle API

Exemple d'enregistrement d'une nouvelle API :

```python
from my_new_api_adapter import MyNewAPI
from ai_api_manager import get_ai_api_manager

api_manager = get_ai_api_manager()
api_manager.add_api_implementation('my_new_api', MyNewAPI)
```

## API REST pour la gestion des API

Le système expose plusieurs endpoints REST pour gérer les API :

- `GET /api/config/apis` - Liste des API disponibles
- `GET /api/config/apis/current` - API actuellement active
- `POST /api/config/apis/current` - Changer l'API active
- `GET /api/keys` - Liste des clés API configurées
- `POST /api/keys/{api_name}` - Configurer une clé API
- `DELETE /api/keys/{api_name}` - Supprimer une clé API
