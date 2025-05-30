"""
Configuration d'authentification ngrok pour GeminiChat
Ce fichier contient le token d'authentification ngrok pour éviter d'avoir à le saisir manuellement.
"""

# Token d'authentification ngrok
NGROK_AUTH_TOKEN = "2xXewKQix1t5vISm9PGl717DJ9z_3dTRwXTQU73m3LK3aWaHQ"

def get_auth_token():
    """
    Retourne le token d'authentification ngrok
    """
    return NGROK_AUTH_TOKEN
