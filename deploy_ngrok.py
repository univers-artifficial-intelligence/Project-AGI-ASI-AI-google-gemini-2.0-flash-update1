import os
import sys
import time
import signal
import logging
import subprocess
import threading
import webbrowser
import json
import tempfile
import requests
import atexit
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from importlib import reload
import app  # L'application principale

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('ngrok-deployer')

# Variables globales
ngrok_process = None
flask_process = None
observer = None
ngrok_url = None
ngrok_config_file = os.path.join(tempfile.gettempdir(), 'ngrok_config.yml')
current_directory = os.path.dirname(os.path.abspath(__file__))

# Token ngrok
NGROK_AUTH_TOKEN = " "

# Classe pour gérer la surveillance des fichiers
class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, restart_function):
        self.restart_function = restart_function
        self.last_modified = time.time()
        self.cooldown = 2  # Temps de refroidissement en secondes

    def on_modified(self, event):
        # Ignorer les fichiers cachés et les répertoires
        if event.src_path.startswith(".") or not event.src_path.endswith(('.py', '.html', '.css', '.js')):
            return
        
        current_time = time.time()
        if current_time - self.last_modified > self.cooldown:
            self.last_modified = current_time
            logger.info(f"Changement détecté dans {event.src_path}")
            self.restart_function()

def create_ngrok_config():
    """Crée un fichier de configuration temporaire pour ngrok"""
    config = {
        "version": "2",
        "authtoken": NGROK_AUTH_TOKEN,  # Utilisation du token ajouté
        "web_addr": "localhost:4040",
        "tunnels": {
            "geminichat": {
                "proto": "http",
                "addr": "5000",
                "inspect": True
            }
        }
    }
    
    with open(ngrok_config_file, 'w') as f:
        json.dump(config, f)
    
    return ngrok_config_file

def setup_ngrok_auth():
    """Configure l'authentification ngrok avec le token"""
    try:
        # Vérifier si ngrok est déjà authentifié
        auth_process = subprocess.run(
            ['ngrok', 'config', 'check'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Si le token n'est pas configuré ou s'il est différent, on l'ajoute
        if b"authtoken" not in auth_process.stdout and b"authtoken" not in auth_process.stderr:
            logger.info("🔑 Configuration du token ngrok...")
            auth_result = subprocess.run(
                ['ngrok', 'config', 'add-authtoken', NGROK_AUTH_TOKEN],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            if auth_result.returncode == 0:
                logger.info("✅ Token ngrok configuré avec succès")
            else:
                logger.error(f"❌ Erreur lors de la configuration du token ngrok: {auth_result.stderr.decode('utf-8')}")
        else:
            logger.info("✅ ngrok est déjà authentifié")
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de la configuration du token ngrok: {str(e)}")

def start_ngrok():
    """Démarre ngrok avec la configuration optimisée"""
    global ngrok_process, ngrok_url
    
    # D'abord, on s'assure que l'authentification est bien configurée
    setup_ngrok_auth()
    
    config_path = create_ngrok_config()
    
    try:
        # Vérifier si ngrok est installé
        if sys.platform == 'win32':
            ngrok_cmd = ['ngrok', 'http', '--config', config_path, '5000']
        else:
            ngrok_cmd = ['ngrok', 'http', '--config', config_path, '5000']
        
        ngrok_process = subprocess.Popen(
            ngrok_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Attendre que ngrok soit prêt et récupérer l'URL
        time.sleep(3)  # Donner du temps à ngrok pour démarrer
        
        try:
            response = requests.get('http://localhost:4040/api/tunnels')
            tunnels = response.json()['tunnels']
            for tunnel in tunnels:
                if tunnel['proto'] == 'https':
                    ngrok_url = tunnel['public_url']
                    break
            
            if ngrok_url:
                logger.info(f"🚀 Site déployé avec succès sur: {ngrok_url}")
                # Ouvrir le navigateur vers l'URL
                webbrowser.open(ngrok_url)
            else:
                logger.error("❌ Impossible de récupérer l'URL ngrok")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération de l'URL ngrok: {str(e)}")
    
    except FileNotFoundError:
        logger.error("❌ ngrok n'est pas installé ou n'est pas dans le PATH")
        logger.info("📢 Installez ngrok depuis https://ngrok.com/download et ajoutez-le à votre PATH")
        sys.exit(1)

def start_flask():
    """Démarre le serveur Flask avec des optimisations"""
    global flask_process
    
    # Variables d'environnement pour optimisations
    env = os.environ.copy()
    env['FLASK_ENV'] = 'production'
    env['FLASK_APP'] = 'app.py'
    
    # Configurer pour utiliser le mode production mais avec le rechargement activé
    flask_cmd = [sys.executable, '-m', 'flask', 'run', '--host=0.0.0.0', '--port=5000', '--reload']
    
    logger.info("🌐 Démarrage du serveur Flask...")
    flask_process = subprocess.Popen(
        flask_cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Thread pour afficher les logs de Flask
    def log_output(pipe, prefix):
        for line in iter(pipe.readline, b''):
            logger.info(f"{prefix} {line.decode('utf-8').strip()}")
    
    threading.Thread(target=log_output, args=(flask_process.stdout, "📊 Flask:"), daemon=True).start()
    threading.Thread(target=log_output, args=(flask_process.stderr, "⚠️ Flask:"), daemon=True).start()

def restart_server():
    """Redémarre le serveur Flask"""
    global flask_process
    
    logger.info("🔄 Redémarrage du serveur Flask...")
    
    # Arrêter le processus existant
    if flask_process:
        if sys.platform == 'win32':
            # Pour Windows, utiliser taskkill
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(flask_process.pid)], 
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # Pour Unix
            os.kill(flask_process.pid, signal.SIGTERM)
            flask_process.wait()
    
    # Recharger le module d'application
    try:
        reload(app)
        logger.info("📚 Module app.py rechargé")
    except Exception as e:
        logger.error(f"❌ Erreur lors du rechargement du module: {str(e)}")
    
    # Redémarrer Flask
    start_flask()
    logger.info("✅ Serveur Flask redémarré")

def start_file_watcher():
    """Démarre la surveillance des fichiers pour le rechargement à chaud"""
    global observer
    
    event_handler = FileChangeHandler(restart_server)
    observer = Observer()
    
    # Observer le dossier principal et les sous-dossiers
    paths_to_watch = [
        current_directory,
        os.path.join(current_directory, 'templates'),
        os.path.join(current_directory, 'static')
    ]
    
    for path in paths_to_watch:
        if os.path.exists(path):
            observer.schedule(event_handler, path, recursive=True)
    
    observer.start()
    logger.info("👁️ Surveillance des fichiers activée pour le rechargement à chaud")

def cleanup():
    """Nettoyer les processus au moment de quitter"""
    logger.info("🧹 Nettoyage des processus...")
    
    if ngrok_process:
        ngrok_process.terminate()
        ngrok_process.wait()
    
    if flask_process:
        if sys.platform == 'win32':
            # Pour Windows
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(flask_process.pid)], 
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # Pour Unix
            os.kill(flask_process.pid, signal.SIGTERM)
            flask_process.wait()
    
    if observer:
        observer.stop()
        observer.join()
    
    # Supprimer le fichier de configuration temporaire
    if os.path.exists(ngrok_config_file):
        try:
            os.remove(ngrok_config_file)
        except:
            pass

def check_requirements():
    """Vérifier et installer les dépendances si nécessaire"""
    required_packages = [
        'flask', 'flask_compress', 'watchdog', 'requests',
        'pyngrok', 'flask_ngrok',
    ]
    
    logger.info("🔍 Vérification des dépendances...")
    
    # Vérifier les packages installés
    try:
        import pkg_resources
        installed = {pkg.key for pkg in pkg_resources.working_set}
        missing = [pkg for pkg in required_packages if pkg.lower() not in installed]
        
        if missing:
            logger.info(f"📦 Installation des packages manquants: {', '.join(missing)}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
            logger.info("✅ Packages installés avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur lors de la vérification/installation des packages: {str(e)}")
        sys.exit(1)

def main():
    """Fonction principale"""
    logger.info("🚀 Démarrage du déploiement GeminiChat avec ngrok")
    
    # Vérifier et installer les dépendances
    check_requirements()
    
    # Enregistrer la fonction de nettoyage
    atexit.register(cleanup)
    
    try:
        # Démarrer Flask
        start_flask()
        time.sleep(2)  # Attendre que Flask démarre
        
        # Démarrer ngrok avec authentification
        start_ngrok()
        
        # Démarrer la surveillance des fichiers
        start_file_watcher()
        
        logger.info("✅ Système déployé et en cours d'exécution")
        logger.info("🌐 Interface d'administration ngrok: http://localhost:4040")
        
        # Montrer comment arrêter le serveur
        logger.info("💡 Appuyez sur Ctrl+C pour arrêter le serveur")
        
        # Maintenir le script en cours d'exécution
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("👋 Arrêt du serveur...")
    
    except Exception as e:
        logger.error(f"❌ Erreur: {str(e)}")
    
    finally:
        cleanup()

if __name__ == "__main__":
    main()
