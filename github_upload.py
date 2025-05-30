import os
import subprocess
import sys
from datetime import datetime

def run_command(command, description):
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erreur: {e}")
        print(f"Sortie de commande: {e.stdout}")
        print(f"Erreur de commande: {e.stderr}")
        return False

def create_version_tag():
    """Crée un tag de version basé sur la date actuelle."""
    current_date = datetime.now()
    version_tag = f"v{current_date.year}.{current_date.month:02d}.{current_date.day:02d}"
    
    # Vérifier si le tag existe déjà
    try:
        result = subprocess.run(f"git tag -l {version_tag}", shell=True, capture_output=True, text=True)
        if version_tag in result.stdout:
            # Si le tag existe, ajouter un suffixe avec l'heure
            version_tag = f"v{current_date.year}.{current_date.month:02d}.{current_date.day:02d}-{current_date.hour:02d}{current_date.minute:02d}"
    except:
        pass
    
    return version_tag

def main():
    project_path = os.path.dirname(os.path.abspath(__file__))
    repo_url = "https://github.com/univers-artifficial-intelligence/Project-AGI-ASI-AI-google-gemini-2.0-flash-update1.git"
    
    print(f"Déploiement du projet dans le dossier: {project_path}")
    print(f"Vers le dépôt GitHub: {repo_url}")
    
    os.chdir(project_path)
    
    # Initialisation du dépôt Git
    if not run_command("git init", "Initialisation du dépôt Git"):
        return
    
    # Ajout des fichiers au suivi Git
    if not run_command("git add --all", "Ajout de tous les fichiers au suivi Git"):
        return
    
    # Création du commit initial
    commit_message = "Upload complet du projet AGI-ASI-AI avec nouveau module de raisonnement"
    commit_result = subprocess.run(f'git commit -m "{commit_message}"', shell=True, capture_output=True, text=True)
    
    # Vérifier si le commit a échoué parce qu'il n'y a rien à committer
    if "nothing to commit" in commit_result.stderr or "nothing to commit" in commit_result.stdout:
        print("Aucun changement détecté, tous les fichiers sont déjà commités. Poursuite du déploiement...")
    elif commit_result.returncode != 0:
        # Si une autre erreur s'est produite
        print(f"Erreur lors du commit: {commit_result.stderr}")
        return
    
    # Création d'un tag de version pour permettre les releases
    version_tag = create_version_tag()
    if not run_command(f'git tag -a {version_tag} -m "Release {version_tag} - Ajout module de raisonnement analogique"', 
                      f"Création du tag de version {version_tag}"):
        print("Attention: Impossible de créer le tag, mais on continue...")
    
    # Configuration du dépôt distant
    # Vérifier si le remote origin existe déjà
    try:
        subprocess.run("git remote get-url origin", shell=True, check=True, capture_output=True, text=True)
        print("Le remote 'origin' existe déjà. Suppression pour reconfiguration...")
        run_command("git remote remove origin", "Suppression du remote existant")
    except subprocess.CalledProcessError:
        pass  # Le remote n'existe pas, continue normalement
    
    if not run_command(f'git remote add origin {repo_url}', "Configuration du dépôt distant"):
        return
    
    # Détermination de la branche par défaut
    print("\nDétection de la branche par défaut...")
    default_branch = "main"  # La plupart des nouveaux dépôts utilisent "main" comme branche par défaut
    
    # Envoi du code vers GitHub
    if not run_command(f"git push -u origin {default_branch}", f"Envoi du code vers la branche {default_branch}"):
        print("\nTentative de push vers la branche 'master' à la place...")
        if not run_command("git push -u origin master", "Envoi du code vers la branche master"):
            print("\nÉchec du déploiement. Vérifiez vos identifiants GitHub et les paramètres du dépôt.")
            return
        default_branch = "master"
    
    # Envoi des tags vers GitHub
    if not run_command("git push origin --tags", "Envoi des tags vers GitHub"):
        print("Attention: Impossible d'envoyer les tags, mais le code a été transféré.")
    
    print(f"\nTerminé! Votre code a été transféré avec succès vers GitHub.")
    print(f"URL du dépôt: {repo_url}")
    print(f"Tag créé: {version_tag}")
    print(f"\nPour créer une release sur GitHub:")
    print(f"1. Allez sur {repo_url}/releases")
    print(f"2. Cliquez sur 'Create a new release'")
    print(f"3. Sélectionnez le tag '{version_tag}'")
    print(f"4. Ajoutez un titre et une description pour votre release")

if __name__ == "__main__":
    main()
    input("\nAppuyez sur Entrée pour quitter...")
