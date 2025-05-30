@echo off
echo Commandes à exécuter dans le terminal pour transférer votre projet vers GitHub
echo Copiez et collez ces commandes une par une dans votre terminal

echo.
echo Initialisation du dépôt Git...
git init

echo.
echo Ajout de TOUS les fichiers au suivi Git...
git add -A

echo.
echo Vérification des fichiers qui seront commités...
git status

echo.
echo Création du commit initial...
git commit -m "Upload complet du projet AGI-ASI-AI"

echo.
echo Configuration du dépôt distant...
git remote add origin https://github.com/univers-artifficial-intelligence/Project-AGI-ASI-AI-google-gemini-2.0-flash4.git

echo.
echo Envoi du code vers GitHub...
git push -u origin main

echo.
echo Si l'envoi échoue, essayez avec l'option force:
echo git push -u -f origin main

echo.
echo Note: Si votre branche principale s'appelle "master" au lieu de "main", utilisez:
echo git push -u origin master

echo.
echo Terminé! Votre code a été transféré vers GitHub.
pause
