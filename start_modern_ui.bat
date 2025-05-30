@echo off
echo Démarrage de Gemini 2.0 Flash AGI avec l'interface moderne...
echo.

REM Sauvegarder l'app.py original si nécessaire
if not exist app.py.bak (
    echo Sauvegarde de l'application originale...
    copy app.py app.py.bak
)

REM Utiliser le fichier app moderne
echo Activation de l'interface moderne...
copy app-modern.py app.py

echo.
echo Démarrage du serveur web...
python app.py

REM Restaurer l'app.py original à la fin
echo.
echo Restauration de la configuration originale...
copy app.py.bak app.py

echo.
echo Serveur arrêté. Au revoir!
pause
