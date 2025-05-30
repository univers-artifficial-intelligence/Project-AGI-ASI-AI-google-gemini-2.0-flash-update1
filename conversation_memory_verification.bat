@echo off
echo Verification de la coherence des conversations...
echo.

echo Execution du test de memoire...
python -m tests.test_memory_retrieval_enhancer

echo.
echo Test termine. Verifiez les resultats ci-dessus.
pause
