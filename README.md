multicam_annotator

Prototype d'un annotateur multi-caméras (Python).

But
- Afficher jusqu'à 6 vidéos synchronisées (format .avi ou mp4).
- Permettre une annotation simple (labels par frame) et exporter en Excel.

Prérequis
- Python 3.8+
- Windows (pour l'export en .exe avec PyInstaller) ou autre OS si vous préférez garder en script.

Installation
1. Créer et activer un environnement virtuel:

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
```

2. Installer les dépendances:

```powershell
pip install -r requirements.txt
```

Usage (prototype)
1. Lancer l'application:

```powershell
python src/main.py
```

2. Dans l'UI: "Charger 6 vidéos" (sélectionnez 1 à 6 fichiers), Play/Pause, saisir un label et "Ajouter label (cam active)", puis "Exporter annotations (Excel)".

Générer un .exe (optionnel)
- Installer PyInstaller:

```powershell
pip install pyinstaller
```

- Construire l'exécutable (single-file, console désactivée):

```powershell
pyinstaller --noconsole --onefile --add-data "venv\Lib\site-packages\PySide6;PySide6" src\main.py
```

Remarques
- Pour utiliser le GPU avec OpenCV vous devez installer une build d'OpenCV compilée avec CUDA; sinon l'app utilisera le CPU.
- Le prototype contient un cache mémoire LRU pour améliorer les performances lors de seek/rewind, mais il n'est pas optimisé pour des vidéos HD longues à 120 Hz — vous devrez ajuster `cache_size` dans `VideoWorker`.
- Prochaines améliorations possibles: support natif de dossiers multi-caméras, visualiseur timeline, lecture audio synchronisée, détection automatique du framerate et interpolation temporelle entre flux.
