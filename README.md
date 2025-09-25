# 🎥 Multicam Annotator

Un prototype simple et pratique pour annoter plusieurs caméras synchronisées (jusqu'à 6) — lecture, avance/recul image par image, catalogue de labels et export Excel.

Ce dépôt contient une application PySide6 + OpenCV. L'objectif : permettre à des annotateurs non techniques d'ouvrir plusieurs vidéos, naviguer frame-par-frame et exporter les labels.

---

## 🚀 Points forts

- Lecture synchronisée de plusieurs vidéos (jusqu'à 6 en tuiles, les autres s'ouvrent en popup)
- Contrôles standards : play / pause / seek / prev / next frame
- Molette souris pour avancer/reculer d'1 frame (Shift ×10)
- Popups par caméra avec zoom & pan synchronisés
- Catalogue de labels modifiable (JSON) et export Excel (ANNOTATIONS + METADATA)

---

## 🧰 Prérequis

- Windows / macOS / Linux
- Python 3.8+
- Virtualenv recommandé

Fichiers importants :
- `src/main.py` — application principale (prototype)
- `requirements.txt` — dépendances Python

---

## ⚡ Installation rapide (Windows PowerShell)

1) Créez et activez un environnement virtuel

```powershell
python -m venv .venv; .venv\Scripts\Activate.ps1
```

2) Installez les dépendances

```powershell
pip install -r requirements.txt
```

3) Lancez l'application

```powershell
python src/main.py
```

---

## 🧭 Utilisation — modes rapides

- Cliquez sur "Charger vidéos" et sélectionnez vos fichiers (1..N). Les 6 premières apparaissent en tuiles.
- Double-clic ou clic droit sur une tuile : ouvre la popup zoom/pan pour cette caméra.
- Molette de la souris sur une tuile : avance/recul d'un frame (ajoutez Shift pour ×10).
- Saisissez un label dans le panneau gauche puis cliquez "Ajouter" pour l'attacher au frame courant (il est ajouté pour toutes les caméras à ce frame).
- "Exporter" génère un fichier Excel (`ANNOTATIONS` + `METADATA`).

---

## 📦 Créer un .exe (Windows) — guide pour débutant

Si vous souhaitez distribuer un exécutable Windows, PyInstaller est une option simple.

1) Installer PyInstaller dans le même environnement :

```powershell
pip install pyinstaller
```

2) Construire l'exécutable (exemple simple) :

```powershell
pyinstaller --noconsole --onefile --add-data "venv\Lib\site-packages\PySide6;PySide6" src\main.py
```

Notes/astuces:
- L'option `--onefile` produit un seul .exe mais peut allonger le démarrage.
- L'ajout `--add-data` pour PySide6 dépend de l'emplacement d'installation; adaptez le chemin si votre virtualenv est ailleurs.
- Après compilation, retrouvez l'exécutable dans `dist\main.exe` (ou `dist\src` selon le nom). Testez sur une autre machine où Python n'est pas installé.

Pour une distribution plus robuste (icône, présence de DLLs, tests) je peux préparer un spec PyInstaller adapté et vérifier les ressources PySide6.

---

## 🧪 Tests rapides

- Tester sur vidéos courtes (quelques secondes) pour valider la synchro.
- Vérifier l'export Excel et ouvrir le fichier pour confirmer la feuille `METADATA` contient les noms de fichiers et `fps_used`.

---

## 📁 Structure du projet

- `src/main.py` — code principal (interface + worker OpenCV)
- `requirements.txt` — dépendances Python
- `label_catalog.json` (optionnel) — catalogue de labels

---

## ❓ FAQ rapide

Q — J'ai un crash lié à OpenCV/ffmpeg

R — Vérifiez `requirements.txt`. Si OpenCV (`cv2`) n'est pas installé proprement, vous pouvez installer `opencv-python` (CPU). Pour GPU, installez une build CUDA (avancée).

Q — Les popups s'ouvrent off-screen

R — Faites `Alt+Space` sur la fenêtre (Windows) pour la ramener ou fermez-la et relancez l'app; je peux également forcer la position initiale de la popup si besoin.

---

## ✨ Prochaines étapes (si vous voulez que je m'en occupe)

- Nettoyage/formatage du code et suppression des fallbacks obsolètes
- Verifier et consolider `requirements.txt` (versions exactes)
- Préparer un spec PyInstaller et tester la génération d'un .exe reproductible
- Ajouter un mini-guide vidéo / screenshots pour l'interface

Dites-moi laquelle de ces tâches vous voulez que je fasse en priorité — je peux commencer par fixer `requirements.txt` pour garantir que `pip install -r requirements.txt` installe tout proprement, puis préparer le .exe.
