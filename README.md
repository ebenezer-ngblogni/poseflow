<h1 align="center">PoseFlow</h1>
<p align="center"><b>Un seul moteur temps réel de pose et de reconnaissance d'action multi-personnes.<br>Décliné en 4 produits.</b></p>

<p align="center">
<img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-TCN-EE4C2C?logo=pytorch&logoColor=white">
<img alt="Ultralytics" src="https://img.shields.io/badge/YOLO11--pose-ByteTrack-0A0A0A">
<img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-stream-009688?logo=fastapi&logoColor=white">
<img alt="Docker" src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white">
</p>

<p align="center"><img src="docs/gifs/coach.gif" width="70%"></p>

## L'idée

La plupart des projets de pose estimation s'arrêtent à un notebook qui affiche un squelette. J'ai voulu aller plus loin : écrire un vrai cœur d'inférence réutilisable (pose multi-personnes, tracking, analyse temporelle du mouvement) puis le décliner en plusieurs applications concrètes, en ne changeant que la couche métier et l'affichage.

Autrement dit, une petite plateforme produit plutôt qu'un script jetable.

## Les modes

| | Application | Ce qu'il repère |
|:--:|-------------|------------------|
| <img src="docs/icons/safety.svg" width="24"> | **Crowd Safety** · `safety` | Chute, mouvement de panique, bagarre |
| <img src="docs/icons/football.svg" width="24"> | **Sports Analytics** · `football` | Sprint, saut, tacle et chute |
| <img src="docs/icons/fitness.svg" width="24"> | **Coach Fitness** · `fitness` | Comptage de répétitions et contrôle de posture |
| <img src="docs/icons/flow.svg" width="24"> | **People Flow** · `flow` | Comptage, temps d'attente, attroupements |
| <img src="docs/icons/action.svg" width="24"> | **Learned Actions** · `action` | Actions apprises par un réseau temporel (TCN) |

## Reconnaissance d'action en deux étages

C'est le choix d'architecture dont je suis le plus content, parce qu'il rend la démo robuste.

**Étage 1, géométrique et explicable, sans aucun entraînement.**
Chute, course, immobilité, saut ou attroupement se déduisent directement de la géométrie du squelette : angle du torse, proportions du corps, vitesse verticale du bassin, cadence des chevilles, énergie de mouvement. Résultat, la démo fonctionne toujours, même sans GPU ni dataset, et chaque décision reste lisible.

**Étage 2, appris, avec PyTorch.**
Un TCN compact (Temporal Convolutional Network) classe des fenêtres de squelettes normalisés en actions plus fines. Comme l'entrée n'est pas une image mais un squelette (34 valeurs par frame), il s'entraîne en quelques minutes sur une simple GTX ou sur Colab.

```
Vidéo ─▶ YOLO11-pose ─▶ ByteTrack ─▶ buffer temporel ─┬─▶ Étage 1 (règles géométriques)
        (17 keypoints)   (IDs stables)  (fenêtre 32f)   └─▶ Étage 2 (TCN entraîné)
                                                                 │
                                                       overlay + KPIs + alertes
                                                                 │
                                                    CLI  ·  API web  ·  Dashboard
```

## Démos

<p align="center">
<img src="docs/gifs/football.gif" width="88%"><br>
<sub>Mode <code>football</code> : 7 joueurs suivis en simultané, sprint et jog distingués en direct.</sub>
</p>
<p align="center">
<img src="docs/gifs/crowd.gif" width="88%"><br>
<sub>Mode <code>safety</code> : comptage de foule et surveillance des chutes, sans fausse alerte.</sub>
</p>

> Les scènes très denses et de face occultent les jambes : le comptage et les alertes restent fiables, mais le squelette est plus propre sur une vue légèrement en hauteur, façon caméra de surveillance.

## Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Trois façons de tester

**En ligne de commande**, pour annoter une vidéo (sortie `.mp4` plus un journal d'événements `.json`) :
```bash
python apps/cli.py --source samples/match.mp4 --mode football --out match_annot.mp4
python apps/cli.py --source 0 --mode fitness --exercise squat        # webcam
```

**En interface web**, un flux vidéo temps réel dans le navigateur avec sélecteur de mode :
```bash
uvicorn apps.api:app --port 8000        # http://localhost:8000
```

**En dashboard**, pour déposer une vidéo, choisir un mode et suivre les courbes en direct :
```bash
streamlit run apps/dashboard.py         # http://localhost:8501
```

## Entraîner le modèle de l'étage 2

On range des clips par action, on extrait les squelettes, puis on entraîne le TCN :

```
data/
 ├── running/  clip1.mp4 ...
 ├── falling/  ...
 └── waving/   ...
```
```bash
python train/extract.py --data data --out dataset.npz
python train/train.py   --data dataset.npz --out weights/action.pt --epochs 40
python apps/cli.py --source clip.mp4 --mode action --out out.mp4
```

Sans dataset sous la main, `train/train.py` s'entraîne sur un petit jeu synthétique intégré, histoire que la chaîne complète tourne du premier coup.

## Déploiement

```bash
docker compose up --build       # api sur :8000, dashboard sur :8501
```

Une seule image, deux services, prête à tourner sur un VPS comme n'importe quel service ML en production.

## Stack

`YOLO11-pose` · `ByteTrack` · `PyTorch` (TCN) · `OpenCV` · `NumPy` · `FastAPI` · `Uvicorn` · `Streamlit` · `Plotly` · `Docker`

## Structure

```
poseflow/
 ├── engine/     pose (YOLO11), tracking, buffer temporel, features géométriques
 ├── actions/    heuristiques (étage 1), modèle TCN et classifieur (étage 2)
 ├── modes/      safety, football, fitness, flow, action
 ├── render.py   overlay squelette, HUD, bannières d'alerte
 └── pipeline.py orchestration
apps/    cli, api (FastAPI), dashboard (Streamlit)
train/   extract (clips vers squelettes), train (TCN)
```

## Limites et suite

Les détecteurs de l'étage 1 sont volontairement simples et explicables, l'étage 2 vient les affiner. La précision en foule très dense dépend du modèle de pose choisi (de `yolo11n` à `yolo11x`). Prochaines étapes envisagées : ré-identification inter-caméras, ST-GCN complet, export ONNX/TensorRT, et une évaluation chiffrée sur PoseTrack.

<br>

<p align="center">
<b>Eben Ezer J.N. NGBLOGNI</b>, Ingénieur Data &amp; IA<br>
<a href="https://linkedin.com/in/eben-ngb">LinkedIn</a> ·
<a href="https://github.com/ebenezer-ngblogni">GitHub</a> ·
<a href="https://eben-ezer.site">eben-ezer.site</a>
</p>
