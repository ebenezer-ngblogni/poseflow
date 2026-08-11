<h1 align="center">🎯 PoseFlow</h1>
<p align="center"><b>Un seul moteur temps réel de pose + reconnaissance d'action multi-personnes.<br>Décliné en 4 produits.</b></p>

<p align="center">
<img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-TCN-EE4C2C?logo=pytorch&logoColor=white">
<img alt="Ultralytics" src="https://img.shields.io/badge/YOLO11--pose-ByteTrack-0A0A0A">
<img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-stream-009688?logo=fastapi&logoColor=white">
<img alt="Docker" src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white">
</p>

<p align="center"><img src="docs/gifs/coach.gif" width="70%"></p>

---

## Le concept

La plupart des projets de *human pose estimation* s'arrêtent à un notebook qui affiche un squelette. **PoseFlow va plus loin** : un **cœur d'inférence réutilisable** (pose multi-personnes → tracking → analyse temporelle de l'action) que je décline en **plusieurs applications métier** en ne changeant que la couche « logique + overlay + KPIs ».

> C'est le pattern d'une vraie plateforme produit, pas d'un script jetable.

| Mode | Application | Ce qu'il détecte |
|------|-------------|------------------|
| 🛡️ `safety` | **Crowd Safety** | Chute, mouvement de panique, bagarre |
| ⚽ `football` | **Sports Analytics** | Sprint, saut, tacle / chute |
| 🏋️ `fitness` | **Coach Fitness** | Comptage de répétitions + contrôle de posture |
| 🚶 `flow` | **People Flow** | Comptage, temps d'attente, attroupements |
| 🧠 `action` | **Learned Actions** | Actions apprises par réseau temporel (TCN) |

---

## Reconnaissance d'action à 2 étages

Le point clé de l'architecture — et ce qui rend la démo **robuste** :

**① Tier-1 — géométrique & explicable** (zéro entraînement)
Chute, course, immobilité, saut, attroupement sont dérivés directement de la géométrie du squelette : angle du torse, ratio de la boîte, vitesse verticale du bassin, cadence des chevilles, énergie de mouvement. → *La démo marche toujours, même sans GPU ni dataset.*

**② Tier-2 — appris** (`PyTorch`)
Un **TCN compact** (Temporal Convolutional Network) classe des fenêtres de squelettes normalisés en actions fines. L'entrée étant des squelettes (34 valeurs/frame, pas des images), il s'entraîne **en minutes sur une simple GTX ou sur Colab**.

```
Vidéo ─▶ YOLO11-pose ─▶ ByteTrack ─▶ buffer temporel ─┬─▶ Tier-1 (règles géométriques)
        (17 keypoints)   (IDs stables)  (fenêtre 32f)   └─▶ Tier-2 (TCN entraîné)
                                                                 │
                                                       overlay + KPIs + alertes
                                                                 │
                                                    CLI  ·  API (MJPEG)  ·  Dashboard
```

---

## Démarrage rapide

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**CLI — annoter une vidéo** (produit un `.mp4` + un journal d'événements `.json`)
```bash
python apps/cli.py --source samples/people-detection.mp4 --mode safety --out out.mp4
python apps/cli.py --source 0 --mode fitness --exercise squat        # webcam
python apps/cli.py --source match.mp4 --mode football --out match_annot.mp4
```

**API temps réel** (flux MJPEG + endpoints `/health`, `/state`)
```bash
uvicorn apps.api:app --port 8000     # http://localhost:8000
```

**Dashboard interactif** (Streamlit + courbes Plotly)
```bash
streamlit run apps/dashboard.py      # http://localhost:8501
```

---

## Entraîner le modèle Tier-2

Organise des clips par action, puis extrais les squelettes et entraîne le TCN :

```
data/
 ├── running/  clip1.mp4 …
 ├── falling/  …
 └── waving/   …
```
```bash
python train/extract.py --data data --out dataset.npz     # squelettes -> fenêtres
python train/train.py   --data dataset.npz --out weights/action.pt --epochs 40
python apps/cli.py --source clip.mp4 --mode action --out out.mp4
```

> Sans dataset, `train/train.py` s'entraîne sur un jeu **synthétique** intégré : la chaîne est exécutable immédiatement (`val_acc` affichée).

---

## Déploiement

```bash
docker compose up --build     # api :8000  ·  dashboard :8501
```
Conçu pour un déploiement VPS/cloud identique à un service ML de production (image unique, deux services).

---

## Stack technique

`YOLO11-pose` · `ByteTrack` · `PyTorch` (TCN) · `OpenCV` · `NumPy` · `FastAPI` · `Uvicorn` · `Streamlit` · `Plotly` · `Docker`

## Structure

```
poseflow/
 ├── engine/     pose (YOLO11) · tracking · buffer temporel · features géométriques
 ├── actions/    heuristiques Tier-1 · modèle TCN + classifieur Tier-2
 ├── modes/      safety · football · fitness · flow · action
 ├── render.py   overlay squelette + HUD + bannières d'alerte
 └── pipeline.py orchestration
apps/    cli · api (FastAPI) · dashboard (Streamlit)
train/   extract (clips -> squelettes) · train (TCN)
```

## Limites & pistes

- Les détecteurs Tier-1 sont volontairement simples et explicables ; le Tier-2 les affine.
- La précision en foule très dense dépend du modèle de pose choisi (`yolo11n → x`).
- Roadmap : ré-identification inter-caméras, ST-GCN complet, export ONNX/TensorRT, évaluation quantitative sur PoseTrack.

---

<p align="center">
<b>Eben Ezer J.N. NGBLOGNI</b> — Ingénieur Data &amp; IA<br>
<a href="https://linkedin.com/in/eben-ngb">LinkedIn</a> ·
<a href="https://github.com/ebenezer-ngblogni">GitHub</a> ·
<a href="https://eben-ezer.site">eben-ezer.site</a>
</p>
