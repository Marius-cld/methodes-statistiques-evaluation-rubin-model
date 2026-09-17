# Méthodes Statistiques de l'Évaluation

Exercices réalisés dans le cadre du cours *Méthodes Statistiques de l'Évaluation* (Master 2), en deux
chapitres : le modèle de Rubin (biais de sélection et d'hétérogénéité) et les méthodes de
Difference-in-Differences (panel, étude d'événement).

## Structure du dépôt

```
.
├── notebooks/
│   ├── chapitre-1-analyse-biais-selection.ipynb
│   └── chapitre-2-diff-in-diff.ipynb
├── src/
│   ├── chapitre-1-generate-data.py       # génère data/chapitre-1/
│   ├── chapitre-1-analyse-biais-selection.py
│   ├── chapitre-2-generate-data.py       # génère data/chapitre-2/panel-*.csv
│   └── chapitre-2-diff-in-diff.py
├── data/
│   ├── chapitre-1/                       # généré, non versionné
│   └── chapitre-2/
│       ├── panel-{1,2,3}.csv             # généré, non versionné
│       ├── data_rio/                     # données réelles UPP de Rio, versionnées (voir plus bas)
│       └── processed/                    # snapshots intermédiaires, non versionné
├── figures/chapitre-2/                   # graphiques générés — non versionné
├── tables/
│   ├── chapitre-1/                       # résultats OLS, ATE/ATT, biais — non versionné
│   └── chapitre-2/                       # résultats TWFE/DiD/event-study — non versionné
├── docs/
│   ├── chapitre-1-enonce.pdf
│   └── chapitre-2-enonce.pdf
└── requirements.txt
```

## Chapitre 1 — Modèle de Rubin, biais de sélection et d'hétérogénéité

L'objectif est d'appliquer le **modèle de Rubin** (cadre contrefactuel / potential outcomes) pour identifier
statistiquement l'effet causal d'un traitement sur un résultat observé, et d'illustrer comment un
**biais de sélection** et un **biais d'hétérogénéité** peuvent faire diverger un estimateur naïf de
l'effet causal réel.

### Méthode

Le modèle de Rubin définit, pour chaque individu $i$, deux résultats potentiels : $Y_i(0)$ en
l'absence de traitement et $Y_i(1)$ sous traitement. Seul l'un des deux est observé :

$$Y_i = T_i \cdot Y_i(1) + (1 - T_i) \cdot Y_i(0)$$

Le notebook compare, sur trois jeux de données aux propriétés différentes, l'estimateur naïf
($\text{E}[Y \mid T=1] - \text{E}[Y \mid T=0]$) à l'**ATE** (*Average Treatment Effect*) et l'**ATT**
(*Average Treatment Effect on Treated*) calculés à partir des résultats potentiels, puis décompose
l'écart entre le naïf et l'ATE en un **biais de sélection** et un **biais d'hétérogénéité**.

### Données

Les données originales fournies avec l'exercice (≈370 Mo) sont trop volumineuses pour être versionnées
proprement dans un dépôt public. `src/chapitre-1-generate-data.py` reproduit un processus génératif (DGP)
simulant les trois mêmes scénarios pédagogiques :

| Jeu de données | Assignation du traitement | Effet du traitement | Biais |
|---|---|---|---|
| `data0` | indépendante de X | homogène | aucun |
| `data1` | corrélée à X | homogène | sélection |
| `data2` | corrélée à X | hétérogène selon X | sélection + hétérogénéité |

Les valeurs numériques exactes diffèrent donc de l'énoncé original, mais les propriétés statistiques
(signe et présence des biais) et les conclusions de l'analyse sont identiques.

### Résultats principaux

- **data0** : estimateur naïf ≈ ATE ≈ ATT → assignation aléatoire, pas de biais.
- **data1** : ATE ≈ ATT mais naïf ≠ ATE → biais de sélection pur (l'assignation dépend de X, corrélé
  au résultat potentiel de base).
- **data2** : ATE ≠ ATT et naïf ≠ ATE → biais de sélection *et* d'hétérogénéité (l'effet du traitement
  varie lui-même selon X).

Le notebook confirme ce diagnostic en comparant les moyennes de X entre groupes traités/non-traités et
en ré-estimant les modèles avec et sans contrôle par X sur un sous-échantillon (10 %).

### Reproduire l'analyse

```bash
pip install -r requirements.txt
python src/chapitre-1-generate-data.py          # génère data/chapitre-1/data{0,1,2}.csv
jupyter notebook notebooks/chapitre-1-analyse-biais-selection.ipynb
# ou, de façon équivalente :
python src/chapitre-1-analyse-biais-selection.py
```

## Chapitre 2 — Difference-in-Differences

Deux exercices : (1) estimation d'un effet de traitement sur données panel par les méthodes DiD
(effets fixes doubles et étude d'événement), (2) préparation d'un jeu de données réel de criminalité
par UPP (Unité de Police Pacificatrice) de Rio de Janeiro.

### Exercice 1 — DiD sur données panel

Trois jeux de données panel équilibrés (1000 individus × 120 mois, janvier 2007 à décembre 2016),
générés synthétiquement par `src/chapitre-2-generate-data.py` (mêmes principes que pour le Chapitre 1 :
DGP documenté, propriétés qualitatives préservées, valeurs numériques différentes de l'énoncé original).
Chaque individu appartient de façon fixe à un groupe traité ou contrôle ; le traitement produit un effet
uniquement après la période 71 (novembre 2012). Les trois jeux varient par l'amplitude de cet effet
(2, 4 et 6), pour illustrer son impact sur la significativité des résultats.

Pour chaque jeu :
1. **Test des tendances parallèles** : évolution comparée de la moyenne de `y` entre groupes, avant/après
   la rupture.
2. **Estimateur à effets fixes doubles (TWFE)** : `linearmodels.PanelOLS`, effets entité + temps, erreurs
   standard clusterisées par entité.
3. **DiD et étude d'événement** : `pyfixest`, régression sur l'interaction traitement × post-période, puis
   un coefficient par période relative à la rupture (fenêtre de ±24 périodes — l'ensemble des 120 périodes
   rendrait le graphique illisible).

Les trois estimateurs TWFE retrouvent l'effet simulé à moins de 0,5 % près (ex. 5,98 estimé pour un effet
simulé de 6,0), et l'étude d'événement ne montre aucune tendance pré-traitement significative — signe
d'un design DiD propre.

### Exercice 2 — Agrégation criminalité / UPP (Rio de Janeiro)

`data/chapitre-2/data_rio/{data_crime,data_upp}.xlsx` sont les données réelles fournies avec l'exercice
(criminalité mensuelle par UPP, dates de pacification) — **versionnées telles quelles** : elles sont
déjà publiques dans le projet [Interactive-Map-Rio-UPP](https://github.com/Marius-cld/Interactive-Map-Rio-UPP) et largement sous
la limite de taille de GitHub.

Le script agrège les infractions en catégories mortelles/non mortelles, les rapporte à la population,
détermine le statut de traitement (pré/post-pacification) et regroupe le résultat par UPP et trimestre
(`data/chapitre-2/processed/rio-crime-upp-aggregated.csv`).

### Reproduire l'analyse

```bash
pip install -r requirements.txt
python src/chapitre-2-generate-data.py          # génère data/chapitre-2/panel-{1,2,3}.csv
jupyter notebook notebooks/chapitre-2-diff-in-diff.ipynb
# ou, de façon équivalente :
python src/chapitre-2-diff-in-diff.py
```

---

*Les chemins de données/sorties sont résolus depuis l'emplacement des scripts/notebooks, quel que soit
le répertoire de travail au lancement.*
