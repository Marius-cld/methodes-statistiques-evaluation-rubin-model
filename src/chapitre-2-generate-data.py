"""Génère trois jeux de données panel synthétiques pour l'exercice de
Difference-in-Differences du Chapitre 2 (`data/chapitre-2/panel-{1,2,3}.csv`).

Panel équilibré de 1000 individus x 120 mois (janvier 2007 - décembre 2016),
avec une rupture de traitement à la période 71 (novembre 2012 — date déjà
utilisée dans le notebook comme repère "Début traitement"). Chaque individu
appartient de façon fixe au groupe traité ou non-traité ; l'effet du
traitement n'apparaît qu'après la période 71 pour le groupe traité.

DGP : effet fixe individuel + tendance temporelle commune aux deux groupes
(garantit des tendances parallèles avant traitement) + effet de traitement
post-période pour le groupe traité + bruit idiosyncratique. Les trois jeux
ne diffèrent que par l'amplitude de l'effet de traitement, pour illustrer
comment celle-ci se traduit dans le graphique des tendances parallèles et
dans la significativité des modèles DiD.

Ce générateur remplace les données de cours originales (mêmes propriétés
qualitatives — tendances parallèles avant rupture, effet net après — mais
valeurs numériques différentes).
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "chapitre-2"

N_IDS = 1_000
START_YEAR = 2007
N_MONTHS = 120  # 10 ans, janvier 2007 à décembre 2016
TREATMENT_START_PERIOD = 71  # novembre 2012

# Mois au format utilisé par le notebook (mélange EN/FR d'origine : "fev").
MONTHS = ["jan", "fev", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

P_TREATED = 0.4
ENTITY_EFFECT_SD = 3.0
TIME_TREND_PER_PERIOD = 0.05
NOISE_SD = 1.5

# Amplitude de l'effet de traitement (post-période 71), un scénario par jeu.
TREATMENT_EFFECTS = {"panel-1": 2.0, "panel-2": 4.0, "panel-3": 6.0}


def generer_panel(effet_traitement, rng):
    ids = np.arange(1, N_IDS + 1)
    periodes = np.arange(1, N_MONTHS + 1)

    treatment_group = rng.binomial(1, P_TREATED, size=N_IDS)
    entity_effect = rng.normal(0, ENTITY_EFFECT_SD, size=N_IDS)

    lignes = []
    for i, id_ in enumerate(ids):
        for periode in periodes:
            annee = START_YEAR + (periode - 1) // 12
            mois = MONTHS[(periode - 1) % 12]

            post = periode >= TREATMENT_START_PERIOD
            effet = effet_traitement if (treatment_group[i] == 1 and post) else 0.0

            y = (
                entity_effect[i]
                + TIME_TREND_PER_PERIOD * periode
                + effet
                + rng.normal(0, NOISE_SD)
            )

            lignes.append((id_, annee, mois, round(y, 4), treatment_group[i]))

    return pd.DataFrame(lignes, columns=["id", "year", "month", "y", "treatment"])


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    for nom, effet in TREATMENT_EFFECTS.items():
        df = generer_panel(effet, rng)
        df.to_csv(DATA_DIR / f"{nom}.csv", sep=";", index=False)
        print(f"{nom}.csv généré (effet de traitement = {effet})")

    print(f"Jeux de données générés dans {DATA_DIR}")


if __name__ == "__main__":
    main()
