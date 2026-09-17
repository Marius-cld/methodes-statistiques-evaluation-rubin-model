"""
Génère trois jeux de données synthétiques reproduisant les trois scénarios
étudiés dans le notebook (`notebooks/chapitre-1-analyse-biais-selection.ipynb`) :

- data0 : assignation du traitement aléatoire  -> aucun biais
- data1 : assignation du traitement corrélée à X -> biais de sélection
- data2 : assignation corrélée à X + effet du traitement hétérogène selon X
          -> biais de sélection ET biais d'hétérogénéité

Le processus génératif (DGP) est volontairement simple et documenté afin de
rester reproductible et lisible. Les valeurs numériques exactes diffèrent du
jeu de données original fourni en cours, mais les propriétés statistiques
(biais nul / sélection / hétérogénéité) sont préservées à l'identique, ce qui
ne change ni la méthode ni les conclusions du notebook.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "chapitre-1"

N = 50_000
SEED = 42

# Effet de X sur le résultat potentiel de base (Y0)
X_EFFECT = -4.2
Y0_INTERCEPT = 15.0
NOISE_Y0 = 2.9

# Effet homogène du traitement utilisé pour data0 et data1
HOMOGENEOUS_EFFECT = 5.3
NOISE_EXTRA = 1.0

# Effet hétérogène du traitement utilisé pour data2 (plus faible, dépend de X)
HETEROGENEOUS_EFFECT = {0: 1.6, 1: 0.8}

# Probabilité de traitement selon X (data1 et data2) : les individus X=0
# (potentiel Y0 plus élevé) sont davantage traités -> corrélation traitement/X
P_TREATED_BY_X = {0: 0.35, 1: 0.15}


def _base_outcomes(rng):
    X = rng.binomial(1, 0.5, size=N)
    Y0 = Y0_INTERCEPT + X_EFFECT * X + rng.normal(0, NOISE_Y0, size=N)
    return X, Y0


def make_data0(rng):
    """Aucun biais : traitement assigné aléatoirement, effet homogène."""
    X, Y0 = _base_outcomes(rng)
    Y1 = Y0 + HOMOGENEOUS_EFFECT + rng.normal(0, NOISE_EXTRA, size=N)
    Treated = rng.binomial(1, 0.25, size=N)
    return _assemble(X, Y0, Y1, Treated)


def make_data1(rng):
    """Biais de sélection : P(Treated) dépend de X, effet homogène."""
    X, Y0 = _base_outcomes(rng)
    Y1 = Y0 + HOMOGENEOUS_EFFECT + rng.normal(0, NOISE_EXTRA, size=N)
    p_treated = np.where(X == 0, P_TREATED_BY_X[0], P_TREATED_BY_X[1])
    Treated = rng.binomial(1, p_treated)
    return _assemble(X, Y0, Y1, Treated)


def make_data2(rng):
    """Biais de sélection + biais d'hétérogénéité : effet du traitement varie selon X."""
    X, Y0 = _base_outcomes(rng)
    effect = np.where(X == 0, HETEROGENEOUS_EFFECT[0], HETEROGENEOUS_EFFECT[1])
    Y1 = Y0 + effect + rng.normal(0, NOISE_EXTRA, size=N)
    p_treated = np.where(X == 0, P_TREATED_BY_X[0], P_TREATED_BY_X[1])
    Treated = rng.binomial(1, p_treated)
    return _assemble(X, Y0, Y1, Treated)


def _assemble(X, Y0, Y1, Treated):
    Y = np.where(Treated == 1, Y1, Y0)
    return pd.DataFrame(
        {"Y0": Y0.round(2), "Y1": Y1.round(2), "X": X, "Treated": Treated, "Y": Y.round(2)}
    )


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    make_data0(rng).to_csv(DATA_DIR / "data0.csv", sep=";", index=False)
    make_data1(rng).to_csv(DATA_DIR / "data1.csv", sep=";", index=False)
    make_data2(rng).to_csv(DATA_DIR / "data2.csv", sep=";", index=False)

    print(f"Jeux de données générés dans {DATA_DIR} : data0.csv, data1.csv, data2.csv")


if __name__ == "__main__":
    main()
