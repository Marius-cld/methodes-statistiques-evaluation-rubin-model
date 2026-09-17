"""Chapitre 1 — Modèle de Rubin, biais de sélection et d'hétérogénéité.

Applique le modèle de Rubin (cadre contrefactuel / potential outcomes) sur
trois jeux de données (`data/chapitre-1/data{0,1,2}.csv`, générés par
`chapitre-1-generate-data.py`) illustrant chacun un scénario différent :

- data0 : assignation du traitement indépendante de X -> aucun biais
- data1 : assignation corrélée à X -> biais de sélection
- data2 : assignation corrélée à X + effet hétérogène selon X -> biais de
          sélection ET d'hétérogénéité

Pour chaque jeu : estimation par OLS (Y ~ Treated), calcul de l'ATE/ATT,
décomposition du biais, test d'indépendance de l'assignation (équilibre de
X entre groupes), et ré-estimation sur un sous-échantillon (10 %) avec et
sans contrôle par X.
"""

from pathlib import Path

import pandas as pd
import statsmodels.api as sm

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "chapitre-1"
TABLE_DIR = ROOT_DIR / "tables" / "chapitre-1"

DATASETS = ("data0", "data1", "data2")


def charger_donnees():
    dfs = {}
    for nom in DATASETS:
        df = pd.read_csv(DATA_DIR / f"{nom}.csv", sep=";")
        df["Treated"] = df["Treated"].astype("category")
        dfs[nom] = df
    return dfs


def estimer_ols(dfs):
    """Régression Y ~ Treated par OLS pour chaque jeu de données.

    Y_i = alpha + Delta * T_i + omega_i, où Delta est aussi l'estimateur de
    l'ATT (effet moyen sur les traités) lorsque la régression est non biaisée.
    """
    modeles = {}
    for nom, df in dfs.items():
        model = sm.OLS(df["Y"], sm.add_constant(df["Treated"])).fit()
        with open(TABLE_DIR / f"model-{nom}.txt", "w") as f:
            f.write(model.summary().as_text())
        modeles[nom] = model
    return modeles


def analyser_biais(dfs, modeles):
    """ATE, ATT (calculés directement à partir des résultats potentiels
    Y0/Y1, normalement non observables mais connus ici car simulés),
    décomposition en biais de sélection + biais d'hétérogénéité, et
    vérification de l'équilibre de X entre groupes traité/non-traité.

    ATE = E[Y(1) - Y(0)]                    effet moyen sur la population
    ATT = E[Y(1) - Y(0) | T=1]              effet moyen sur les traités
    naif = E[Y | T=1] - E[Y | T=0]
    biais de sélection = E[Y(0) | T=1] - E[Y(0) | T=0]
    biais d'hétérogénéité = ATE - ATT
    """
    lignes = []
    for nom, df in dfs.items():
        traites = df["Treated"] == 1
        non_traites = df["Treated"] == 0

        ate = df["Y1"].mean() - df["Y0"].mean()
        att = df.loc[traites, "Y1"].mean() - df.loc[traites, "Y0"].mean()
        att_ols = modeles[nom].params["Treated"]

        biais_selection = df.loc[traites, "Y0"].mean() - df.loc[non_traites, "Y0"].mean()
        biais_heterogeneite = ate - att

        y1_traite = df.loc[traites, "Y1"].mean()
        y0_non_traite = df.loc[non_traites, "Y0"].mean()

        x_traite = df.loc[traites, "X"].mean()
        x_non_traite = df.loc[non_traites, "X"].mean()

        lignes.append(
            {
                "dataset": nom,
                "ATE": ate,
                "ATT": att,
                "ATT_OLS": att_ols,
                "biais_selection": biais_selection,
                "biais_heterogeneite": biais_heterogeneite,
                "score_moyen_traites": y1_traite,
                "score_moyen_non_traites": y0_non_traite,
                "diff_score_brute": y1_traite - y0_non_traite,
                "X_moyen_traites": x_traite,
                "X_moyen_non_traites": x_non_traite,
                "diff_X_traites_non_traites": x_traite - x_non_traite,
            }
        )

    resultats = pd.DataFrame(lignes).set_index("dataset")
    resultats.to_csv(TABLE_DIR / "biais-ate-att.csv")
    return resultats


def sous_echantillon(dfs):
    """Ré-estime Y ~ T et Y ~ T + X sur un sous-échantillon (10 %) de chaque
    jeu de données : si l'ajout de X modifie le coefficient de Treated,
    l'assignation au traitement dépend de X (non randomisée)."""
    for nom, df in dfs.items():
        df_sub = df.sample(frac=0.1, replace=False, random_state=42).reset_index(drop=True)

        model = sm.OLS(df_sub["Y"], sm.add_constant(df_sub["Treated"])).fit()
        with open(TABLE_DIR / f"model-{nom}-souséchantillon.txt", "w") as f:
            f.write(model.summary().as_text())

        model_bis = sm.OLS(df_sub["Y"], sm.add_constant(df_sub[["Treated", "X"]])).fit()
        with open(TABLE_DIR / f"model-{nom}-souséchantillon-avec-X.txt", "w") as f:
            f.write(model_bis.summary().as_text())


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    dfs = charger_donnees()
    modeles = estimer_ols(dfs)
    resultats = analyser_biais(dfs, modeles)
    print(resultats)

    sous_echantillon(dfs)
    print("Terminé.")


if __name__ == "__main__":
    main()
