"""Chapitre 2 — Difference-in-Differences (panel) et agrégation de données
de criminalité (Rio de Janeiro / UPP).

Exercice 1 : sur trois jeux de données panel synthétiques (`panel-{1,2,3}`,
générés par `chapitre-2-generate-data.py`), teste l'hypothèse des tendances
parallèles, puis estime l'effet causal d'un traitement par deux approches
équivalentes : l'estimateur à effets fixes doubles (TWFE, `linearmodels`) et
une régression DiD avec étude d'événement (`pyfixest`).

Exercice 2 : agrège des données réelles de criminalité par UPP (Unité de
Police Pacificatrice) de Rio de Janeiro — déjà publiques dans le projet
`Interactive-Map-Rio-UPP` — en infractions mortelles/non mortelles
trimestrielles par UPP, avec indicateur de traitement (post-pacification).
"""

import io
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfixest as pf
from linearmodels.panel import PanelOLS

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "chapitre-2"
FIGURE_DIR = ROOT_DIR / "figures" / "chapitre-2"
TABLE_DIR = ROOT_DIR / "tables" / "chapitre-2"

PANELS = ("panel-1", "panel-2", "panel-3")
TRAITEMENT_PERIODE = 71  # novembre 2012, cf. chapitre-2-generate-data.py
MONTH_MAP = {
    "jan": 1, "fev": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def capturer_summary(fn):
    """pyfixest affiche son summary() sur stdout au lieu de le retourner :
    on capture cet affichage pour pouvoir le sauvegarder."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        fn()
    return buffer.getvalue()


### Exercice 1 : DiD sur données panel ###

def preparer_panel(df):
    df = df.copy()
    df["treatment"] = df["treatment"].astype("category")
    month_num = df["month"].map(MONTH_MAP)
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + month_num.astype(str), format="%Y-%m"
    ).dt.to_period("M")

    date_en_nombre = df["year"] * 12 + month_num
    df["periode"] = (date_en_nombre - date_en_nombre.min() + 1).astype(int)
    return df


def graphique_tendances_paralleles(df, nom):
    """Test visuel des tendances parallèles : évolution de y (moyenne par
    date) pour le groupe traité vs le groupe contrôle, avant/après rupture."""
    plot_df = df.copy()
    plot_df["date"] = plot_df["date"].dt.to_timestamp()
    agg = plot_df.groupby(["date", "treatment"], observed=True)["y"].mean().reset_index()
    controle = agg[agg["treatment"] == 0]
    traite = agg[agg["treatment"] == 1]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(controle["date"], controle["y"], color="steelblue", lw=2, label="Contrôle")
    ax.plot(traite["date"], traite["y"], color="tomato", lw=2, label="Traité")
    ax.axvline(
        pd.Timestamp("2012-11-01"), color="black", linestyle="--", linewidth=1.5,
        label="Début traitement",
    )
    ax.set_title(f"Évolution de y : test des tendances parallèles ({nom})", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("y moyen")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"tendances-paralleles-{nom}.png")
    plt.close(fig)


def modele_twfe(df, nom):
    """Estimateur à effets fixes doubles (entité + temps), erreurs standard
    clusterisées par entité."""
    df = df.copy()
    df["treatment"] = df["treatment"].astype(int)
    df["post"] = (df["periode"] >= TRAITEMENT_PERIODE).astype(int)
    df["did_interaction"] = df["treatment"] * df["post"]

    df_panel = df.set_index(["id", "periode"])
    mod = PanelOLS.from_formula(
        "y ~ did_interaction + EntityEffects + TimeEffects", data=df_panel, drop_absorbed=True
    )
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    with open(TABLE_DIR / f"twfe-{nom}.txt", "w") as f:
        f.write(res.summary.as_text())

    return df


EVENT_STUDY_WINDOW = 24  # périodes avant/après la rupture affichées sur le coefplot


def modele_did_event_study(df, nom):
    """DiD simple puis étude d'événement (effet par période relative à la
    rupture, référence = la période juste avant traitement).

    L'étude d'événement est estimée sur une fenêtre de ±EVENT_STUDY_WINDOW
    périodes autour de la rupture : sur les 120 périodes complètes, un
    coefficient par période relative (~119 coefficients) rendrait le
    coefplot illisible ; la fenêtre resserrée est la pratique standard pour
    une lecture graphique d'un event-study.
    """
    res = pf.feols("y ~ did_interaction | id + periode", data=df, vcov={"CRV1": "id"})
    summary_text = capturer_summary(res.summary)
    with open(TABLE_DIR / f"did-{nom}.txt", "w") as f:
        f.write(summary_text)

    df = df.copy()
    df["time_to_treat"] = df["periode"] - TRAITEMENT_PERIODE
    df_fenetre = df[df["time_to_treat"].abs() <= EVENT_STUDY_WINDOW]

    res_es = pf.feols(
        "y ~ i(time_to_treat, treatment, ref=-1) | id + periode",
        data=df_fenetre, vcov={"CRV1": "id"},
    )
    summary_es_text = capturer_summary(res_es.summary)
    with open(TABLE_DIR / f"event-study-{nom}.txt", "w") as f:
        f.write(summary_es_text)

    fig = res_es.coefplot()
    fig.set_size_inches(10, 6)
    fig.savefig(FIGURE_DIR / f"event-study-{nom}.png", bbox_inches="tight")
    plt.close(fig)


def exercice_1():
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    for nom in PANELS:
        df = pd.read_csv(DATA_DIR / f"{nom}.csv", sep=";")
        df = preparer_panel(df)

        traites = df[df["treatment"] == 1]["date"].min()
        non_traites = df[df["treatment"] == 0]["date"].min()
        print(f"[{nom}] traité depuis : {traites} | non-traité depuis : {non_traites}")

        graphique_tendances_paralleles(df, nom)
        df = modele_twfe(df, nom)
        modele_did_event_study(df, nom)

        df.to_excel(processed_dir / f"{nom}-enrichi.xlsx", index=False)


### Exercice 2 : agrégation criminalité / UPP (Rio) ###

def exercice_2():
    df_crime = pd.read_excel(DATA_DIR / "data_rio" / "data_crime.xlsx")
    df_upp = pd.read_excel(DATA_DIR / "data_rio" / "data_upp.xlsx")

    for df in (df_crime, df_upp):
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)

    ### Agrégation : "crimes non mortels" | "crimes mortels" ###
    death = ["homicideintentional", "bodyinjurydeathfollowed", "robberydeathfollowed"]
    no_death = ["attemptedmurder", "bodyinjuryintentional"]
    df_crime["nonfataloffences"] = df_crime[no_death].sum(axis=1)
    df_crime["fataloffences"] = df_crime[death].sum(axis=1)

    list_a = np.unique(df_crime["UPP"])
    list_b = np.unique(df_upp["UPP"])
    missing_in_b = np.setdiff1d(list_a, list_b)
    print(f"UPP dans crimeUPP : {len(list_a)} | UPP dans dataUPP : {len(list_b)} | absents de dataUPP : {list(missing_in_b)}")

    df = df_crime.merge(df_upp, how="outer")

    df["nonfataloffences-pop"] = 1 + (df["nonfataloffences"] / df["population"])
    df["fataloffences-pop"] = 1 + (df["fataloffences"] / df["population"])
    df["log-nonfataloffences-pop"] = np.log(df["nonfataloffences-pop"])
    df["log-fataloffences-pop"] = np.log(df["fataloffences-pop"])

    df["date_format"] = pd.to_datetime(
        "01/" + df["month"].astype(str) + "/" + df["year"].astype(str), format="%d/%m/%Y"
    )
    df["date_Bope"] = pd.to_datetime(df["date_Bope"], format="%d/%m/%y")
    df["date_upp"] = pd.to_datetime(df["date_upp"], format="%d/%m/%y")

    start = df["date_format"].min()
    df["date_upp_num"] = (
        (df["date_upp"].dt.year - start.year) * 12 + (df["date_upp"].dt.month - start.month) + 1
    )
    df["treatment"] = np.where(df["date_format"] >= df["date_upp"], 1, 0)

    df["quarter"] = (df["date"] - 1) // 3 + 1
    df_grouped = (
        df.groupby(["UPP", "quarter"])
        .agg(
            {
                "City": "first",
                "Complexo": "first",
                "cod_upp": "first",
                "Gang": "first",
                "date_Bope": "first",
                "date_upp": "first",
                "date_upp_num": "first",
                "treatment": "first",
                "log-nonfataloffences-pop": "sum",
                "log-fataloffences-pop": "sum",
            }
        )
        .reset_index()
    )

    df_grouped["upp_id"] = pd.factorize(df_grouped["UPP"])[0] + 1
    df_grouped["date_upp_num"] = df_grouped["date_upp_num"].fillna(0).astype(int)

    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    df_grouped.to_csv(processed_dir / "rio-crime-upp-aggregated.csv", index=False)

    return df_grouped


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    exercice_1()
    df_grouped = exercice_2()
    print(df_grouped.head())

    print("Terminé.")


if __name__ == "__main__":
    main()
