import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import altair

    """ CSS Styling """
    mo.Html(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700&family=STIX+Two+Text:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

          /* Fließtext: STIX Two Text */
          body, .prose, .markdown, p, li {
              font-family: 'STIX Two Text', Georgia, serif;
              font-size: 18px;
              line-height: 1.65;
              letter-spacing: 0.01em;
          }

          /* Überschriften: Entweder STIX Two (fett) oder Inter */
          .prose h1, .prose h2, .prose h3, h1, h2, h3 {
              /* Variante A (Modern Science): Sans-Überschrift */
              font-family: 'Inter', system-ui, sans-serif;
              font-weight: 600;
              letter-spacing: -0.02em;

              /* Variante B (Voll-Serif): Wenn gewünscht, Zeile oben auskommentieren und hier aktivieren: */
              /* font-family: 'STIX Two Text', Georgia, serif; font-weight: 700; */
          }

          /* UI-Widgets & Steuerelemente */
          .marimo-ui, label, button, input, select {
              font-family: 'Inter', system-ui, sans-serif !important;
              font-size: 15px;
          }

          /* Codeblöcke & Zahlen */
          code, pre, .font-mono {
              font-family: 'JetBrains Mono', monospace !important;
              font-size: 0.9em;
          }
        </style>
        """
    )
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Der Untergang der Titanic

    **Abstract**

    Die Überlebenschancen beim Untergang der Titanic waren nicht zufällig verteilt. Diese Analyse modelliert die Passagierdaten anhand von Geschlecht, Reiseklasse und Altersgruppe, um mithilfe eines Entscheidungsbaums die entscheidenden Faktoren und Kohorten offenzulegen.

    ![](https://www.kroesch.ch/posts/der_untergang/Titanic_wreck_bow.jpg)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Überblick
    """)
    return


@app.cell
def _():
    """ Einlesen Dataframe """
    import polars as pl
    url = "https://kernlogik.github.io/titanic_passengers/titanic.parquet"
    # url = "titanic.parquet"
    df = pl.read_parquet(url).select(pl.all().name.to_lowercase())
    return df, pl


@app.cell(hide_code=True)
def _(df, mo):
    _df = mo.sql(
        f"""
        SELECT 
            CASE WHEN survived = 1 THEN 'Ja' ELSE 'Nein' END AS Überlebt,
            CASE WHEN sex = 'male' THEN 'Männlich' ELSE 'Weiblich' END AS Geschlecht,
            pclass AS Klasse,
            age AS Alter,
            name as Name
        FROM df
        """,
        output=False
    )
    return


@app.cell
def _(df, mo, pl):
    display_df = df.select([
        pl.when(pl.col("survived") == 1)
          .then(pl.lit("Ja"))
          .when(pl.col("survived") == 0)
          .then(pl.lit("Nein"))
          .otherwise(None)
          .alias("Überlebt"),

        pl.when(pl.col("sex").cast(pl.String).is_in(["male", "0"]))
          .then(pl.lit("Männlich"))
          .when(pl.col("sex").cast(pl.String).is_in(["female", "1"]))
          .then(pl.lit("Weiblich"))
          .otherwise(None)
          .alias("Geschlecht"),

        pl.col("pclass").alias("Klasse"),
        pl.col("age").alias("Alter"),
        pl.col("name").alias("Name"),
    ])

    mo.ui.table(display_df, page_size=10)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Überlebende nach Merkmalen

    Die Titanic hatte mehrere Reiseklassen, wobe sich die erste Klasse auf den oberen Decks befand. Die Passagiere wurden dort früher geweckt und hatten besseren Zugang zzu den Rettungsbooten. Die zweite und dritte Klasse befand sich auf den tieferen Decks im Schiffsrumpf und Fluchtwege waren ausserdem durch Absperrgitter versperrt. Ausserdem gilt in der Seefahrt "Frauen und Kinder zuerst"; dieser Grundsatz soll die Restlebenszeit maximieren.

    Wir vermuten also, dass generell mehr Frauen und Passagiere der ersten und zweiten Klasse das Unglück überlebt haben.
    """)
    return


@app.cell
def _(df, mo):
    _df = mo.sql(
        f"""
        SELECT 
             pclass AS Klasse,
             COUNT(*) AS Passagiere,
             SUM(survived) AS Überlebende,
             ROUND(AVG(survived) * 100, 1) AS "Quote ges.",
             ROUND(AVG(survived) FILTER (WHERE sex = 'female') * 100, 1) AS "Quote Frauen",
             ROUND(AVG(survived) FILTER (WHERE sex = 'male') * 100, 1) AS "Quote Männer"
         FROM df
         GROUP BY pclass
         ORDER BY pclass ASC;
        """,
        output=False
    )
    return


@app.cell
def _(df, mo, pl):
    summary_df = (
        df.group_by("pclass")
        .agg([
            pl.len().alias("Passagiere"),
            pl.col("survived").sum().alias("Überlebende"),
            (pl.col("survived").mean() * 100).round(1).alias("Quote ges."),
            (
                pl.col("survived")
                .filter(pl.col("sex") == "female")
                .mean() * 100
            ).round(1).alias("Quote Frauen"),
            (
                pl.col("survived")
                .filter(pl.col("sex") == "male")
                .mean() * 100
            ).round(1).alias("Quote Männer"),
        ])
        .rename({"pclass": "Klasse"})
        .sort("Klasse")
    )

    # Interaktives Marimo-Tabellen-Widget anzeigen:
    mo.ui.table(summary_df)
    return


@app.cell
def _(df, pl, plt):
    import seaborn as sns

    sns.set_theme()

    category_names = ["Kinder", "Jugendliche", "Erwachsene", "Senioren"]

    """ Null-Werte ausschliessen, Altersgruppen und deutsches Geschlecht anlegen """
    df_binned = (
        df
        .drop_nulls(subset=["age"])
        .with_columns([
            pl.col("age")
            .cut(breaks=[15, 25, 65], labels=category_names)
            .alias("age_group"),
            pl.col("sex")
            .cast(pl.String)
            .replace({
                "male": "Männlich", "0": "Männlich",
                "female": "Weiblich", "1": "Weiblich"
            })
            .alias("Geschlecht"),
            # survived in Float wandeln für saubere Mittelwertbildung
            pl.col("survived").cast(pl.Float64)
        ])
    )

    """ Pivotieren: mean * 100 liefert die Überlebensquote in Prozent """
    pivot_df = (
        df_binned
        .pivot(
            on="Geschlecht",
            index="age_group",
            values="survived",
            aggregate_function="mean",
        )
        .with_columns([
            (pl.col("Männlich") * 100).round(1),
            (pl.col("Weiblich") * 100).round(1)
        ])
        .sort("age_group")
    )

    """ Achsen und Matrix vorbereiten """
    x_labels = ["Männlich", "Weiblich"]
    y_labels = pivot_df["age_group"].to_list()
    heatmap_matrix = pivot_df.select(x_labels).to_numpy()

    """ Heatmap mit Prozentwerten und korrekter Farbskala ausgeben """
    h_fig, h_ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        heatmap_matrix,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",  # Rot = geringe Chance, Grün = hohe Chance
        vmin=0,
        vmax=100,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cbar_kws={"label": "Überlebensquote in %"},
        ax=h_ax,
    )
    h_ax.invert_yaxis()

    h_ax
    return (sns,)


@app.cell
def _(df, pl, plt, sns):

    sns.set_theme(style="whitegrid")

    """ Daten aggregieren und Überlebensrate in Prozent berechnen """
    bar_df = (
        df.with_columns([
            pl.col("pclass").cast(pl.String).replace({
                "1": "1. Klasse",
                "2": "2. Klasse",
                "3": "3. Klasse"
            }).alias("Klasse"),
            pl.col("sex").cast(pl.String).replace({
                "male": "Männer", "0": "Männer",
                "female": "Frauen", "1": "Frauen"
            }).alias("Geschlecht"),
            pl.col("survived").cast(pl.Float64)
        ])
        .group_by(["Klasse", "Geschlecht"])
        .agg((pl.col("survived").mean() * 100).round(1).alias("Überlebensrate"))
        .sort(["Klasse", "Geschlecht"])
    )

    """ Plot initialisieren und zeichnen (WASM-sicher ohne Arrow-Konvertierung) """
    b_fig, b_ax = plt.subplots(figsize=(7, 4.5))

    sns.barplot(
        data=bar_df.to_dict(as_series=False),
        x="Klasse",
        y="Überlebensrate",
        hue="Geschlecht",
        palette={"Frauen": "#2ca02c", "Männer": "#1f77b4"},
        ax=b_ax
    )

    """ Beschriftungen, Skala und Werte-Labels setzen """
    b_ax.set_title("Überlebensrate nach Klasse und Geschlecht", fontsize=13, weight="bold")
    b_ax.set_ylabel("Überlebensquote in %")
    b_ax.set_xlabel("Reiseklasse")
    b_ax.set_ylim(0, 105)

    for container in b_ax.containers:
        b_ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9)

    b_ax.legend(title="Geschlecht", loc="upper right")
    b_fig.tight_layout()

    b_ax
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Regelbasierte Klassifikation der Überlebenswahrscheinlichkeit

    Ein Entscheidungsbaum funktioniert wie ein logisches Flussdiagramm: Er stellt nacheinander einfache Ja/Nein-Fragen (etwa „Ist die Person weiblich?“ oder „Reist sie in der 3. Klasse?“), um Passagiere Schritt für Schritt in immer eindeutigere Gruppen zu sortieren. Am Ende jedes Pfads steht eine feste Kohorte samt historischer Überlebenswahrscheinlichkeit.

    Einzelne Bäume neigen stark zum "Auswendiglernen" (Overfitting): Lässt man sie zu tief wachsen, lernen sie zufälliges Rauschen der Passagierliste statt robuster Muster. Sie sind auch nicht stabil: Wenige geänderte Datenpunkte können die gesamte Baumstruktur kippen. Geht es rein um maximale Vorhersagekraft, greift man heute zu Ensembles wie Random Forests oder Gradient Boosting.

    Warum machen wir das hier trotzdem? Bei wenigen Merkmalen und knapp 1000 Datensätzen holen komplexe Ensembles kaum mehr heraus, machen das Ergebnis aber unlesbar. Ein flacher Entscheidungsbaum bildet die hierarchische Rettungsmechanik („Frauen und Kinder zuerst“, physischer Zugang nach Deckklasse) ausreichend ab.
    """)
    return


@app.cell
def _(df, pl):
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
    import matplotlib.pyplot as plt

    df_clean = (
        df.select(["survived", "sex", "pclass", "age"])
        .drop_nulls(subset=["survived", "sex", "pclass"])
        .with_columns(
            # guess missing age with median
            pl.col("age").fill_null(pl.col("age").median()),
            pl.col("sex").replace({"female": 1, "male": 0}).cast(pl.Int8)
        )
        .with_columns(
            pl.when(pl.col("age") <= 14).then(pl.lit("Kind"))
            .when(pl.col("age") <= 30).then(pl.lit("Junger_Erwachsener"))
            .when(pl.col("age") <= 60).then(pl.lit("Erwachsener"))
            .otherwise(pl.lit("Senior"))
            .alias("age_group"),
        )
    )

    """ Dummies für kategorische Features erzeugen """
    df_encoded = df_clean.to_dummies(
        columns=["sex", "pclass", "age_group"],
        drop_first=True
    )

    """ Feature-Matrix und Target trennen"""
    target_col = "survived"
    feature_cols = ["pclass", "sex", "age"]

    X = df_clean.select(feature_cols).to_numpy()
    y = df_clean.select("survived").to_numpy().ravel()

    """ Stratifizierter Split """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    """ Entscheidungsbaum trainieren """
    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=3,
        min_samples_leaf=15,
        random_state=42
    )
    clf.fit(X_train, y_train)

    """ Grafisch plotten """
    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)

    plot_tree(
        clf,
        ax=ax,
        feature_names=feature_cols,
        class_names=["Verstorben", "Überlebt"],
        filled=True,
        proportion=True,
        rounded=True,
        fontsize=9
    )
    fig.tight_layout()

    # Letzter Ausdruck der Zelle:
    ax
    return (
        StratifiedKFold,
        X,
        X_test,
        classification_report,
        clf,
        cross_val_score,
        plt,
        y,
        y_test,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Simulation

    Anhand der gelernten Trennkriterien des Baums lassen sich individuelle Profile testen: Wähle Klasse, Geschlecht und Alter, um zu sehen, welchem Entscheidungspfad das Modell folgt und wie hoch die geschätzte Überlebenschance ausfällt.
    """)
    return


@app.cell
def _(mo):
    import numpy as np

    pclass_input = mo.ui.dropdown(options={"1. Klasse": 1, "2. Klasse": 2, "3. Klasse": 3}, value="3. Klasse", label="Reiseklasse")
    sex_input = mo.ui.radio(options={"Weiblich": 1, "Männlich": 0}, value="Männlich", label="Geschlecht")
    age_input = mo.ui.slider(start=1, stop=80, step=1, value=25, label="Alter")

    mo.hstack([pclass_input, sex_input, age_input])
    return age_input, np, pclass_input, sex_input


@app.cell
def _(age_input, clf, mo, np, pclass_input, sex_input):

    # Wichtig: Falls clf mit 6 Merkmalen trainiert wurde, 
    # müssen hier auch 6 Werte übergeben werden:
    sample = np.array([[
        pclass_input.value, 
        sex_input.value, 
        age_input.value
    ]])

    pred = clf.predict(sample)[0]
    prob = clf.predict_proba(sample)[0][1]

    status = "Überlebt" if pred == 1 else "Verstorben"
    mo.md(f"**Prognose:** {status} *(Überlebenswahrscheinlichkeit: {prob * 100:.1f}%)*")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bewertung

    Der Klassifikationsbericht schlüsselt die Leistung des Modells getrennt nach Klassen auf. Das verhindert, dass eine scheinbar gute Gesamttrefferquote überdeckt, dass eine Klasse systematisch falsch vorhergesagt wird.

    **Die Kennzahlen**

    - Precision (Treffsicherheit): Wie verlässlich ist die Vorhersage?
    - Recall (Trefferquote / Sensitivität): Wie vollständig wird eine Klasse erkannt? Ein hohe Recall bedeutet wenige _False Negatives_.
    - F1-Score: Das harmonische Mittel aus Precision und Recall.
    - Support: Die absolute Anzahl der tatsächlichen Fälle dieser Klasse im Testdatensatz.
    """)
    return


@app.cell
def _(
    StratifiedKFold,
    X,
    X_test,
    classification_report,
    clf,
    cross_val_score,
    mo,
    y,
    y_test,
):
    """ Klassifikationsbericht """

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    cv_auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")

    # Evaluation auf Testdaten
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    mo.md("**Klassifikationsbericht**")
    report = classification_report(y_test, y_pred, target_names=["Verstorben", "Überlebt"])
    mo.md(f"""
    ```
    {report}
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Quellen
    - [Projektseite](https://github.com/kernlogik/titanic_passengers)
    - Rohdaten: https://s3.kroesch.net/example_datasets/titanic.parquet
    - [Scikit-Learn: Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
    - [Der Untergang](https://www.kroesch.ch/posts/der_untergang/): Blogpost zu den Ursachen der Katastrophe und den nautischen Konsequenzen.
    """)
    return


if __name__ == "__main__":
    app.run()
