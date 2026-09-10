import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Der Untergang der Titanic

    **Abstract**

    Die Überlebenschancen beim Untergang der Titanic waren keineswegs zufällig verteilt. Diese Analyse modelliert die Passagierdaten anhand von Geschlecht, Reiseklasse und Altersgruppe, um mithilfe eines Entscheidungsbaums die entscheidenden Faktoren und Kohorten offenzulegen.
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
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Modellierung als Entscheidungsbaum

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
            pl.col("age").fill_null(pl.col("age").median())
        )
        .with_columns(
            pl.when(pl.col("age") <= 14).then(pl.lit("Kind"))
            .when(pl.col("age") <= 30).then(pl.lit("Junger_Erwachsener"))
            .when(pl.col("age") <= 60).then(pl.lit("Erwachsener"))
            .otherwise(pl.lit("Senior"))
            .alias("age_group"),
            pl.col("pclass").cast(pl.String)
        )
    )

    """ Dummies für kategorische Features erzeugen """
    df_encoded = df_clean.to_dummies(
        columns=["sex", "pclass", "age_group"],
        drop_first=True
    )

    """ Feature-Matrix und Target trennen"""
    target_col = "survived"
    feature_cols = [c for c in df_encoded.columns if c not in (target_col, "age")]

    X = df_encoded.select(feature_cols).to_numpy()
    y = df_encoded[target_col].to_numpy()

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
        y,
        y_test,
    )


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
    - [Scikit-Lear: Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
    - [Der Untergang](https://www.kroesch.ch/posts/der_untergang/): Blogpost zu den Ursachen der Katastrophe und den nautischen Konsequenzen.
    """)
    return


if __name__ == "__main__":
    app.run()
