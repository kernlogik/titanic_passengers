import polars as pl
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# Read from S3 storage and sanitize data
url = "https://s3.kroesch.net/example_datasets/titanic.parquet"
df = pl.read_parquet(url).select(pl.all().name.to_lowercase())

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

# 3. Dummies für kategorische Features erzeugen
df_encoded = df_clean.to_dummies(
    columns=["sex", "pclass", "age_group"],
    drop_first=True
)

# Feature-Matrix und Target trennen
target_col = "survived"
feature_cols = [c for c in df_encoded.columns if c not in (target_col, "age")]

X = df_encoded.select(feature_cols).to_numpy()
y = df_encoded[target_col].to_numpy()

# Stratifizierter Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Entscheidungsbaum trainieren
clf = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    min_samples_leaf=15,
    random_state=42
)
clf.fit(X_train, y_train)


"""
Klassifikationsbericht
"""


# 5-Fold Stratified Cross-Validation (Robustheit gegen Varianz)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
cv_auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")

print(f"CV Accuracy: {cv_acc.mean():.3f} (±{cv_acc.std():.3f})")
print(f"CV ROC-AUC:  {cv_auc.mean():.3f} (±{cv_auc.std():.3f})\n")

# Evaluation auf Testdaten
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("Klassifikationsbericht:")
print(classification_report(y_test, y_pred, target_names=["Verstorben", "Überlebt"]))

print("Konfusionsmatrix:")
print(confusion_matrix(y_test, y_pred))

print(f"\nTest ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

"""
Plot
"""

# Regeln textuell ausgeben
print(export_text(clf, feature_names=feature_cols))

# Baum grafisch darstellen
plt.figure(figsize=(14, 7), dpi=150)
plot_tree(
    clf,
    feature_names=feature_cols,
    class_names=["Verstorben", "Überlebt"],
    filled=True,
    proportion=True,  # zeigt relative Wahrscheinlichkeiten p0, p1 im Blatt
    rounded=True,
    fontsize=9
)
plt.tight_layout()
plt.show()
plt.savefig("tree.png")

