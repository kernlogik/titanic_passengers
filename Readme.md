# Titanic Survival Explorer & Decision Tree Simulation

An interactive data analysis dashboard and survival simulator based on the historical Titanic passenger manifest. 

This project was developed during the introductory seminar for the **Data Science & Artificial Intelligence** program at the **University of Applied Sciences and Arts Northwestern Switzerland (FHNW)**, consolidating the exploratory and predictive modeling work of a four-person team.

---

## Features

* **Exploratory Data Analysis:** Fast data aggregation and filtering powered natively by Polars.
* **Demographic Visualizations:** Breakdown of passenger demographics, class divisions, and survival rates via Seaborn heatmaps and grouped bar charts.
* **Machine Learning Pipeline:** A Scikit-Learn `DecisionTreeClassifier` trained on passenger class, sex, and age.
* **Interactive What-If Simulation:** Reactive Marimo UI sliders and controls to navigate decision paths and predict individual survival probabilities in real time.
* **Client-Side WASM Ready:** Optimized for zero-backend browser execution via Pyodide and GitHub Pages.

---

## Tech Stack

* **Runtime & Reactive UI:** [Marimo](https://marimo.io/)
* **Data Processing:** [Polars](https://pola.rs/)
* **Visualization:** [Seaborn](https://seaborn.pydata.org/) & [Matplotlib](https://matplotlib.org/)
* **Machine Learning:** [scikit-learn](https://scikit-learn.org/)

---

## Getting Started

### Prerequisites

Ensure you have Python 3.10+ installed. Using [`uv`](https://github.com/astral-sh/uv) or `pip`:

```bash
# Clone the repository
git clone [https://github.com/](https://github.com/)<your-username>/<your-repo>.git
cd <your-repo>

# Install dependencies
pip install marimo polars scikit-learn seaborn matplotlib

```

*(Or using `uv`: `uv pip install marimo polars scikit-learn seaborn matplotlib`)*

---

### Running the Notebook

You can run the notebook in either application mode or interactive edit mode:

**1. Run as an interactive web app:**

```bash
marimo run synopsis.py

```

**2. Open in the Marimo notebook editor:**

```bash
marimo edit synopsis.py

```

Marimo will launch a local server and open the notebook automatically in your default browser (typically at `http://localhost:2718`).


### WASM Export (GitHub Pages)

To compile the notebook into a standalone, client-side HTML bundle for deployment:

```bash
marimo export html-wasm synopsis.py --output dist/index.html --mode run

```


## Team

Developed collaboratively by a 4-person team during the FHNW BSc Data Science & AI kick-off event.

<a href="https://github.com/kernlogik/titanic_passengers/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=kernlogik/titanic_passengers>" alt="Contributors" />
</a>

