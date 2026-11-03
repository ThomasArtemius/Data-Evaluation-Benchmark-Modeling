"""The deliberately small candidate registry; edit here to add algorithms."""
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import Schema


def make_candidates(task: str, schema: Schema, seed: int, trees: int = 150) -> dict:
    if trees < 1:
        raise ValueError("The number of trees must be positive.")
    if task == "classification":
        models = {
            "dummy_prior": DummyClassifier(strategy="prior"),
            "logistic_regression": LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
            "random_forest": RandomForestClassifier(n_estimators=trees, min_samples_leaf=2,
                                                    random_state=seed, n_jobs=1),
        }
    else:
        models = {
            "dummy_mean": DummyRegressor(strategy="mean"),
            "ridge": Ridge(alpha=10.0),
            "random_forest": RandomForestRegressor(n_estimators=trees, min_samples_leaf=2,
                                                   random_state=seed, n_jobs=1),
        }
    pipelines = {}
    for name, model in models.items():
        # Every candidate gets its own preprocessing objects. All are fitted by
        # Pipeline.fit exclusively on training rows, including both baselines.
        transform = ColumnTransformer([
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]), schema.numeric),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), schema.categorical),
        ], remainder="drop")
        pipelines[name] = Pipeline([("preprocess", transform), ("model", model)])
    return pipelines
