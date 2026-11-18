from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

def get_models():
    return {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=15, weights="distance")),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=SEED),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                          max_iter=300, random_state=SEED)),
    }
