
import optuna
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from evaluate_models import cross_validate_model

# Load training data only
df = pd.read_csv("train_candidates_final.csv")

def objective(trial):

    hidden_layer_1 = trial.suggest_categorical(
        "hidden_layer_1", [32, 64, 128]
    )

    hidden_layer_2 = trial.suggest_categorical(
        "hidden_layer_2", [16, 32, 64]
    )

    alpha = trial.suggest_float(
        "alpha", 0.00001, 0.01, log=True
    )

    learning_rate_init = trial.suggest_float(
        "learning_rate_init", 0.0001, 0.01, log=True
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(hidden_layer_1, hidden_layer_2),
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=100,
            batch_size=1024,
            early_stopping=False,
            random_state=42
        ))
    ])

    results = cross_validate_model("MLP_Optimized", model, df)

    mean_f1 = results["macro_f1"].mean()

    print("Trial macro F1:", mean_f1)

    return mean_f1


# Bayesian optimisation
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42)
)

study.optimize(objective, n_trials=5)

print("\nBEST PARAMETERS:")
print(study.best_params)

print("\nBEST MACRO F1:")
print(study.best_value)

study.trials_dataframe().to_csv(
    "evaluation/results/MLP_optimization_trials.csv",
    index=False
)

print("\nOptimization results saved!")
