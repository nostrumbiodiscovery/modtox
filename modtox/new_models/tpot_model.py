import argparse
from tpot import TPOTClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from utils.utils import get_data


def run(csv, generations, population):
    X, y = get_data(csv)
    model = fit(X, y, generations, population)
    return model


def fit(X, y, generations, population):

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Fit TPOT
    model = TPOTClassifier(
        n_jobs=-1,
        generations=generations,
        population_size=population,
        verbosity=2,
        random_state=42,
        periodic_checkpoint_folder="tpot_checkpoints",
        log_file=None,
    )

    model.fit(X_train, y_train)

    # Export pipeline
    print(f"Final score {model.score(X_test, y_test)}.")
    print("Exporting the pipeline...")
    model.export("tpot_exported_pipeline.py")

    y_predicted = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_predicted).ravel()
    print("Confusion matrix: tp", tp, "fp", fp, "fn", fn, "tn", tn)

    return model


def parse_args(parser):
    parser.add_argument("--csv", help="CSV with the final model input.")
    parser.add_argument(
        "--generations", help="Number of TPOT generations.", default=20
    )
    parser.add_argument("--population", help="TPOT population size.", default=20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supply model_input.csv.")
    parse_args(parser)
    args = parser.parse_args()
    run(args.csv, int(args.generations), int(args.population))
