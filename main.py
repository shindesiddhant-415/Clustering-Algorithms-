"""
Practical: Ensemble Learning and Performance Evaluation
Dataset: Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn

Run:
    pip install -r requirements.txt
    python main.py

The script trains five ensemble classifiers, evaluates them, and saves:
    outputs/model_comparison.csv
    outputs/classification_report.txt
    outputs/confusion_matrices.png
    outputs/accuracy_comparison.png
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
TEST_SIZE = 0.20
OUTPUT_DIR = Path("outputs")


def load_and_split_data():
    """Load the built-in dataset and create stratified train/test sets."""
    dataset = load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")
    class_names = list(dataset.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, class_names, X.shape


def build_models():
    """Create the ensemble models used in this practical."""
    random_forest = RandomForestClassifier(
        n_estimators=150,
        random_state=RANDOM_STATE,
    )

    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        n_estimators=100,
        random_state=RANDOM_STATE,
    )

    adaboost = AdaBoostClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
    )

    gradient_boosting = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
    )

    voting = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
            ("ada", AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE)),
            ("gb", GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ],
        voting="soft",
    )

    return {
        "Random Forest": random_forest,
        "Bagging": bagging,
        "AdaBoost": adaboost,
        "Gradient Boosting": gradient_boosting,
        "Voting Classifier": voting,
    }


def evaluate_models(models, X_train, X_test, y_train, y_test, class_names):
    """Train each model and calculate classification performance metrics."""
    results = []
    reports = []
    fitted_models = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_train, y_train)
        fitted_models[model_name] = model

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1-Score": f1_score(y_test, predictions, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, probabilities),
        }
        results.append(metrics)

        reports.append(
            f"\n{'=' * 65}\n{model_name}\n{'=' * 65}\n"
            + classification_report(
                y_test,
                predictions,
                target_names=class_names,
                zero_division=0,
            )
        )

        print(
            f"Accuracy: {metrics['Accuracy']:.4f} | "
            f"Precision: {metrics['Precision']:.4f} | "
            f"Recall: {metrics['Recall']:.4f} | "
            f"F1: {metrics['F1-Score']:.4f} | "
            f"ROC-AUC: {metrics['ROC-AUC']:.4f}"
        )

    return pd.DataFrame(results), reports, fitted_models


def save_outputs(results, reports, fitted_models, X_test, y_test, class_names):
    """Save the comparison table, classification reports and charts."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    results = results.sort_values("Accuracy", ascending=False)
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    with open(OUTPUT_DIR / "classification_report.txt", "w", encoding="utf-8") as file:
        file.write("ENSEMBLE LEARNING - CLASSIFICATION REPORTS\n")
        file.write("Dataset: Breast Cancer Wisconsin (Diagnostic)\n")
        file.write("Train/test split: 80% / 20%; random_state=42\n")
        file.write("\n".join(reports))

    # Accuracy comparison chart
    plt.figure(figsize=(10, 5))
    plt.bar(results["Model"], results["Accuracy"])
    plt.title("Accuracy Comparison of Ensemble Learning Algorithms")
    plt.xlabel("Ensemble Algorithm")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_comparison.png", dpi=160)
    plt.close()

    # Confusion matrices
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for ax, (model_name, model) in zip(axes, fitted_models.items()):
        predictions = model.predict(X_test)
        display = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(y_test, predictions),
            display_labels=class_names,
        )
        display.plot(ax=ax, colorbar=False, values_format="d")
        ax.set_title(model_name)
        ax.tick_params(axis="x", rotation=20)

    for ax in axes[len(fitted_models):]:
        ax.axis("off")

    fig.suptitle("Confusion Matrices for Ensemble Classifiers", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=160)
    plt.close(fig)

    return results


def main():
    print("=" * 65)
    print("PRACTICAL: ENSEMBLE LEARNING AND PERFORMANCE EVALUATION")
    print("=" * 65)

    X_train, X_test, y_train, y_test, class_names, data_shape = load_and_split_data()
    print(f"\nDataset loaded successfully: {data_shape[0]} samples, {data_shape[1]} features")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Classes: {class_names}")

    models = build_models()
    results, reports, fitted_models = evaluate_models(
        models, X_train, X_test, y_train, y_test, class_names
    )
    results = save_outputs(
        results, reports, fitted_models, X_test, y_test, class_names
    )

    print("\n" + "=" * 65)
    print("FINAL MODEL COMPARISON")
    print("=" * 65)
    print(results.to_string(index=False, formatters={
        column: (lambda value: f"{value:.4f}")
        for column in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    }))

    best_model = results.iloc[0]
    print(f"\nBest model by accuracy: {best_model['Model']}")
    print(f"Best accuracy: {best_model['Accuracy']:.4f}")
    print(f"\nAll output files have been saved in: {OUTPUT_DIR.resolve()}")
    print("Program executed successfully!")


if __name__ == "__main__":
    main()
