import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
DATA_FILE = "wine_quality.csv"
MODEL_FILE = "wine_model.pkl"

def main():
    df = pd.read_csv(DATA_URL, sep=";")
    df.to_csv(DATA_FILE, index=False)
    print("Dataset shape:", df.shape)
    print(df.head())
    print("Missing values:\n", df.isnull().sum())
    print("Duplicate rows:", df.duplicated().sum())
    print("Quality distribution:\n", df["quality"].value_counts().sort_index())

    df = df.drop_duplicates().copy()
    y = (df["quality"] >= 6).astype(int)
    X = df.drop(columns=["quality"])  # Prevent target leakage

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=150, random_state=42, class_weight="balanced"
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
    print("Classification report:\n", classification_report(
        y_test, predictions,
        target_names=["Needs improvement", "Good quality"],
        zero_division=0
    ))
    joblib.dump(model, MODEL_FILE)
    print(f"Saved {MODEL_FILE}")

if __name__ == "__main__":
    main()
