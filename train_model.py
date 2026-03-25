import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def clean_text(text):
    """
    Removes decision leakage + basic text cleanup
    """
    if pd.isna(text):
        return ""

    text = text.upper()

    # Remove decision words (CRITICAL to prevent leakage)
    text = re.sub(r"\b(GRANTED|DENIED|REFUSED|REJECTED|APPEAL DENIED|APPEAL SUSTAINED)\b", "", text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main():

    print("Loading dataset...")

    df = pd.read_csv("zba_cases_dataset.csv")

    print("Initial rows:", len(df))

    # Drop missing decisions
    df = df[df["decision"].notnull()]

    print("After dropping missing decisions:", len(df))

    # Binary target
    df["approved"] = df["decision"].apply(lambda x: 1 if x == "GRANTED" else 0)

    # Clean text
    df["clean_text"] = df["raw_text"].apply(clean_text)

    # Remove empty rows
    df = df[df["clean_text"].str.len() > 50]

    print("After cleaning text:", len(df))

    # Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["clean_text"])

    y = df["approved"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")

    # Model with class balancing
    model = LogisticRegression(class_weight="balanced", max_iter=1000)

    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)

    print("\nMODEL PERFORMANCE:\n")
    print(classification_report(y_test, preds))


if __name__ == "__main__":
    main()