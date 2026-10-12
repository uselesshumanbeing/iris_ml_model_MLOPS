
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def load_data():
    """Load the Iris flower dataset."""
    iris = load_iris()
    return iris.data, iris.target


def train_model():
    """Train and evaluate a Decision Tree classifier."""
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, X_test, y_test, accuracy


def predict_species(model, features):
    """Predict the Iris class from flower measurements."""
    return model.predict(features)


if __name__ == "__main__":
    model, X_test, y_test, accuracy = train_model()

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Test samples: {len(y_test)}")

    predictions = predict_species(model, X_test[:5])
    print(f"First five predictions: {predictions}")
