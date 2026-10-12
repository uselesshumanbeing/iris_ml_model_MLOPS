
from src.model import train_model, predict_species


def test_model_trains_successfully():
    model, X_test, y_test, accuracy = train_model()

    assert model is not None
    assert accuracy >= 0.90


def test_test_dataset_dimensions():
    model, X_test, y_test, accuracy = train_model()

    assert X_test.shape == (30, 4)
    assert len(y_test) == 30


def test_predictions_have_valid_labels():
    model, X_test, y_test, accuracy = train_model()

    predictions = predict_species(model, X_test)

    assert len(predictions) == len(y_test)
    assert set(predictions).issubset({0, 1, 2})


def test_model_accuracy():
    model, X_test, y_test, accuracy = train_model()

    predictions = predict_species(model, X_test)
    correct_predictions = sum(predictions == y_test)

    assert accuracy >= 0.90
    assert correct_predictions >= 27
