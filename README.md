# Iris Flower Classification

## Overview
This project uses a Decision Tree classifier to predict Iris flower species using the built-in Iris dataset from scikit-learn. The project demonstrates Python development, Git version control, automated testing, and continuous integration using GitHub Actions.

## Technologies Used
- Python
- scikit-learn
- pytest
- uv
- Git and GitHub
- GitHub Actions

## Project Structure
```text
iris-friend-project/
├── src/
│   ├── __init__.py
│   └── model.py
├── tests/
│   └── test_model.py
├── docs/
├── .github/
│   └── workflows/
│       └── tests.yml
├── pyproject.toml
├── uv.lock
├── README.md
└── .gitignore
```

## Setup
Install uv if needed, then install the project dependencies:

```bash
uv sync
```

## Run the Model
```bash
uv run python src/model.py
```

The script trains the classifier, evaluates its accuracy, and prints sample predictions.

## Run Automated Tests
```bash
uv run python -m pytest -v
```

The test suite checks model training, dataset dimensions, prediction labels, and accuracy.

## Continuous Integration
GitHub Actions runs the automated tests when code is pushed to the main branch or when a pull request targets main. The workflow uses uv to install dependencies and execute the tests.

## Results
Record the accuracy produced by your actual model run here.

## Author
Jashan Shah (uselesshumanbeing)