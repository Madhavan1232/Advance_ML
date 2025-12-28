import pandas as pd
import numpy as np
import os
import sys
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

warnings.filterwarnings("ignore")

def main():
    filename = input().strip()

    try:
        df = pd.read_csv(os.path.join(sys.path[0], filename))
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    X = df[['Fasting blood', 'bmi', 'age', 'FamilyHistory', 'HbA1c']].values
    y = df['target'].values.ravel()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_grid = {
        'max_depth': [2, 3, 4, 5, 6],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 3]
    }

    dt_classifier = DecisionTreeClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=dt_classifier,
        param_grid=param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1
    )

    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Stratified CV Accuracy: {best_score:.3f}")

if __name__ == "__main__":
    main()