import sys
import os
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    recall_score,
    f1_score,
    precision_score
)

warnings.simplefilter(action='ignore')

def data_scale(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def evaluate_classifier(y_test, y_pred):
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("===================")

    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))
    print("===================")

    print(f"accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"recall: {recall_score(y_test, y_pred, pos_label=1):.3f}")
    print(f"f1-score: {f1_score(y_test, y_pred, pos_label=1):.3f}")
    print(f"precision: {precision_score(y_test, y_pred, pos_label=1):.3f}")

def main():
    try:
        file_name = input()
        file_path = os.path.join(sys.path[0], file_name)

        if not os.path.exists(file_path):
            print(f"Error: File '{file_name}' not found.")
            return

        df = pd.read_csv(file_path)

        le = LabelEncoder()
        df['salary.enc'] = le.fit_transform(df['salary'])
        df['Department.enc'] = le.fit_transform(df['Department'])

        df.drop(['salary', 'Department'], axis=1, inplace=True)

        X = df.drop('left', axis=1)
        y = df['left']

        X_scaled = data_scale(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.20, random_state=42
        )

        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'kernel': ['rbf']
        }

        svm = SVC()
        grid = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)

        y_pred = grid.best_estimator_.predict(X_test)

        evaluate_classifier(y_test, y_pred)

    except FileNotFoundError:
        print("Error: File not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
