import pandas as pd
import numpy as np
import os , sys
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score, precision_score


def main():
    try:
        file_input = input().strip()
        df = pd.read_csv(os.path.join(sys.path[0] , file_input))

        le = LabelEncoder()
        df['salary.enc'] = le.fit_transform(df['salary'])
        df['Department.enc'] = le.fit_transform(df['Department'])
        
        X = df[['satisfaction_level', 'last_evaluation', 'number_project', 
                'average_montly_hours', 'time_spend_company', 'Work_accident', 
                'promotion_last_5years', 'salary.enc', 'Department.enc']]
        y = df['left']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        print("Confusion Matrix")
        print(confusion_matrix(y_test, y_pred))
        print("===================\n")
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))
        print("===================\n")
        
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        
        print(f"accuracy: {acc:.3f}")
        print(f"recall: {rec:.3f}")
        print(f"f1-score: {f1:.3f}")
        print(f"precision: {prec:.3f}")

    except:
        pass

if __name__ == "__main__":
    main()