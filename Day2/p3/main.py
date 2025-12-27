import pandas as pd
import os
import sys
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
pd.set_option('display.show_dimensions', False)

def check_correlation(input_df):
    corr_matrix = input_df.corr()
    bool_corr_matrix = corr_matrix.abs() >= 0.75
    return bool_corr_matrix

def data_scale(input_df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(input_df)
    scaled_df = pd.DataFrame(scaled_data, columns=input_df.columns)
    return scaled_df

def main():
    try:
        file_name = input()
        file_path = os.path.join(sys.path[0], file_name)

        if not os.path.exists(file_path):
            print(f"Error: File '{file_name}' not found.")
            return

        df = pd.read_csv(file_path)

        required_columns = [
            'satisfaction_level', 'last_evaluation', 'number_project',
            'average_montly_hours', 'time_spend_company', 'Work_accident',
            'left', 'promotion_last_5years', 'Department', 'salary'
        ]

        if not all(col in df.columns for col in required_columns):
            print(f"Error: CSV must contain exactly these columns: {required_columns}")
            return

        print("=== Label Encoding Categorical Columns ===")

        le_salary = LabelEncoder()
        df['salary.enc'] = le_salary.fit_transform(df['salary'])
        print(f"Encoded salary classes: {sorted(le_salary.classes_.tolist())}")

        le_dept = LabelEncoder()
        df['Department.enc'] = le_dept.fit_transform(df['Department'])
        print(f"Encoded Department classes: {sorted(le_dept.classes_.tolist())}")

        df_encoded = df.drop(['salary', 'Department'], axis=1)

        print("\n=== Separating Features and Label ===")
        y = df_encoded['left']
        X = df_encoded.drop('left', axis=1)

        print(f"Input Features Shape: {X.shape}")
        print(f"Label Shape: {y.shape}")

        print("\n=== Correlation Boolean Matrix (correlation >= 0.75) ===")
        corr_bool = check_correlation(X)
        print(corr_bool)
        print(f"\n[{corr_bool.shape[0]} rows x {corr_bool.shape[1]} columns]")

        print("\n=== Scaled Feature Sample (First 5 Rows) ===")
        X_scaled = data_scale(X)
        print(X_scaled.head())
        print(f"\n[5 rows x {X.shape[1]} columns]")

        print("\n=== Splitting Data into Train (80%) and Test (20%) ===")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.20, random_state=42
        )

        print(f"Training Features Shape: {X_train.shape}")
        print(f"Training Labels Shape: {y_train.shape}")
        print(f"Testing Features Shape: {X_test.shape}")
        print(f"Testing Labels Shape: {y_test.shape}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
