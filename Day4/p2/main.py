import pandas as pd
import numpy as np
import sys , os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score


def data_scale(X_DT):
    numeric_df = X_DT.select_dtypes(include=['number'])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    return pd.DataFrame(scaled_data, columns=numeric_df.columns)

def main():
    filename = input()

    try:
        df = pd.read_csv(os.path.join(sys.path[0] , filename))
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    df_prepared = df.drop(columns=['Department', 'salary'])

    X = df_prepared.drop(columns=['average_monthly_hours'])
    y = df_prepared['average_monthly_hours']

    X_scaled = data_scale(X)

    regressor = DecisionTreeRegressor(random_state=42)

    cv_scores = cross_val_score(regressor, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
    mse_scores = -cv_scores
    avg_mse = np.mean(mse_scores)

    print(f"Cross-validated MSE: {avg_mse}")

    regressor.fit(X_scaled, y)
    predictions = regressor.predict(X_scaled)

    print("Predictions: ", end="")
    print(predictions)

if __name__ == "__main__":
    main()