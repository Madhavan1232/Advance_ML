import pandas as pd
import sys , os
from sklearn.preprocessing import StandardScaler

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

    print(f"The number of samples in data is {len(df)}.")

    print("\nData Types:")
    print(df.dtypes)

    print("\nNumeric Summary:")
    summary = df.describe()
    print(summary)

    df_dropped = df.drop(columns=['Department', 'salary'])
    
    print("\nData After Dropping Irrelevant Columns:")
    df_dropped.info()
    print("None")

    input_df = df_dropped.drop(columns=['average_monthly_hours'])
    
    print("\nInput Features:")
    print(input_df.head(5))

    output_df = df['average_monthly_hours']
    
    print("\nTarget Variable:")
    print(output_df.head(5))

    scaled_features_df = data_scale(input_df)
    
    print("\nScaled Feature Data:")
    print(scaled_features_df.head(5))

if __name__ == "__main__":
    main()