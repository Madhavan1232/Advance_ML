#p1
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import warnings, os, sys

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

filename = input().strip()

try:
    df = pd.read_csv(os.path.join(sys.path[0], filename))
except:
    print(f"Error: Unable to read file '{filename}'.")
    exit()

for col in df.columns:
    if df[col].dtype == object:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

print("--- Outlier Assessment ---")

numeric_cols = df.select_dtypes(include=[np.number]).columns
outlier_bounds = {}

for col in df.columns:
    if col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        count = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_bounds[col] = (lower, upper)
    else:
        count = 0
    print(f"{col}: {count} outliers")

for col in outlier_bounds:
    lower, upper = outlier_bounds[col]
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])

if "Purchase Likelihood" not in df.columns:
    print("Error: Target column 'Purchase Likelihood' not found in dataset.")
    exit()

X = df.drop("Purchase Likelihood", axis=1)
y = df["Purchase Likelihood"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

split_index = int(0.8 * len(X_scaled))

X_train = X_scaled.iloc[:split_index]
X_test = X_scaled.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100



print("==============================")
print(f"Model Accuracy: {round(accuracy, 2)} %")
print("==============================")