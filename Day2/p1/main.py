import pandas as pd
import os ,sys
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(os.path.join(sys.path[0] , input()))
print("=== First 5 Rows of Data ===")
print(df.head())

print(f"\nThe number of samples in data is {df.shape[0]}.\n")

print("=== Data Types ===")
print(df.dtypes)

print("\n=== Statistical Summary (Describe) ===")
print(df.describe())

print("\n=== Missing Values Per Column ===")
print(df.isnull().sum())

print("\n=== Salary Encoding Classes ===")
salary_uni = sorted(df['salary'].unique().tolist())
print(salary_uni)


le = LabelEncoder()
le.fit(salary_uni)
df['salary.enc'] = le.transform(df['salary'])


print("\n=== Department Encoding Classes ===")
depart_uni = sorted(df['Department'].unique().tolist())
print(depart_uni)

le_dep = LabelEncoder()
le_dep.fit(depart_uni)
df['Department.enc'] = le_dep.transform(df['Department'])

df = df.drop(columns = ["salary" , "Department"])
print("\n=== Dropping 'Department' and 'salary' columns ===")

print("\n=== Updated DataFrame Info ===")
df.info()