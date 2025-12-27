import pandas as pd
import os , sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score


df = pd.read_csv(os.path.join(sys.path[0] , input()))

print("First 5 rows of the dataset:")
print(df.head())

print("\nNumber of samples in the data:")
print(df.shape[0])

print("\nData types of each column:")
print(df.dtypes)

x = df[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years']]
print("\nFeature columns:")
print("['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years']")

print("\nStatistical summary of numeric columns:")
print(df.describe())

y = df['left']
scalar = StandardScaler()
df_scalared = scalar.fit_transform(x)

xtrain , xtest , ytrain ,ytest = train_test_split(x ,y , test_size = 0.2 , random_state = 42 )

model = DecisionTreeClassifier()
model.fit(xtrain , ytrain);

y_pred = model.predict(xtest)

print(f"\nModel Accuracy: {accuracy_score(ytest , y_pred)}")

print(f"\nClassification Report:")
print(classification_report(ytest , y_pred))


