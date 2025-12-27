import pandas as pd
import os ,sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, recall_score, f1_score, accuracy_score, precision_score

df = pd.read_csv(os.path.join(sys.path[0] , input()))

print(df.head())
print(f"\n{df.dtypes}")
print("\nModel trained.")

features = ['Glucose', 'BMI', 'Age', 'FamilyHistory', 'HbA1c']
x = df[features]
y = df["Outcome"]

xtrain , xtest , ytrain , ytest =  train_test_split(x ,y , test_size = 0.33 , random_state = 42)

classifier = GaussianNB()
classifier.fit(xtrain , ytrain)

predct = classifier.predict(xtest)
print(f"Predicted Values: {repr(predct)}")

print("Confusion Matrix")
print(confusion_matrix(ytest , predct))
print("===================")
print(f"accuracy: {accuracy_score(ytest , predct):.3f}")
print(f"recall: {recall_score(ytest , predct):.3f}")
print(f"f1-score: {f1_score(ytest , predct):.3f}")
print(f"precision: {precision_score(ytest , predct):.3f}")