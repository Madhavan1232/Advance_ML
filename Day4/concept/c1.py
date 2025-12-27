import pandas as pd
import os ,sys
from sklearn.model_selection import KFold , LeaveOneOut , StratifiedKFold  ,cross_val_score , train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv(os.path.join(sys.path[0] , input()))

print(df.head())

features = ['Fasting blood', 'bmi', 'age', 'FamilyHistory', 'HbA1c']

x = df[features].values
y = df['target'].values

df_tree = DecisionTreeClassifier(max_depth = 4 , random_state = 42)

kf = KFold(n_splits = 5 , shuffle = True , random_state = 42)
kf_score = cross_val_score(df_tree , x ,y , cv = kf)

print(f"\nK-Fold Accuracy Scores: {kf_score}")
print(f"Mean CV Accuracy: {kf_score.mean():.3f}")

xtrain , xtest , ytrain , ytest = train_test_split(x ,y , test_size = 0.3 , random_state = 42)
df_tree.fit(xtrain , ytrain)
pred = df_tree.predict(xtest)
hold_out = accuracy_score(ytest , pred);

print(f"Hold-Out Method Accuracy: {hold_out:.3f}")

lo = LeaveOneOut()
loo_score = cross_val_score(df_tree , x , y , cv = lo)
print(f"LOOCV Accuracy: {loo_score.mean():.3f}")

skf = StratifiedKFold(n_splits = 5 , shuffle = True , random_state = 42)
skf_score = cross_val_score(df_tree , x , y , cv = skf )
print(f"Accuracy: {skf_score.mean():.3f}")


