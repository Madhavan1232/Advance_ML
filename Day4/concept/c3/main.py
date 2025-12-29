import pandas as pd
import os ,sys
from sklearn.model_selection import train_test_split , GridSearchCV ,cross_val_score
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.metrics import mean_squared_error

df = pd.read_csv(os.path.join(sys.path[0] , input()))

print(df.head())

x = df[['bmi', 'age', 'insulin', 'FamilyHistory', 'bp']]
y = df['Fasting blood']

xtrain , xtest , ytrain , ytest = train_test_split(x , y , test_size = 0.3 , random_state = 42)
model = DecisionTreeRegressor(random_state = 42)
params = {
        'max_depth': [3 , 5, 10 , None],
        'min_samples_split': [2 , 5, 10],
        'min_samples_leaf' :[1 , 2 , 4]
    }

grid_search = GridSearchCV(
    estimator = model,
    param_grid = params,
    cv = 5,
    scoring='neg_mean_squared_error',
    n_jobs = -1,
    refit = True
)

grid_search.fit(xtrain , ytrain)
best_model = grid_search.best_estimator_

best_param = dict(sorted(grid_search.best_params_.items()))

print(f"\nBest Hyperparameters: {best_param}")

cv_score = cross_val_score(best_model , x ,y ,cv = 5 , scoring='neg_mean_squared_error')
rmse = np.sqrt(-cv_score)

print(f"Cross-Validation RMSE Scores: {rmse}")
print(f"Mean RMSE: {np.mean(rmse)}")

y_pred = best_model.predict(xtest)
test_rmse = np.sqrt(mean_squared_error(ytest , y_pred))
print(f"RMSE: {test_rmse}")

y_std = y.std()
print(f"Standard Deviation of Label: {y.std()}")

if test_rmse <= y_std:
    print("The model's RMSE is within the standard deviation, indicating good performance.")
else:
    print("The model's RMSE exceeds the standard deviation, suggesting room for improvement.")

