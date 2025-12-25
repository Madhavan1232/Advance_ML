import pandas as pd
import os , sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

df = pd.read_csv(os.path.join(sys.path[0] , input()))

x = df["HealthText"]
y = df["Outcome"]

vector = CountVectorizer()
vector.fit(x)
x = vector.transform(x)

xtrain , xtest , ytrain , ytest = train_test_split(x , y , test_size = 0.2 , random_state = 42)
model = MultinomialNB()
model.fit(xtrain , ytrain)

test_input = "Age group: Senior | BMI status: Overweight | Glucose category: Very High Glucose Level"
res = vector.transform([test_input])
print(f"Prediction: {model.predict(res)[0]}")