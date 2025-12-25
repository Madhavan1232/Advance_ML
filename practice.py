from sklearn.feature_extraction.text import CountVectorizer


documents = [
    "I love python",
    "python is easy",
    "I love machine learning"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
print(vectorizer.get_feature_names_out())
print(X.toarray())