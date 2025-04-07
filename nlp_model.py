import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

df = pd.read_csv('dataset_training.tsv', sep='\t')
print(df.head())
print(len(df))
print(df.columns)
print(df['label'].unique())
#print(df.isnull().sum())
print(df['label'].value_counts())

X = df['text'] 
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Build a pipeline
text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),
])
# Feed the training data through the pipeline
text_clf.fit(X_train, y_train)  

# Tset model

predictions = text_clf.predict(X_test)
# Confusion matrix
from sklearn import metrics
print("confusion matrix")
print(metrics.confusion_matrix(y_test,predictions))
print(metrics.classification_report(y_test,predictions))
print(metrics.accuracy_score(y_test,predictions))

# Feature_names and there weights - Model
clf = text_clf.named_steps['clf']
feature_names = text_clf.named_steps['tfidf'].get_feature_names_out()
weights = clf.coef_
#print(feature_names)
#print(weights)

# Test
print(text_clf.predict(["Hey, just wanted to check in and see what's up with that thing I sent you. No rush, just curious. Talk soon!",
                  "Dear Sir or Madam, I am writing to inquire about the status of my recent application. I would greatly appreciate any updates you could provide at your earliest convenience. Thank you for your time and consideration."]))