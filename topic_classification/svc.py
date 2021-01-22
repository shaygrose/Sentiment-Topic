import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from helper import process_text, clean_text
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn import svm
import pickle

df = pd.read_csv('../input/stemmed_bbc_articles.csv')
min_df = 10
max_df = .70
max_features = 1000

tfidf = TfidfVectorizer(
    ngram_range=(1, 2), # unigrams and bigrams
    stop_words=None,
    lowercase=False,
    max_df=max_df,
    min_df=min_df,
    max_features=max_features,
    sublinear_tf=True # replace tf with 1 + log(tf)
)

category_codes = {
    'business': 0,
    'entertainment': 1,
    'politics': 2,
    'sports': 3,
    'technology': 4,
}


features_train = tfidf.fit_transform(df['article']).toarray()
# features_train = features_train.toarray()

# features_train = df['article']
# labels_train = df['category']

df['category_codes'] = df['category'].map(category_codes)
labels_train = df['category_codes']
labels_train = labels_train.astype('int')

# model training
# create the parameter grid based on the results of random search
C = [.0001, .001, .01, .1]
degree = [3, 4, 5]
gamma = [1, 10, 100]
probability = [True]

# specifies that we are exploring 3 grids with the same parameters, but different kernels
param_grid = [
    {'C': C, 'kernel': ['linear'], 'probability':probability},
    {'C': C, 'kernel': ['poly'], 'degree':degree, 'probability':probability},
    {'C': C, 'kernel': ['rbf'], 'gamma':gamma, 'probability':probability}
]

# base model
svc = svm.SVC(random_state=8)

# Yields indices to split data into training and test sets
cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=8)

# instantiate the grid search model
# Exhaustive search over specified parameter values for an estimator
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
grid_search = GridSearchCV(
    estimator=svc,
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv_sets,
    verbose=1
)

# fit the grid search to the data
grid_search.fit(features_train, labels_train)

best_svc = grid_search.best_estimator_
best_svc.fit(features_train, labels_train)

with open('svc_model.pickle', 'wb') as output:
    pickle.dump(best_svc, output)

with open('svc_tfidf.pickle', 'wb') as output:
    pickle.dump(tfidf, output)
