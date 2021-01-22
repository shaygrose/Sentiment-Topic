import pandas as pd 
import numpy as np
import glob
import os
import csv
from helper import process_text, clean_text
from sklearn.feature_extraction.text import TfidfVectorizer



business_files = os.path.join('bbc-data/business', '*.txt')
business_filenames = glob.glob(business_files)

entertainment_files = os.path.join('bbc-data/entertainment', '*.txt')
entertainment_filenames = glob.glob(entertainment_files)

politics_files = os.path.join('bbc-data/politics', '*.txt')
politics_filenames = glob.glob(politics_files)

sports_files = os.path.join('bbc-data/sport', '*.txt')
sports_filenames = glob.glob(sports_files)

tech_files = os.path.join('bbc-data/tech', '*.txt')
tech_filenames = glob.glob(tech_files)

titles = pd.Series([], dtype=object)
articles = pd.Series([], dtype=object)
category = []
filename = []

final = pd.DataFrame(columns=['title', 'title-length', 'article', 'article-length', 'category', 'source'])
for f in business_filenames:
    df = pd.read_csv(f, sep="\n", header=None, names=["article"])
    #splitting out first line in txt file as article title
    titles = titles.append(df.loc[0:0, 'article'])
    # slicing the dataframe to not include title
    df = df.loc[1:, :]
    df = df.T
    articles = articles.append(pd.Series(df.values.tolist()).str.join(' '))
    category.append("business")
    filename.append(f)


for f in entertainment_filenames:
    df = pd.read_csv(f, sep="\n", header=None, names=["article"])
    #splitting out first line in txt file as article title
    titles = titles.append(df.loc[0:0, 'article'])
    # slicing the dataframe to not include title
    df = df.loc[1:, :]
    df = df.T
    articles = articles.append(pd.Series(df.values.tolist()).str.join(' '))
    category.append("entertainment")
    filename.append(f)

for f in politics_filenames:
    df = pd.read_csv(f, sep="\n", header=None, names=["article"])
    #splitting out first line in txt file as article title
    titles = titles.append(df.loc[0:0, 'article'])
    # slicing the dataframe to not include title
    df = df.loc[1:, :]
    df = df.T
    articles = articles.append(pd.Series(df.values.tolist()).str.join(' '))
    category.append("politics")
    filename.append(f)


for f in sports_filenames:
    df = pd.read_csv(f, sep="\n", header=None,names=["article"])
    #splitting out first line in txt file as article title
    titles = titles.append(df.loc[0:0, 'article'])
    # slicing the dataframe to not include title
    df = df.loc[1:, :]
    df = df.T
    articles = articles.append(pd.Series(df.values.tolist()).str.join(' '))
    category.append("sports")
    filename.append(f)

for f in tech_filenames:
    df = pd.read_csv(f, sep="\n", header=None, names=["article"])
    #splitting out first line in txt file as article title
    titles = titles.append(df.loc[0:0, 'article'])
    # slicing the dataframe to not include title
    df = df.loc[1:, :]
    df = df.T
    articles = articles.append(pd.Series(df.values.tolist()).str.join(' '))
    category.append("technology")
    filename.append(f)
    
    
final['title'] = titles.values
final['article'] = articles.values
final['title-length'] = final['title'].str.split(' ').str.len()
final['article-length'] = final['article'].str.split(' ').str.len()
final['category'] = category
final['source'] = filename
final = final.drop_duplicates(subset=['title'])


#outputting stemmed articles csv
stemmed = final.copy()
stemmed['article'] = stemmed['article'].apply(process_text)
stemmed.to_csv('../input/stemmed_bbc_articles.csv', index='False')



final.to_csv('../input/bbc_articles.csv', index='False')



# clean the text but don't do any word stemming
final['article'] = final['article'].apply(
        clean_text)
final.to_csv('../input/clean_bbc_articles.csv', index='False')

#TF-IDF 
no_below = 10
no_above = 0.60
keep_n = 5000

tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), # unigrams and bigrams
    max_df=no_above,
    min_df=no_below,
    max_features=keep_n,
    preprocessor=' '.join
)


tfidf_vectorizer.fit(final['article'])
features = tfidf_vectorizer.transform(final['article'])
tfidf_df =  pd.DataFrame(features.todense(), columns = tfidf_vectorizer.get_feature_names())

tfidf_df.to_csv('../input/tfidf_bbc_articles.csv', index='False')

