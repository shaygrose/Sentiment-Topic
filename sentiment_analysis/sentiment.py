import pandas as pd
import os
import glob
import re
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# from helper import clean_text


# uncomment to use the web scraped news
# df = pd.read_csv("../input/scraped_news.csv") 

# comment out if using the web scraped news
df = pd.read_csv("../input/bbc_articles.csv") 
tfidf_data = pd.read_csv("../input/tfidf_bbc_articles.csv")

# negative lexicon #
f = open("opinionLexicon/negative-words.txt", "r")
nLex = f.read().splitlines()
#print(nLex)
f.close()

# positive lexicon #
f = open("opinionLexicon/positive-words.txt", "r")
pLex = f.read().splitlines()
f.close()

cols = tfidf_data.columns
bt = tfidf_data.apply(lambda x: x > 0)
bt = bt.apply(lambda x: list(cols[x.values]), axis=1)

numPosWords_tfidf = [] 
numNegWords_tfidf = []
classification_tfidf = []
classificationTitle_tfidf = []
# only looking at words that are deemed important by tfidf
for row in bt:
    pos = 0
    neg = 0
    # for each word in the article
    for word in row:
        #checking if word is in positive corpus
        if word.lower() in pLex:
            pos += 1
        #checking if word is in negative corpus
        if word.lower() in nLex:
            neg += 1
    # appending the number of negative and positive words for each article
    numPosWords_tfidf.append(pos)
    numNegWords_tfidf.append(neg)
    # compare the values and give it a sentiment classification
    if pos < neg:
        classification_tfidf.append("Negative")
    elif pos > neg:
        classification_tfidf.append("Positive")
    else:
        classification_tfidf.append("Neutral")
    
    #do the same thing for the article title
    posTitle = 0
    negTitle = 0
   

df['Positive Words'] = numPosWords_tfidf
df['Negative Words'] = numNegWords_tfidf
df['Article Sentiment'] = classification_tfidf
# df['Title Sentiment'] = classificationTitle

# Uncomment if using the scraped news articles & comment out all graphing code below
# df.to_csv("output/scraped_sentiment_analysis.csv", index=False)

df.to_csv("../output/sentiment/bbc_sentiment_analysis_tfidf.csv", index=False)


# Create visualizations of the results #

# Graph the number of positive/negative articles in each category
labels = ['business', 'entertainment', 'politics', 'sports', 'tech']
groupedCategory = df.groupby(['category', 'Article Sentiment'])['title'].count().reset_index()

graph = sns.catplot(x = 'category',       # x variable name
            y = 'title',       # y variable name
            hue = "Article Sentiment",  # group variable name
            data = groupedCategory,     # dataframe to plot
            kind = "bar",
            palette=sns.color_palette(['red', 'grey', 'green']))

ax = graph.facet_axis(0,0)
for p in ax.patches:
    ax.text(p.get_x() + 0.015, 
            p.get_height() * 1.02, 
            '{0:.0f}'.format(p.get_height()), 
            color='black', rotation='horizontal', size='small')

graph.set(xlabel='Article Category', ylabel='Number of Articles')

graph.savefig("../output/sentiment/Sentiment-vs-Category-tfidf.png")




### Sentiment Analysis using the article text
numPosWords = [] 
numNegWords = []
classification = []
classificationTitle = []
articleNum = 0
for article in df['article']:
    pos = 0
    neg = 0
    # for each word in the article
    for word in str(article).split():
        #checking if word is in positive corpus
        if word.lower() in pLex:
            pos += 1
        #checking if word is in negative corpus
        if word.lower() in nLex:
            neg += 1
    # appending the number of negative and positive words for each article
    numPosWords.append(pos)
    numNegWords.append(neg)
    # compare the values and give it a sentiment classification
    if pos < neg:
        classification.append("Negative")
    elif pos > neg:
        classification.append("Positive")
    else:
        classification.append("Neutral")
    
    #do the same thing for the article title
    posTitle = 0
    negTitle = 0
    # grab the title of the corresponding article we are currently looking at
    for word in str(df['title'].iloc[articleNum]).split():
        #checking if word is in positive corpus
        if word.lower() in pLex:
            posTitle += 1
        #checking if word is in negative corpus
        if word.lower() in nLex:
            negTitle += 1
        # compare the values and give it a sentiment classification

    if posTitle < negTitle:
        classificationTitle.append("Negative")
    elif posTitle > negTitle:
        classificationTitle.append("Positive")
    else:
        classificationTitle.append("Neutral")
    articleNum += 1
  
df['Positive Words'] = numPosWords
df['Negative Words'] = numNegWords
df['Article Sentiment'] = classification
df['Title Sentiment'] = classificationTitle

# Uncomment if using the scraped news articles & comment out all graphing code below
# df.to_csv("../output/scraped_sentiment_analysis.csv", index=False)

# df.to_csv("../output/bbc_sentiment_analysis.csv", index=False)

# # Create visualizations of the results

# Graph the number of positive/negative articles in each category
labels = ['business', 'entertainment', 'politics', 'sports', 'tech']
groupedCategory = df.groupby(['category', 'Article Sentiment'])['title'].count().reset_index()

graph1 = sns.catplot(x = 'category',       # x variable name
            y = 'title',       # y variable name
            hue = "Article Sentiment",  # group variable name
            data = groupedCategory,     # dataframe to plot
            kind = "bar",
            palette=sns.color_palette(['red', 'grey', 'green']))

bx = graph1.facet_axis(0,0)
for p in bx.patches:
    bx.text(p.get_x() + 0.015, 
            p.get_height() * 1.02, 
            '{0:.0f}'.format(p.get_height()), 
            color='black', rotation='horizontal', size='small')

graph1.set(xlabel='Article Category', ylabel='Number of Articles')

graph1.savefig("../output/sentiment/Sentiment-vs-Category.png")

### COMPARING THE RAW COUNTS WITH THE TFIDF RESULTS ###
same = 0
different = 0
compared = []
for i in range(len(classification)):
    if (classification[i] == classification_tfidf[i]):
        same += 1
        compared.append("Same")
    else:
        different += 1
        compared.append("Different")
# print(same)
# print(different)
# print(len(compared))

df['Comparison'] = compared
df.to_csv("../output/sentiment/bbc_sentiment_comparison_data.csv", index=False)

### Graph the classification comparison between raw counts and tfidf according to categories ###
labels = ['business', 'entertainment', 'politics', 'sports', 'tech']
groupedCategory = df.groupby(['category', 'Comparison'])['title'].count().reset_index()

graph3 = sns.catplot(x = 'category',       # x variable name
            y = 'title',       # y variable name
            hue = "Comparison",  # group variable name
            data = groupedCategory,     # dataframe to plot
            kind = "bar",
            palette=sns.color_palette(['red', 'blue']))

cx = graph3.facet_axis(0,0)
for p in cx.patches:
    cx.text(p.get_x() + 0.015, 
            p.get_height() * 1.02, 
            '{0:.0f}'.format(p.get_height()), 
            color='black', rotation='horizontal', size='small')

graph3.set(xlabel='Article Category', ylabel='Number of Articles')

graph3.savefig("../output/sentiment/Sentiment-RawCount-vs-tfidf-categories.png")
