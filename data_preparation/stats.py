import pandas as pd
import os
import numpy as np
import glob
import re
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


# scraped_articles = pd.read_csv("../output/sentiment/scraped_sentiment_analysis.csv") 
# scraped = pd.DataFrame(scraped_articles, columns=['title', 'title-length', 'article', 'article-length', 'source', 'Positive Words', 'Negative Words', 'Article Sentiment', 'Title Sentiment'])



bbc_articles = pd.read_csv("../output/sentiment/bbc_sentiment_comparison_data.csv") 
bbc = pd.DataFrame(bbc_articles, columns=['title', 'title-length', 'article', 'article-length', 'category', 'source', 'Positive Words', 'Negative Words', 'Article Sentiment', 'Title Sentiment', 'Comparison'])


# Gather some statistics about the data - for use in the report only

# BBC data
bbc_average_article_length = bbc['article-length'].mean()
bbc_average_title_length = bbc['title-length'].mean()
bbc_count_sentiment = bbc.groupby(['Article Sentiment'])['title'].count().reset_index()

grouped = bbc.groupby('category')['title'].count()

print('Number of articles in each category: \n')
print(grouped)

labels = ['business', 'entertainment', 'politics', 'sports', 'technology']

_, ax = plt.subplots()
plt.title("Number of Articles in Each Category")
grouped.plot(kind='bar', ax=ax, figsize=(7, 5))


for p in ax.patches:
    ax.text(p.get_x() + 0.015,
            p.get_height() * 1.02,
            '{0:.0f}'.format(p.get_height()),
            color='black', rotation='horizontal', size='small')

plt.xticks(np.arange(5), labels, rotation='horizontal')
ax.set(xlabel='Article Category', ylabel='Number of Articles')


plt.savefig("Articles-per-Category.png")
# plt.show()

print("\nStatistics for BBC Data Set...\n")
print("Total number of articles: ", len(bbc.index))
print("Average article length: ", "%.0f" % bbc_average_article_length) #379 words
print("Average title length: ", "%.0f" % bbc_average_title_length) #5 words
print("Number of positively classified articles: ", "%.0f" % bbc_count_sentiment['title'].iloc[2])
print("Number of neutrally classified articles: ", "%.0f" % bbc_count_sentiment['title'].iloc[1])
print("Number of negatively classified articles: ", "%.0f" % bbc_count_sentiment['title'].iloc[0])



# # Scraped data
# scraped_average_article_length = scraped['article-length'].mean()
# scraped_average_title_length = scraped['title-length'].mean()
# scraped_count_sentiment = scraped.groupby(['Article Sentiment'])['title'].count().reset_index()
# scraped_count_source = scraped.groupby(['source'])['article'].count().reset_index()



# print("\nStatistics for Web Scraped Data Set...\n")
# print("Total number of articles: ", len(scraped.index))
# print("Average article length: ", "%.0f" % scraped_average_article_length) #379 words
# print("Average title length: ", "%.0f" % scraped_average_title_length) #5 words
# print("Number of positively classified articles: ", "%.0f" % scraped_count_sentiment['title'].iloc[2])
# print("Number of neutrally classified articles: ", "%.0f" % scraped_count_sentiment['title'].iloc[1])
# print("Number of negatively classified articles: ", "%.0f\n" % scraped_count_sentiment['title'].iloc[0])
# print("Number of articles scraped from each news outlet:\n")
# print(scraped_count_source.to_csv(sep='\t', index=False, columns=['article', 'source']))


