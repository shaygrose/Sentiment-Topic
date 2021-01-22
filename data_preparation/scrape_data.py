import newspaper
import pandas as pd
from newspaper import Article
from newspaper import fulltext
from helper import process_text

cols = ["source", "title", "article"]

urls = {
    "cbc": "https://www.cbc.ca/news",
    "ctv": "https://www.ctvnews.ca/",
    "national_post": "https://nationalpost.com/",
    "toronto_sun": "https://torontosun.com/category/news",
    "toronto_star": "https://www.thestar.com/",
    "cp_24": "https://www.cp24.com/",
    "maple_ridge": "https://www.mapleridgenews.com/local-news/",
    "tri_city": "https://www.tricitynews.com/",
    "langley_advance_times": "https://www.langleyadvancetimes.com/local-news/",
    "abbotsford": "https://www.abbynews.com/local-news/",
    "chilliwack_progress": "https://www.theprogress.com/local-news/",
    "delta_optimist": "https://www.delta-optimist.com/local-news",
    "north_delta_reporter": "https://www.northdeltareporter.com/local-news/",
    "surrey_now_leader": "https://www.surreynowleader.com/local-news/",
    "vancouver_observer": "https://www.vancouverobserver.com/",
    "vancouver_courier": "https://www.vancourier.com/",
    "georgia_straight": "https://www.straight.com/news",
    "north_shore": "https://www.nsnews.com/",
    "richmond": "https://www.richmond-news.com/",
    "richmond_senitel": "https://richmondsentinel.ca/",
    "burnaby_now": "https://www.burnabynow.com/",
    "new_west_record": "https://www.newwestrecord.ca/",
    "bowen_island_undercurrent": "https://www.bowenislandundercurrent.com/"
}

data = []
for url in urls:

    news = newspaper.build(urls[url], memoize_articles=False)
    # memoize = false because we don't want to cache articles already seen

    print(url + " news...")
    print(news.size())

    # loop through all articles scraped and save into dataframe for each news outlet
    for article in news.articles:
        try:
            article.download()
            article.parse()
        except newspaper.article.ArticleException:
            pass
        # append each article and its fields into list
        data.append([news.brand, article.title, article.text])


articles = pd.DataFrame(data, columns=cols)
articles = articles.fillna('')
# drop articles with the same title (keeps the first entry by default)
articles = articles.drop_duplicates(subset="title")

# drop articles with the same text (keeps the first entry by default)
articles = articles.drop_duplicates(subset="article")

# convert text column to string
articles["article"] = articles["article"].astype(str)


# this line will drop any rows that have titles which match any of the strings in the list
articles = articles[~articles["title"].isin(
    ["Terms of Use", "Privacy Policy", "-", "- The Weather Network", "Public Appearances"])]

# drops articles with empty body text
articles = articles[~articles["article"].isin(["nan"])]

# drops articles with empty title
articles = articles[~articles["title"].isin(["nan"])]

# only keeps rows that have an outlet which we have scraped
# MODIFY THIS LINE WHEN YOU ADD NEW OUTLETS
articles = articles[articles["source"].isin(["thestar", "The Record", "cbc", "ctvnews", "nationalpost", "torontosun", "cp24", "mapleridgenews", "tricitynews", "langleyadvancetimes", "abbynews", "theprogress", "delta-optimist",
                                             "northdeltareporter", "surreynowleader", "vancouverobserver", "vancourier", "srtraight", "nsnews", "richmond-news", "burnabynow", "richmondsentinel", "newwestrecord", "bowenislandundercurrent"])]


articles['title-length'] = articles['title'].str.split(' ').str.len()
articles['article-length'] = articles['article'].str.split(' ').str.len()


columns = ['title', 'title-length', 'article', 'article-length', 'source']

articles = articles[columns]
articles.to_csv("../input/scraped_news.csv", index=False)


stemmed = articles
stemmed['article'] = stemmed['article'].apply(process_test)
stemmed.to_csv("../input/stemmed_scraped_news.csv", index=False)


