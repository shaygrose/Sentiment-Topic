## Data Scraping

_Steps to Run_

1. Install all requirements listed in requirements.txt

   > `pip3 install -r requirements.txt`

2. Ensure you're running python 3

   > `python3 --version` or `which python3`

3. Once all dependencies are installed, run the cleaning script that takes all the separate bbc
   text files and converts them into 1 csv file, if not done so already. This file relies on there being a bbc-data folder containing 
   5 subfolders with text files for all the different categories, which we have included.
   Inside the data_preparation folder:

   > `python3 clean_data.py`  
   > Or just `python` depending on how you have your aliases set up.

4. This should create a file called `bbc_articles.csv` into the input folder.  
   _These are the articles that will be used for sentiment analysis and topic classification, since they are labelled._

5. If you want, run the news website scraper in the data_preparation directory. ** Warning: this will take 2+ hours **

   > `python3 scrape_data.py`

6. This should create a file called `scraped_news.csv` in the input folder.  


7. To run sentiment analysis, open `sentiment.py` in the sentiment_analysis folder and make sure it is reading the bbc data csv. Running it with the bbc data will produce a graph, with scraped data it will not. It takes 3-5 minutes to run depending on which data set you are using. This will output a number of csv's and graphs into the output/sentiment folder.
   _There is commented code to change between reading the bbc data or the scraped data

   > `python3 sentiment.py`

8. To see some statistics about the two datasets, run `stats.py` inside the data_preparation directory which outputs information to the terminal.

   > `python3 stats.py`

9. To train the SVC model using the clean data obtained from previous steps, run `SVC.py` inside the topic_classification directory. This will output two pickle files into the topic classification folder. This takes around 30 minutes, so if you don't want to wait, the model files can be downloaded from https://drive.google.com/drive/folders/1myTIUoPaSt1ujO9rfDI6IJIVPdYy5lt2?usp=sharing.

   > `python3 svc.py`

10. To train the two NMF models, run `nmf.py` inside the topic_classification directory. This will output two pickle files into the topic classification folder. This takes around 20 minutes, so if you don't want to wait, the model files can be downloaded from https://drive.google.com/drive/folders/1myTIUoPaSt1ujO9rfDI6IJIVPdYy5lt2?usp=sharing.

   > `python3 nmf.py`


10. To check the performance of the topic classification and see how accurate it is, run `get_accuracy.py` inside the topic_classification directory. This will output numerous graphs into the output folder. Predicted-vs-Category is most interesting.
    > `python3 get_accuracy.py`

11. Access the Map-based user application we made which allow users to filter articles based on topic and sentiment of news articles. 
      > https://teletubbies-front-end.vercel.app/

   
As a footnote, `helper.py` is included because it contains functions we created which are used for cleaning and processing text in multiple places.
