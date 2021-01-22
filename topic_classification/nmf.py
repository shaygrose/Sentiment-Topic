import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.models.nmf import Nmf
from operator import itemgetter
import pickle
import sys
import os

from helper import topic_table, whitespace_tokenizer, unique_words, get_unstemmed_word, get_category_name, process_text


# using the SVC models to actually predict a category given the 8 topic words
def predict_category(df):

    with open('svc_tfidf.pickle', 'rb') as data:
        tfidf = pickle.load(data)

    with open('svc_model.pickle', 'rb') as data:
        model = pickle.load(data)

    # if we pass in a file path, read the csv, otherwise we're passing in a dataframe object
    if isinstance(df, str):
        df_topics = pd.read_csv(df)
    else:
        df_topics = df


    categories = []
    probabilities = []
    for idx, row in df_topics.iterrows():
        topic_matrix = tfidf.transform([row['topics']])
        topic_matrix = topic_matrix.toarray()
        svc_pred = model.predict(topic_matrix)[0]
        svc_pred_proba = model.predict_proba(topic_matrix)[0]
        category = get_category_name(svc_pred)
        probability = svc_pred_proba.max()*100
        categories.append(category)
        probabilities.append(probability)

    df_topics['predicted_category'] = categories
    df_topics['probability'] = probabilities

    return df_topics


def nmf(filename):

    # tf-idf
    min_df = 10
    max_df = 0.60
    max_features = 5000
    # sklearn-model nmf
 

    # get data
    newspaper_input = pd.read_csv(filename, na_filter=False)

    newspaper_input['processed_text'] = newspaper_input['article'].apply(
        process_text)
    texts = newspaper_input['processed_text']

    # create dictionary to pass as input to gensim model
    dictionary = Dictionary(newspaper_input['processed_text'])

    # filter out words that are above or below the thresholds set
    dictionary.filter_extremes(
        no_below=10,
        no_above=0.60,
        keep_n=5000
    )

    # convert to bag of words (corpus) to pass to gensim nmf model
    # [[(word_id, # times word appears in document),...],...]
    corpus = [dictionary.doc2bow(text) for text in texts]

    # find optimal number of topics using gensim NMF https://radimrehurek.com/gensim/models/nmf.html
    # testing topic numbers 10,15,20...55 to find best number to fit the data
    topic_nums = list(np.arange(10, 56, 5))
    coherence_scores = []
    for num in topic_nums:
        # initialize NMF model
        nmf = Nmf(
            corpus=corpus,
            num_topics=num,
            id2word=dictionary,
            chunksize=500, #Number of documents to be used in each training chunk
            passes=10, #Number of full passes over the training corpus
            kappa=0.1, #Gradient descent step size
            minimum_probability=0.001,
            w_max_iter=300, # Maximum number of iterations to train W per each batch
            w_stop_condition=0.0001, # If error difference gets less than that, training of W stops for the current batch
            h_max_iter=100,
            h_stop_condition=0.001,
            normalize=True,
            random_state=42
        )

        # initialize Coherence Model https://radimrehurek.com/gensim/models/coherencemodel.html
        # Calculate topic coherence for topic models
        cm = CoherenceModel(
            model=nmf,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )

        coherence_scores.append(round(cm.get_coherence(), 5))

    # get list of different topic numbers and their respective scores
    scores = list(zip(topic_nums, coherence_scores))
    # sort scores by score (not topic_num)
    scores = sorted(scores, key=itemgetter(1), reverse=True)
    # get the best number of topics
    best_num_topics, best_coherence_score = scores[0]
    # best_coherence_score = scores[0][1]
    print('scores: ',scores)
    
    print('num_topics: ',str(best_num_topics))
    print('coherence: ',str(best_coherence_score))
    # print(df.head())


    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), # unigrams and bigrams
        max_df=0.60,
        min_df=10,
        max_features=5000,
        preprocessor=' '.join
    )

    # fit+transform: returns document-term matrix (frequency of word i in document j)
    tfidf = tfidf_vectorizer.fit_transform(texts)
    # all the words we'll be looking at
    tfidf_fn = tfidf_vectorizer.get_feature_names()


    # grid search for best alpha, l1_ratio combination
    # measured by lowest sum squared residual
    # l1_ratio: regularization mixing parameter (0 => l2 penalty, 1 => l1 penalty, (0,1) => mixture)
    # alpha: constant that multiplies the regularization terms (0 => no regularization)
    squared_residuals = []
    params = {}
    models = []
    sorted_articles_dfs = []
    complete_topics_dfs = []
    alphas = list(np.arange(0.0, 1.2, 0.2))
    l1_ratios = list(np.arange(0.0, 1.2, 0.2))
    count_params = 0
    successes = 0
    count_successes = {}
    for a in alphas:
        for b in l1_ratios:
            # print('alpha: {}, l1_ratio: {}'.format(a,b))

            # learn a model
            nmf = NMF(
                n_components=best_num_topics,
                init='nndsvd',  # Non-negative double singular value decomposition
                max_iter=500,
                l1_ratio=b,
                solver='cd',  # coordinate descent
                alpha=a,
                tol=0.0001,  # 0.001
                random_state=42
            ).fit(tfidf)

            try:
                # transforms documents -> document-term matrix, transforms data according to model
                docweights = nmf.transform(tfidf)  # (articles x topics)

                # topic dataframe: (best_num_topics x 8)
                # (topic num : top 8 words that describe the topic)
                n_top_words = 8
                topic_df = topic_table(
                    nmf,
                    tfidf_fn,
                    n_top_words
                ).T

                # clean the topic words
                topic_df['topics'] = topic_df.apply(
                    lambda x: [' '.join(x)], axis=1)
                topic_df['topics'] = topic_df['topics'].str[0]
                topic_df['topics'] = topic_df['topics'].apply(
                    lambda x: whitespace_tokenizer(x))
                topic_df['topics'] = topic_df['topics'].apply(
                    lambda x: unique_words(x))
                topic_df['topics'] = topic_df['topics'].apply(
                    lambda x: [' '.join(x)])
                topic_df['topics'] = topic_df['topics'].str[0]

                # clean topic dataframe
                topic_df = topic_df['topics'].reset_index()
                topic_df.columns = ['topic_num', 'topics']

                topics = topic_df[['topic_num', 'topics']]

                # assign topics to each article
                title = newspaper_input['title'].tolist()
                df_temp = pd.DataFrame({
                    'title': title,
                    'topic_num': docweights.argmax(axis=1)
                })
                merged_topic = df_temp.merge(
                    topic_df,
                    on='topic_num',
                    how='left'
                )
                complete_df = merged_topic.merge(
                    newspaper_input,
                    on='title',
                    how='left'
                )

                # complete_df = complete_df.drop('processed_text', axis=1)

                # maybe unecessary ?
                complete_df = complete_df.drop_duplicates(subset=['title'])
                sorted_articles = complete_df.sort_values(by=['topic_num'])

                # get num articles per topic
                num_articles_per_topic = []
                for topic in range(best_num_topics):
                    count = 0
                    for index, row in sorted_articles.iterrows():
                        if row['topic_num'] == topic:
                            count += 1
                    num_articles_per_topic.append(count)

                # keep track of how many articles are given each topic
                topics['num_articles'] = num_articles_per_topic

                # matrices from nmf (A = WH)
                mat_A = tfidf_vectorizer.transform(texts)
                mat_W = nmf.components_
                mat_H = nmf.transform(mat_A)

                # residuals: measurement of how well the topics approximate the data (observed value - predicted value)
                # 0 -> topic perfectly predicts data
                # residual = Frobenius norm tf-idf weights (A) - coefficients of topics (H) X coefficients of topics (W)
                r = np.zeros(mat_A.shape[0])  # num articles
                for row in range(mat_A.shape[0]):
                    r[row] = np.linalg.norm(
                        mat_A[row, :] - mat_H[row, :].dot(mat_W), 'fro')

                sum_sqrt_res = round(sum(np.sqrt(r)), 3)
                squared_residuals.append(sum_sqrt_res)

                # add avg residual column to topics
                complete_df['resid'] = r
                sorted_articles = complete_df.sort_values(by=['topic_num'])
                resid_data = complete_df[[
                    'topic_num', 'resid'
                ]].groupby('topic_num').mean().sort_values(by='resid')
                complete_topics = topics.merge(
                    resid_data,
                    on='topic_num',
                    how='left'
                )

                # save results
                sorted_articles_dfs.append(sorted_articles)
                complete_topics_dfs.append(complete_topics)
                models.append(nmf)

                count_successes[count_params] = successes
                successes += 1

            except Exception as e:
                # print('test {}, error occurred'.format(count_params))
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)

            # print('test {} complete'.format(count_params))
            params[count_params] = (a, b)
            count_params += 1

    # find best params
    params_test = np.arange(36)
    resid_scores = list(zip(params_test, squared_residuals))
    resid_scores = sorted(resid_scores, key=itemgetter(1))
    best_params = resid_scores[0][0]
    print('test #{} had best residual score'.format(best_params))
    print('params: a={}, b={}'.format(
        params[best_params][0], params[best_params][1]))
    print('residual scores: {}'.format(resid_scores))

    best_articles = sorted_articles_dfs[count_successes[best_params]]
    best_topics = complete_topics_dfs[count_successes[best_params]]

    # call function that uses svc model to predict category based on topic words
    best_topics = predict_category(best_topics)

    # save best topics
    for idx, row in best_topics.iterrows():
        new_words = ''
        topics_itr = row['topics'].split()
        for word in topics_itr:
            new_words += get_unstemmed_word(word)
            new_words += ' '
        best_topics.at[idx, 'topics'] = new_words

    categories = []
    for idx, row in best_articles.iterrows():
        topic_num = row['topic_num']
        topics = best_topics.at[topic_num, 'topics']
        categories.append(best_topics.at[topic_num, 'predicted_category'])
        best_articles.at[idx, 'topics'] = topics
    best_articles['predicted_category'] = categories


    best_articles = best_articles.drop('processed_text', axis=1)
    best_articles = best_articles.drop('Unnamed: 0', axis=1)

    best_articles.to_csv(
        '../output/topic/articles_with_nmf_topics.csv', header=True, index=False)
    best_topics.to_csv('../output/topic/nmf_generated_topics.csv', header=True, index=False)

    # save model
    with open('nmf_model.pickle', 'wb') as output:
        pickle.dump(models[best_params], output)

    with open('nmf_tfidf.pickle', 'wb') as output:
        pickle.dump(tfidf_vectorizer, output)


if __name__ == '__main__':
    nmf('../input/bbc_articles.csv')

    prediction = predict_category('../output/topic/articles_with_nmf_topics.csv')
    prediction.to_csv('../output/topic/articles_with_predicted_categories.csv', index=False)
