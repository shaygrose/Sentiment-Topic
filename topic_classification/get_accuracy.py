import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_fscore_support


def check(f):
    data = pd.read_csv(f)

    categorized_articles_num = data['predicted_category'].count()
    source_category = data['category'].count()


    grouped = data.groupby(
        ['category', 'predicted_category'])['title'].count().reset_index()

    categories = data[['category', 'predicted_category']]


    categories['match'] = categories['category'] == categories['predicted_category']

    # print(categories[(categories['match'] == False) & (categories['category'] == 'technology')])

    grouped_cat = categories.groupby(['category', 'predicted_category']).count().reset_index()

    print("How articles were classified according to category: \n")
    print(grouped_cat, "\n")

    category_labels = ['business', 'entertainment', 'politics', 'sports', 'technology']


    tech = grouped_cat[grouped_cat['category']=='technology']
    _, ax = plt.subplots()
    plt.title("How Technology Articles were Classified")
    tech.match.plot(kind='bar', ax=ax)
    ax.get_children()[4].set_color('g') 

    for p in ax.patches:
        ax.text(p.get_x() + 0.015,
                p.get_height() * 1.02,
                '{0:.0f}'.format(p.get_height()),
                color='black', rotation='horizontal', size='small')

    plt.xticks(np.arange(5), category_labels, rotation='horizontal')
    ax.set(xlabel='Predicted Category', ylabel='Number of Articles')
    colors = ['green', '#1f77b4']
    lines = [Line2D([0], [0], color=c, linewidth=4) for c in colors]
    labels = ['Correct', 'Incorrect']
    plt.legend(lines, labels)
    # plt.show()
    plt.savefig("../output/topic/Technology-Articles-Classifications.png")
    
    
    total = data.groupby('category').count()['title'].reset_index()

    correctly_classified = grouped[grouped['category'] ==
                                   grouped['predicted_category']]

    counts = correctly_classified.merge(
        total,
        on='category',
        how='left'
    )

    counts = counts.rename(columns={"title_x": "correctly_classified",
                                    "title_y": "total_articles"}, errors="raise")

    counts['percentage'] = (100 * counts['correctly_classified'] /
                            counts['total_articles']).round(2)

    print('Number of articles correctly classified in each category: \n')
    print(counts[['category', 'correctly_classified', 'total_articles', 'percentage']], "\n")

    # Graphing predicted vs actual categories for each category
    bx = counts[['correctly_classified', 'total_articles']].plot.bar(rot=0, color={"#1f77b4","green"})

    bx.set_title("Correctly Classified vs. Total Number of Articles")
    bx.set(xlabel='Article Category', ylabel='Number of Articles')
    plt.xticks(np.arange(5), category_labels, rotation='horizontal')
    for p in bx.patches:
        bx.text(p.get_x() + 0.015,
                p.get_height() * 1.02,
                '{0:.0f}'.format(p.get_height()),
                color='black', rotation='horizontal', size='small', va="center")

    bx.legend(loc="upper right")


    # plt.show()
    plt.savefig("../output/topic/Predicted-vs-Category.png")


    # Calculating precisionr/recall score

    y_true = np.array(data['category'])
    y_pred = np.array(data['predicted_category'])
    # print(y_true)
    # print(y_pred)
    score = precision_recall_fscore_support(
        y_true, y_pred, average=None)

    print("precision recall fscore: \n\n", score)



if __name__ == '__main__':
    check('../output/topic/articles_with_predicted_categories.csv')
