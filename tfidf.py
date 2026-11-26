import math
import pandas as pd

TOP_N = 10

# words in almost every article, so they don't tell topics apart
IGNORE = {'mamdani', 'zohran', 'york', 'nyc'}

# common words with no meaning
with open('data/stopwords.txt') as f:
    stopwords = set(f.read().split())

df = pd.read_csv('data/topics.csv')

# OTH and OTHER are the same thing, and "Other" is too small to use
df = df[~df['topic'].isin(['OTH', 'OTHER'])]
df['year'] = df['date'].str[:4]


def get_words(text):
    words = []
    for w in text.lower().split():
        w = w.strip('.,:;!?\'"()[]-’‘“”')
        if w.isalpha() and len(w) > 1 and w not in stopwords and w not in IGNORE:
            words.append(w)
    return words


def tfidf(articles):
    # count the words in each topic
    counts = {}
    for _, row in articles.iterrows():
        topic = row['topic']
        if topic not in counts:
            counts[topic] = {}
        text = str(row['title']) + ' ' + str(row['description'])
        for w in get_words(text):
            counts[topic][w] = counts[topic].get(w, 0) + 1

    # how many topics each word shows up in
    num_topics_with = {}
    for topic in counts:
        for w in counts[topic]:
            num_topics_with[w] = num_topics_with.get(w, 0) + 1

    num_topics = len(counts)
    top = {}
    for topic in counts:
        scores = {}
        for w in counts[topic]:
            tf = counts[topic][w]
            idf = math.log(num_topics / num_topics_with[w])
            if tf >= 2:
                scores[w] = tf * idf
        top[topic] = sorted(scores, key=scores.get, reverse=True)[:TOP_N]
    return top


for year in ['All', '2025', '2026']:
    print('=====', year, '=====')
    if year == 'All':
        result = tfidf(df)   # both years together
    else:
        result = tfidf(df[df['year'] == year])
    for topic in result:
        print(topic, ':', ', '.join(result[topic]))
    print()

    # save as a table: one column per topic
    table = pd.DataFrame({topic: result[topic] + [''] * (TOP_N - len(result[topic])) for topic in result})
    table.to_csv('output/top_words_' + year + '.csv', index=False)
