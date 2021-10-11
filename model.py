from collections import defaultdict
from collections import Counter
import pandas as pd
import time
import csv
from flask import request
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv('sample30.csv')

# Dropping the rows having null values for reviews text
df = df.dropna(subset=['reviews_text'])

# XGBoost Model
seed = 71
all_text = df['reviews_text']
y = df['reviews_rating']
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 3))
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(all_text)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(all_text)

train_features = hstack([train_char_features, train_word_features])

X_train, X_test, y_train, y_test = train_test_split(train_features, y, test_size=0.3, random_state=seed)
print('X_train', X_train.shape)
print('y_train', y_train.shape)
print('X_test', X_test.shape)
print('y_test', y_test.shape)

time1 = time.time()

xgb = xgb.XGBClassifier(n_jobs=1)
xgb.fit(X_train, y_train)
preds2 = xgb.predict(X_test)

time_taken = time.time() - time1
print('Time taken: {:.2f} seconds'.format(time_taken))
print("XGBoost Model accuracy", accuracy_score(preds2, np.array(y_test)))
print("XGBoost Model accuracy", accuracy_score(preds2, y_test))
print(classification_report(preds2, y_test))
print(confusion_matrix(preds2, y_test))

# Recommendation System using cosine similarity
df_user_product = df[['reviews_username', 'name', 'reviews_text', 'reviews_title']]
df_user_product

df_user_product['tags'] = df_user_product['reviews_title'] + df_user_product['reviews_text']
df_user_product_final = df_user_product.drop(columns=['reviews_title', 'reviews_text'])
df_user_product_final.dropna(inplace=True)

cv = CountVectorizer(max_features=1000, stop_words='english')
vector = cv.fit_transform(df_user_product_final['tags']).toarray()

time1 = time.time()
similarity = cosine_similarity(vector)
time_taken = time.time() - time1
print('Time Taken: {:.2f} seconds'.format(time_taken))
similarity
df_user_product_final[df_user_product_final['reviews_username'] == 'jess'].index[0]


def recommend():
    # index = df_user_product_final[df_user_product_final['reviews_username'] == product].index[0]
    # distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    # for i in distances[1:6]:
    #     print(df_user_product_final.loc[i[0], "name"])

    user_product_map = defaultdict(list)
    product_user_map = defaultdict(list)
    with open('df_user_product.csv', 'r') as csvfile:
        w = csv.reader(csvfile, delimiter=',')
        for row in w:
            user_product_map[row[0]].append(row[1])
            product_user_map[row[1]].append(row[0])

    def get_product_recommendation(user_product_map, product_user_map, u1):
        biglist = []
        for m in user_product_map[u1]:  # For the products a specific user likes
            for u in product_user_map[m]:  # Getting other users who liked those products
                biglist.extend(user_product_map[u])  # Finding the other products those "similar folks" most liked
        return Counter(biglist).most_common(20)  # Returning tuples of (most common id, count)

    pred = get_product_recommendation(user_product_map, product_user_map, "joshua")


recommend()
