# -*- coding: utf-8 -*-

# import Flask class from the flask module
from flask import Flask, request, render_template
# from sklearn.externals import joblib
import numpy as np
import pickle
from collections import defaultdict
from collections import Counter
import csv
import pandas as pd

# Create Flask object to run
app = Flask(__name__)

# Load the model from the file
file_name = open('XGB_SBPRS_model.pkl', 'rb')
sbprs = pickle.load(file_name)


@app.route('/')
def home():
    return 'Sentiment Based Product Recommendation System !!'


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # pd.read_csv('df_user_product.csv')
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

    pred = get_product_recommendation(user_product_map, product_user_map, request.form.values())

    return render_template('template/index.html', prediction_text='Recommended products are {}'.format(pred))


if __name__ == "__main__":
    # Start Application
    app.debug = True
    app.run()
