from flask import Flask,request,jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sparse
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import collaborative_filtering
from pandas.io.json import json_normalize
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
tfidf_matrix = sparse.load_npz("tfidf_matrix.npz")

books, ratings_data = collaborative_filtering.load_data()
model = collaborative_filtering.load_model()

user_mapper, book_mapper, user_inv_mapper, book_inv_mapper = collaborative_filtering.create_X(ratings_data)

book_title_mapper = dict(zip(books['title'], books['book_id']))
book_title_inv_mapper = dict(zip(books['book_id'], books['title']))

def book_by_id(book_idx): 
    book_id = book_inv_mapper[book_idx]
    book = books[books['book_id'].eq(book_id)]
    return book

def prepare_user_items(df):
    book_indexes = df['book_id'].apply(lambda x: book_mapper[x]).values
    rows = [0 for _ in book_indexes]
    rat = df.rating.values
    shape = (1, model.item_factors.shape[0])
    return coo_matrix((rat, (rows, book_indexes)), shape=shape).tocsr()

def recommend(user_items):
    recommendations,sim = model.recommend(0, user_items, recalculate_user=True)
    book_id = []
    titles =[]
    genres =[]
    for r in recommendations:
        book = book_by_id(r)
        print(book.title.values)
        book_id.append(book.book_id.values)
    df = pd.DataFrame(list(zip(book_id)),
               columns =['book_id'])
    return df



def recommendations_for_one(book_id,books):
    book_id = int(book_id)
    index = books[books.book_id.eq(book_id)].index[0]
    cosine_sim = cosine_similarity(tfidf_matrix,tfidf_matrix[index])
    cosine_sim_df = pd.DataFrame(cosine_sim, index=books['book_id'])
    cosine_sim_df.columns = ['sim']
    list(cosine_sim_df.columns)
    recommendations = cosine_sim_df['sim'].nlargest(n=100).sample(5)
    return books[books['book_id'].isin(recommendations.index.tolist())].to_dict('records')


app = Flask(__name__)
CORS(app) 
        


if __name__=='__main__':
    app.run(port = 5000, debug = True)

@app.route('/book', methods=['GET'])
def recommend_books():
    print(request.args.get('book_id'))
    res = recommendations_for_one(request.args.get('book_id'),books)
    return jsonify(res)




@app.route('/personalized', methods=['POST'])
def get_personalized_recommendations():
    req = request.json
    data = pd.DataFrame.from_records(req['ratedBooks'])
    print(data)
    user_items = prepare_user_items(data)
    recommendations = recommend(user_items)
    arr = recommendations.book_id.values
    ids = [item for sublist in arr for item in sublist]
    book_recommendations = books[books['book_id'].isin(ids)].to_dict('records')
    return jsonify(book_recommendations)
    
    

    


