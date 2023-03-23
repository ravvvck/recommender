
import pandas as pd
from scipy.sparse import csr_matrix
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sparse
import implicit
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
from scipy.sparse import coo_matrix


def create_matrix(data, user_col, item_col, rating_col):
    rows = data[user_col].cat.codes
    cols = data[item_col].cat.codes
    rating = data[rating_col]
    ratings = csr_matrix((rating, (rows, cols)))
    ratings.eliminate_zeros()
    return ratings

def load_data():
    books = pd.read_csv('books_decentreads.csv', usecols = ['book_id','title', 'genres', 'cover_image_url'], encoding="utf-8",  delimiter=",", on_bad_lines='skip')
    ratings_data = pd.read_csv('interactions_decentreads.csv')
    ratings_data[['user_id', 'book_id',]] = ratings_data[['user_id', 'book_id',]].astype('category')
    return books, ratings_data

def save_model(model):
    joblib.dump(model, 'model.pkl')

def load_model():
    model = joblib.load('model.pkl')
    return model

def retrain_model(sparse_item_customer):
    model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=50)
    model.fit(sparse_item_customer)
    customer_vecs = sparse.csr_matrix(model.user_factors)
    item_vecs = sparse.csr_matrix(model.item_factors)
    return model,customer_vecs, item_vecs





def create_X(df):
    N = df['user_id'].nunique()
    M = df['book_id'].nunique()
    user_mapper = dict(zip(np.unique(df["user_id"]), list(range(N))))
    book_mapper = dict(zip(np.unique(df["book_id"]), list(range(M))))
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["user_id"])))
    book_inv_mapper = dict(zip(list(range(M)), np.unique(df["book_id"])))    
    return  user_mapper, book_mapper, user_inv_mapper, book_inv_mapper









