# lemtization _5
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w , pos="a") for w in text]
# df['text_to'] = df['text_to'].apply(lemmatize_text)



def Data_prepocessing(df):
  # Init the Wordnet Lemmatizer
    lemmatizer = WordNetLemmatizer()
   

    #caselower_3
    df['new_text'] = df['text'].str.lower()
    # remove stop words _1
    stop_words = set(stopwords.words('english'))  
    df['new_text'] = df['new_text'].apply(lambda x: " ".join([w for w in str(x).split() if not w in stop_words]))


    #remove punctuation_2
    df['new_text'] = df['new_text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    df['new_text'] = df['new_text'].str.replace('[^\w\s]','').str.lower().apply(lambda x: word_tokenize(x))
    df['new_text'] = df['new_text'].apply(lemmatize_text)
    
    #stemaization_6
    stemmer = SnowballStemmer("english")
    df['new_text'] = df['new_text'].apply(lambda x: [stemmer.stem(y) for y in x])
    
    return df

 #################################################################

#tf - idf

def dummy(tokens):
    return tokens

def Data_representation(df):
    vectorizer = TfidfVectorizer(tokenizer=dummy, preprocessor=dummy)
    X = vectorizer.fit_transform(df['new_text'][:1000])
    df1 = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out(), index=df['new_text'].index[:1000])
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    return df1

#################################################################

#inverted index


def inverted_index(df):
    # convert the 'new_text' column to string dtype
    df['new_text'] = df['new_text'].astype(str)

    # initialize the vectorizer
    vectorizer = CountVectorizer()

    # fit the vectorizer on the data
    vectorizer.fit(df['new_text'])

    # create the term-document matrix
    tf = vectorizer.transform(df['new_text'])

    # convert the term-document matrix into an inverted index
    terms = vectorizer.get_feature_names_out()
    inv_index = defaultdict(list)
    for i, term in enumerate(terms):
        doc_ids = list(tf[:,i].nonzero()[0])
        inv_index[term] = doc_ids

    # create a new dataframe with the inverted index
    df = pd.DataFrame({'term': list(inv_index.keys()), 'doc_ids': list(inv_index.values())})
    
    return df 


#################################################################

# ------ To Return doc_text by doc_id

def dum(id):
    
    dataset = ir_datasets.load("lotte/lifestyle/dev")
    doc_id = id
    doc_text = None
    for doc in dataset.docs_iter():
        if doc.doc_id == doc_id:
            doc_text = doc.text
            break
    df = pd.DataFrame({
        "doc_id": [doc_id],
        "text": [doc_text]
    })
    return doc_text



def getresults():
    dataset = ir_datasets.load("lotte/lifestyle/dev/forum")
    results = []
    query_counter = 0
    for query in dataset.queries_iter():
        query_id = query[0]
        query_text = query[1]
        query_results = []
        for qrel in dataset.qrels_iter():
            if qrel.query_id == query_id:
                doc_id = qrel.doc_id
                doc_text = dum(doc_id)
                query_results.append({
                   # "query_id": query_id,
                    "query_text": query_text,
                    #"doc_id": doc_id,
                    "doc_text": doc_text,
                   # "relevance": qrel.relevance
                })
        results.extend(query_results)
        query_counter += 1
        if query_counter >= 50:
            break
    return pd.DataFrame(results)


def load_dataset(i):
    if i == "1":
        dataset = ir_datasets.load("beir/quora")
        df1 = pd.DataFrame(dataset.docs)
        df1.head(1)
    else:
        dataset = ir_datasets.load("lotte/lifestyle/dev")
        df1 = pd.DataFrame(dataset.docs)
        df1.head(20)
        
    return df1 ;     

def query_proccess(query):
    data= []
    data.append({'doc_id': 1, 'text': query})
    q_df = pd.DataFrame(data, columns=['doc_id', 'text'])
    q_df = Data_prepocessing(q_df)
    return q_df


def temp(word , i_df , matching_qids):
    qids = []
    # Retrieve the list of question IDs for this word from the inverted index
    if i_df['term'].isin([word]).any():
        qids = i_df.loc[i_df['term'] == word, 'doc_ids'].iloc[0]
        
    # Add the matching question IDs to the set
    matching_qids.update(qids)
    return matching_qids



def resaults(i_df , q_df , df , my_query):    
    # Initialize a set to store the matching question IDs
    matching_qids = set()
    res_df = pd.DataFrame()
    res = []
    list = ast.literal_eval(q_df.loc[0, 'new_text'])
    for word in list:
        matching_qids = temp(word , i_df , matching_qids)
        for qid in matching_qids:
            res.append({'qid':qid, 'doc': df.loc[qid, 'text'] })

    res_df = pd.DataFrame(res, columns=['qid', 'doc' ])
    return res_df

def qrel_res(i):
    qrel_df = pd.DataFrame()
    qrel_data = []
    # load the qrel results into a DataFrame
    if i == 1:
        dataset = ir_datasets.load("beir/quora/dev")
    else:
        dataset = ir_datasets.load("lotte/lifestyle/dev/forum")

    for qrel in dataset.qrels_iter():
        qrel_data.append({'qid':qrel.query_id, 'docid': qrel.doc_id , 'relevance' : qrel.relevance})

    qrel_df = pd.DataFrame(qrel_data, columns=['qid', 'docid' ,'relevance' ])
    return qrel_df
  
def cosine_sim(i_df , qi_df):
    vectorizer = TfidfVectorizer()
    doc1 = vectorizer.fit_transform(i_df.head())
    doc2 = vectorizer.fit_transform(qi_df.head())
    cosine_sim = cosine_similarity(doc1, doc2)
    return cosine_sim


def evaluation(res_df ,qrel_df ):
    true_labels = []
    predicted_labels = []
    for i in res_df['qid']:
        true_labels.append(int(i))
    for k in qrel_df['docid']:
        predicted_labels.append(int(k))

    matched_labels = [label for label in true_labels if label in predicted_labels]
    #matched_labels = [label for label in predicted_labels if label in true_labels]

    #print(len(matched_labels))


    # create a new array of zeros with the same length as the input arrays
    result = [0] * len(true_labels)

    # iterate over each element in array1 and check whether it is present in array2
    for i in range(len(true_labels)):
        if true_labels[i] in matched_labels:
            result[i] = true_labels[i]

    # print the result

    # define the positive label
    matched_labels

    pos_label =1078

    # count the true positives and false positives
    true_positives = 0
    false_positives = 0
    for true_label, result in zip(true_labels, result):
        if result == pos_label:
            if true_label == pos_label:
                true_positives += 1
            else:
                false_positives += 1

    # calculate the precision
    precision = true_positives / (true_positives + false_positives)
    temp_df = pd.DataFrame()
    temp = []
    temp.append({'True_positives':true_positives, 'False_positives': false_positives , 'Precision' : precision})
    temp_df = pd.DataFrame(res, columns=['docid', 'doc' ,'qid'])
    return temp_df

# file: app.py

from app import cosine_sim
from app import qrel_res
from app import evaluation
from app import resaults
from app import temp
from app import query_proccess
from app import load_dataset
from app import lemmatize_text
from app import Data_prepocessing
from app import dummy
from app import Data_representation
from app import inverted_index
from app import dum
from app import getresults
from flask import Flask, jsonify , request
import json
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import string
import ast
import re
from nltk.stem import WordNetLemmatizer 
from nltk.stem.snowball import SnowballStemmer
from sklearn import preprocessing
import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import ir_datasets
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
from scipy.sparse import csr_matrix
import math


app = Flask(__name__)
df1 = pd.DataFrame()
i_df = pd.DataFrame()

    
@app.route('/choice_dataset/<string:i>', methods=['POST'])
def all_results(i):
    global df1
    global i_df

    data = []
    if  df1.empty:
        df1 = load_dataset(i)
        df1 = Data_prepocessing(df1.head(1000))   
        
    if  i_df.empty:    
        i_df = inverted_index(df1.head(1000))
    
    body = request.form
    my_query = body['query']
    q_df = query_proccess(my_query)
    qi_df = inverted_index(q_df)
    
    res = resaults(i_df , q_df , df1 , my_query)
    qrel_df =  qrel_res(i)
    cos = cosine_sim(i_df , qi_df)

    data1 = []
    for index, row in res.iterrows():
        data1.append({row['qid']:str(row['doc'])})
    return jsonify(data1)      
                                

@app.route('/Data_prepocessing/<string:i>')
def prepocessing(i):
    global df1
    df1 = load_dataset(i)
    df1 = Data_prepocessing(df1.head(1000))        
    
    data = []
    for index, row in df1.iterrows():
        data.append({row['doc_id']:row['new_text']})
    return jsonify(data)

@app.route('/Data_representation/<string:i>')
def representation(i):
    global df1
    df1 = load_dataset(i)
    df1 = Data_prepocessing(df1)
    return jsonify(Data_representation(df1.head(1000)).to_string())

@app.route('/inverted_index/<string:i>')
def inverted(i):
    global df1
    global i_df
    df1 = load_dataset(i)
    df1 = Data_prepocessing(df1.head(1000))
    i_df =inverted_index(df1.head(1000))
    data1 = []
    for index, row in i_df.iterrows():
        data1.append({row['term']:str(row['doc_ids'])})
    return jsonify(data1)


@app.route('/query_processing/<string:i>', methods=['POST'])
def process_form(i):
    data = []
    body = request.form
    my_query = body['query']
    q_df = query_proccess(my_query)
    data =[]
    for index, row in q_df.iterrows():
        data.append({row['doc_id']:row['new_text']})
    return jsonify(data)

@app.route('/query_indexing/<string:i>', methods=['POST'])
def query_index(i):
    data = []
    body = request.form
    my_query = body['query']
    q_df = query_proccess(my_query)
    i_df = inverted_index(q_df)
    data1 = []
    for index, row in i_df.iterrows():
        data1.append({row['term']:str(row['doc_ids'])})
    return jsonify(data1)


@app.route('/')
def hello():
    return "Hello, !"

if __name__ == '__main__':
    app.run(host='192.168.1.106', port=8080)
#################################################################
