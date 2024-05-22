from queries_to_2d import convert_queries_to_2d_array
from index_to_dict import build_tf_index
import math
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


queries, qids = convert_queries_to_2d_array('IR_assignment2/nfcorpus/dev.vid-titles.queries')
tf_index = build_tf_index("IR_assignment2/nfcorpus/tf.txt")

def cosine_normalize(array):
    array = np.array(array)
    norm = np.linalg.norm(array)
    if norm == 0:
        return array
    return array / norm

def stem_and_lemmatize(tokens):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(stemmer.stem(token)) for token in tokens]

def score_documents_nnn(query, tf_index):
    scores = {}
    normalized_query = stem_and_lemmatize(query)
    for term in normalized_query:
        for doc_id, terms in tf_index.items():
            if term in terms:
                if doc_id not in scores:
                    scores[doc_id] = 0
                scores[doc_id] += terms[term]
    return scores

query = ["date", "sugar"] 

def doccument_frequency(tf_index):
    df = {}
    for doc_id, terms in tf_index.items():
        for term in terms:
            if term not in df:
                df[term] = 0
            df[term] += 1
    return df

def inverse_document_frequency(df):
    idf = {}
    for term, freq in df.items():
        idf[term] = 1 + math.log(len(tf_index) / freq)
    return idf

df = doccument_frequency(tf_index)
df_inv = inverse_document_frequency(df)

def score_documents_ntn(query, tf_index):
    scores = {}
    normalized_query = stem_and_lemmatize(query)
    for term in normalized_query:
        for doc_id, terms in tf_index.items():
            if term in terms:
                if doc_id not in scores:
                    scores[doc_id] = 0
                scores[doc_id] += terms[term]*df_inv[term]
    return scores

def score_documents_ntc(query, tf_index):
    scores = {}
    normalized_query = stem_and_lemmatize(query)
    for doc_id, terms in tf_index.items():
        score = []
        for term in normalized_query:
            if term in terms:
                score.append(terms[term]*df_inv[term])
        score = cosine_normalize(score)
        if doc_id not in scores:
            scores[doc_id] = 0
        scores[doc_id] = sum(score)
    return scores

sc = {}
count = 0
for q_id in queries:
    sc[qids[count]] = score_documents_ntc(q_id, tf_index)
    count += 1

def write_scores_to_file(scores_dict, query_ids, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        for qid in query_ids:
            sorted_scores = sorted(scores_dict[qid].items(), key=lambda x: x[1], reverse=True)
            for doc_id, score in sorted_scores:
                file.write(f'{qid} Q0 {doc_id} 0 {score} STANDARD\n')

write_scores_to_file(sc, qids, 'IR_assignment2/q2/output.txt')
