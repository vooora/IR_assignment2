import math
import nltk
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
def process_qf_file(lines):
    
    tf_dict = defaultdict(dict)
    doc_ids = set()  
    for line in lines:
        doc_id, term, freq = line.strip().split('\t')
        doc_ids.add(doc_id)  
        tf_dict[doc_id][term] = int(freq)
    return tf_dict, doc_ids

def process_df_file(lines):
   
    df_dict = defaultdict(int)
    for line in lines:
        term, doc_freq = line.strip().split('\t')
        df_dict[term] = int(doc_freq)
    return df_dict

def process_nnn_file(lines):
    nnn_dict = defaultdict(list)
    for line in lines:
        qid,a,docid,b,c,d = line.strip().split(' ')
        nnn_dict[qid].append(docid)
    return nnn_dict



def process_query_file(lines,qids):
    query_dict = defaultdict(int)
    for line in lines:
        qid, query = line.strip().split('\t')
        qids.append(qid)
        query_dict[qid] = query
    return query_dict

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

def create_vector_query(query_dict,df_list):
    n=20011
    q_final=[]
    for qid in query_dict:
        q_vec=[0] * n
        for term in qid:
            term=term.lower()
            if not term.isalnum():
                continue
            if(term in df_list):
                index=df_list.index(term)
                q_vec[index]+=1
        q_final.append(q_vec)
    return q_final

def create_vector_doc(tf_dict,df_list):
    n=20011
    d_final=[]
    for did in tf_dict:
        d_vec=[0] * n
        for term in did:
            term=term.lower()
            if not term.isalnum():
                continue
            freq=tf_dict[did].get(term,0)
            index=df_list.index(term)
            d_vec[index]=freq
        d_final.append(d_vec)
    return d_final
def create_doc_vector_dict(doc_ids, d_final):
    return dict(zip(doc_ids, d_final))

# def rocchio(alpha,beta,gamma,n,k,q_final,d_final):#get n
#     qm_final=[]
#     for q_vec in q_final:
#         qm_vec = [x * alpha for x in q_vec]
#         result = [sum(elements) for elements in zip(*d_final)] #has to be done for only relevant docs then nonrelevant docs
#         qm_vec+=[x * (beta/k) for x in result]
#         qm_vec-=[x * (beta/n-k) for x in result]#ur mom
#         qm_final.append(qm_vec)
#     return qm_final

import numpy as np

def rocchio(alpha, beta, gamma, n, k, q_final, doc_vector_dict, nnn_dict,qids):
    qm_final = []
    for qid, q_vec in zip(qids, q_final):  
        q_vec = np.array(q_vec)
        relevant_docs = nnn_dict.get(qid, [])[:20] 
        irrelevant_docs = [doc for doc in doc_vector_dict if doc not in relevant_docs]  

        if relevant_docs:
            rel_vecs = np.sum([np.array(doc_vector_dict[doc]) for doc in relevant_docs if doc in doc_vector_dict], axis=0)
        else:
            rel_vecs = np.zeros_like(q_vec)
        
        if irrelevant_docs:
            non_rel_vecs = np.sum([np.array(doc_vector_dict[doc]) for doc in irrelevant_docs if doc in doc_vector_dict], axis=0)
        else:
            non_rel_vecs = np.zeros_like(q_vec)

        qm_vec = q_vec * alpha + (beta / max(len(relevant_docs), 1)) * rel_vecs - (gamma / max(len(irrelevant_docs), 1)) * non_rel_vecs
        qm_final.append(qm_vec.tolist())

    return qm_final

def calculate_dot_products(query_vectors, document_vectors):
    query_matrix = np.array(query_vectors)
    doc_matrix = np.array(document_vectors)

    return np.dot(query_matrix, doc_matrix.T)  

def write_results_to_file(filename, qids, doc_ids, dot_products):
    with open(filename, 'w') as file:
        for i, q_id in enumerate(qids):
            for j, doc_id in enumerate(doc_ids):
                if i < dot_products.shape[0] and j < dot_products.shape[1]:
                    file.write(f"{q_id}\t{doc_id}\t{dot_products[i][j]:.4f}\n")





def main():
    file_df_list = 'IR_assignment2/nfcorpus/docdump/df.txt'

    df_list = []
    qids=[]
    doc_ids=[]

    with open(file_df_list, 'r') as file:
        for line in file:
            parts = line.split()
            if parts: 
                df_list.append(parts[0])

    tf_file_path = "IR_assignment2/nfcorpus/docdump/tf.txt"
    df_file_path = "IR_assignment2/nfcorpus/docdump/df.txt"
    query_file_path="IR_assignment2/nfcorpus/train.titles.queries"
    nnn_file_path="/Users/srivantv/Downloads/nfcorpus/q2/query_scores_nnn.txt"
    
    
    df_lines = read_file(df_file_path)
    query_lines = read_file(query_file_path)
    nnn_lines=read_file(nnn_file_path)
    nnn_dict=process_nnn_file(nnn_lines)
   
    df_dict = process_df_file(df_lines)
    query_dict=process_query_file(query_lines,qids)
  
    tf_lines = read_file(tf_file_path)
    qf_dict, doc_ids_set = process_qf_file(tf_lines)
    alpha=0.93
    beta=0.75
    gamma=0.25
    n=20011
    k=20
    doc_ids = list(doc_ids_set)
    q_vector = create_vector_query(query_dict, df_list)
    d_vector = create_vector_doc(qf_dict, df_list)
    doc_vector_dict = create_doc_vector_dict(doc_ids, d_vector)  

    qm_vectors = rocchio(alpha, beta, gamma, n, k, q_vector, doc_vector_dict, nnn_dict,qids)

    dot_products = calculate_dot_products(qm_vectors, d_vector)
    #print(doc_ids)
    output_file_path = "IR_assignment2/q3/result_rocchio.txt"
    write_results_to_file(output_file_path, qids, doc_ids, dot_products)
    print(f"Dot product results have been written to {output_file_path}")
    # print("hi")
    # print(qm_vectors)


if __name__=="__main__":
    main()
