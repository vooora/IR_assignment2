

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

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def tokenize(text):
    
    return [word for word in word_tokenize(text.lower()) if word not in stop_words]

def stem_and_lemmatize(tokens):
    
    return [lemmatizer.lemmatize(stemmer.stem(token)) for token in tokens]

def read_file(file_path):
  
    with open(file_path, 'r') as file:
        return file.readlines()

def process_document_file(lines):
    doc_lengths = {}
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) < 3:
            print(f"Skipping line: {line}")
            continue
        doc_id = parts[0]
        content = ' '.join(parts[2:])  
        tokens = stem_and_lemmatize(tokenize(content))
        doc_lengths[doc_id] = len(tokens)
    return doc_lengths, len(doc_lengths)


def process_tf_file(lines):
    tf_dict = defaultdict(dict)
    for line in lines:
        doc_id, term, frequency = line.strip().split('\t')
        tf_dict[doc_id][term] = int(frequency)
    return tf_dict

def process_df_file(lines):
    
    df_dict = defaultdict(int)
    for line in lines:
        term, doc_freq = line.strip().split('\t')
        df_dict[term] = int(doc_freq)
    return df_dict



def findmc(term, tf_dict):
    count = 0
    tot_count = 0
    for doc_id, terms in tf_dict.items():
        if term in terms:
            count += terms[term]
        tot_count += sum(terms.values())
    return count / tot_count if tot_count > 0 else 0


def main():
    doc_file_path = "IR_assignment2/nfcorpus/raw/doc_dump.txt"
    tf_file_path = "IR_assignment2/nfcorpus/docdump/tf.txt"
    df_file_path = "IR_assignment2/nfcorpus/docdump/df.txt"
    queries_file_path = "IR_assignment2/nfcorpus/test.titles.queries"

    doc_lines = read_file(doc_file_path)
    tf_lines = read_file(tf_file_path)
    df_lines = read_file(df_file_path)
    queries_lines = read_file(queries_file_path)

    doc_lengths, N = process_document_file(doc_lines)
    tf_dict = process_tf_file(tf_lines)
    df_dict = process_df_file(df_lines)

    lam = 0.7
    results = defaultdict(dict)
    print("hi")

    for line in queries_lines:
        query_id, query = line.strip().split('\t')
        query_terms = stem_and_lemmatize(tokenize(query))
        for docid in doc_lengths:
            product = 1
            for term in query_terms:         
                sum=0
                tf = tf_dict[docid].get(term, 0) if docid in tf_dict else 0
                md = tf / doc_lengths[docid] if doc_lengths[docid] > 0 else 0
                mc = findmc(term, tf_dict)
                sum+=(lam*md)+(1-lam)*mc
                product = product*sum
            results[query_id][docid] = product

    output_file_path = "IR_assignment2/q4/lm_result.txt"

    with open(output_file_path, 'w') as output_file:
        for query_id, doc_scores in results.items():
            sorted_scores = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
            for doc_id, score in sorted_scores:
                output_file.write(f"{query_id}\t{doc_id}\t{score:.4f}\n")

if __name__ == "__main__":
    main()          




