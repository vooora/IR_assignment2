import os
import re
import nltk
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

def preprocess_text(text):
    text = re.sub(r'[\.\n,\?\t\b]+', ' ', text)  
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  
    return text.lower().strip()

def tokenize(text):
    text = preprocess_text(text)
    return word_tokenize(text)

def stem_and_lemmatize(tokens):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(stemmer.stem(token)) for token in tokens]

def add_to_index(index, doc_tf, document_id, text):
    tokens = tokenize(text)
    processed_tokens = stem_and_lemmatize(tokens)
    token_counts = Counter(processed_tokens)
    for token, count in token_counts.items():
        index[token].add(document_id)
        doc_tf[(document_id, token)] = count

def build_indices(doc_dump_path):
    index = defaultdict(set)
    doc_tf = {}
    with open(doc_dump_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                doc_id, _, title, abstract = parts[0], parts[1], parts[2], parts[3]
                full_text = title + " " + abstract
                add_to_index(index, doc_tf, doc_id, full_text)
    return index, doc_tf

def calculate_df(index):
    df = {token: len(doc_ids) for token, doc_ids in index.items()}
    return df

def save_indices(index, doc_tf, df, index_file_path, tf_file_path, df_file_path):
    with open(index_file_path, 'w', encoding='utf-8') as file:
        for token, document_ids in index.items():
            file.write(f"{token}\t{' '.join(document_ids)}\n")
    with open(tf_file_path, 'w', encoding='utf-8') as file:
        for (doc_id, token), frequency in doc_tf.items():
            file.write(f"{doc_id}\t{token}\t{frequency}\n")
    with open(df_file_path, 'w', encoding='utf-8') as file:
        for token, frequency in df.items():
            file.write(f"{token}\t{frequency}\n")

def main():
    doc_dump_path = 'IR_assignment2/q1/index_updated.txt'
    index_file_path = 'IR_assignment2/q1/index_updated.txt'
    tf_file_path = 'IR_assignment2/nfcorpus/tf.txt'
    df_file_path = 'IR_assignment2/nfcorpus/df.txt'
    
    index, doc_tf = build_indices(doc_dump_path)
    df = calculate_df(index)
    
    save_indices(index, doc_tf, df, index_file_path, tf_file_path, df_file_path)
    print("Indexing completed.")

if __name__ == "__main__":
    main()
