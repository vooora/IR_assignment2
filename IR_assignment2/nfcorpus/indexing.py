import os

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')  # For WordNet lemmatizer compatibility with more languages.

from collections import defaultdict, Counter

from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure you've downloaded the necessary NLTK resources
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

def tokenize(text):
    # Tokenize text using NLTK's word_tokenize
    return word_tokenize(text.lower())

def stem_and_lemmatize(tokens):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    # Apply stemming and then lemmatization
    return [lemmatizer.lemmatize(stemmer.stem(token)) for token in tokens]

def add_to_index(index, doc_tf, document_id, text):
    tokens = tokenize(text)
    processed_tokens = stem_and_lemmatize(tokens)
    token_counts = Counter(processed_tokens)
    for token, count in token_counts.items():
        index[token].add(document_id)
        doc_tf[(document_id, token)] = count

def build_indices(doc_dump_path):
    index = defaultdict(set)  # For storing DF: token -> set of document IDs
    doc_tf = {}  # For storing TF: (doc_id, token) -> frequency
    with open(doc_dump_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                doc_id, _, title, abstract = parts[0], parts[1], parts[2], parts[3]
                full_text = title + " " + abstract
                add_to_index(index, doc_tf, doc_id, full_text)
    return index, doc_tf

def calculate_df(index):
    # Calculate document frequency (DF) for each term
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
    doc_dump_path = '/Users/srivantv/Downloads/nfcorpus/raw/nfdump.txt'  # Update this path
    index_file_path = '/Users/srivantv/Downloads/nfcorpus/index.txt'  # Update this path
    tf_file_path = '/Users/srivantv/Downloads/nfcorpus/tf.txt'  # Update this path
    df_file_path = '/Users/srivantv/Downloads/nfcorpus/df.txt'  # Update this path
    
    # Build indices and save
    index, doc_tf = build_indices(doc_dump_path)
    df = calculate_df(index)
    
    save_indices(index, doc_tf, df, index_file_path, tf_file_path, df_file_path)
    print("Indexing with stemming and lemmatization completed.")

if __name__ == "__main__":
    main()
