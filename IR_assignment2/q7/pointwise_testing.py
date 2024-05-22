import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Dot, Flatten, StringLookup
from tensorflow.keras.optimizers import Adam

file_path = 'IR_assignment2/nfcorpus/unfucked_merged.qrel'

query_ids = []
doc_ids = []
relevance_scores = []

with open(file_path, 'r') as file:
    for line in file:
        parts = line.split()
        query_ids.append(parts[0])
        doc_ids.append(parts[2])
        relevance_scores.append(float(parts[3]))

relevance_scores = np.array(relevance_scores)

query_lookup = StringLookup()
doc_lookup = StringLookup()

query_lookup.adapt(query_ids)
doc_lookup.adapt(doc_ids)

train_query_indices = query_lookup(query_ids)
train_doc_indices = doc_lookup(doc_ids)

num_queries = query_lookup.vocabulary_size()
num_docs = doc_lookup.vocabulary_size()
embedding_size = 60

relevance_scores_matrix = np.zeros((num_queries, num_docs))

for i in range(len(relevance_scores)):
    q_index = train_query_indices.numpy()[i]
    d_index = train_doc_indices.numpy()[i]
    relevance_scores_matrix[q_index, d_index] = relevance_scores[i]

query_input = Input(shape=(1,), dtype='int64', name='query_input')
query_embedding = Embedding(num_queries, embedding_size, input_length=1)(query_input)
query_vec = Flatten()(query_embedding)

doc_input = Input(shape=(1,), dtype='int64', name='doc_input')
doc_embedding = Embedding(num_docs, embedding_size, input_length=1)(doc_input)
doc_vec = Flatten()(doc_embedding)

dot_product = Dot(axes=1)([query_vec, doc_vec])

model = Model(inputs=[query_input, doc_input], outputs=dot_product)
model.compile(optimizer=Adam(0.001), loss='mean_squared_error', metrics=['mae'])

model.fit([train_query_indices, train_doc_indices], relevance_scores, epochs=30, batch_size=32)

all_query_ids = np.repeat(np.arange(num_queries), num_docs)
all_doc_ids = np.tile(np.arange(num_docs), num_queries)
predicted_scores = model.predict([all_query_ids, all_doc_ids], batch_size=1024)

predicted_scores_matrix = predicted_scores.reshape(num_queries, num_docs)

def calculate_ndcg(true_relevance, predicted_scores, k=5):
    ranking = np.argsort(-predicted_scores)[:k]
    true_scores = true_relevance[ranking]
    ideal_scores = np.sort(true_relevance)[-1:-k-1:-1]
    
    dcg = np.sum((2**true_scores - 1) / np.log2(np.arange(2, k+2)))
    idcg = np.sum((2**ideal_scores - 1) / np.log2(np.arange(2, k+2)))
    
    return dcg / idcg if idcg > 0 else 0

ndcg_scores = [calculate_ndcg(relevance_scores_matrix[i], predicted_scores_matrix[i]) for i in range(num_queries)]
average_ndcg = np.mean(ndcg_scores)
print(f"{average_ndcg}")
