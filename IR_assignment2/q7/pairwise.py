import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs
import pprint

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

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

query_ids_np = np.array(query_ids)
doc_ids_np = np.array(doc_ids)
relevance_scores_np = np.array(relevance_scores)

unique_query_ids = np.unique(query_ids_np)
unique_doc_ids = np.unique(doc_ids_np)



ratings = {
    "user_id": query_ids,
    "movie_title": doc_ids,
    "user_rating": relevance_scores
}
ratings = tf.data.Dataset.from_tensor_slices(ratings)

tf.random.set_seed(42)


shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

train = tfrs.examples.movielens.sample_listwise(
    train,
    num_list_per_user=50,
    num_examples_per_list=5,
    seed=42
)
test = tfrs.examples.movielens.sample_listwise(
    test,
    num_list_per_user=1,
    num_examples_per_list=5,
    seed=42
)

class RankingModel(tfrs.Model):

  def __init__(self, loss):
    super().__init__()
    embedding_dimension = 32

    self.user_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_query_ids),
      tf.keras.layers.Embedding(len(unique_query_ids) + 2, embedding_dimension)
    ])

    self.movie_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_doc_ids),
      tf.keras.layers.Embedding(len(unique_doc_ids) + 2, embedding_dimension)
    ])

    self.score_model = tf.keras.Sequential([
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      tf.keras.layers.Dense(1)
    ])

    self.task = tfrs.tasks.Ranking(
      loss=loss,
      metrics=[
        tfr.keras.metrics.NDCGMetric(name="ndcg_metric"),
        tf.keras.metrics.RootMeanSquaredError()
      ]
    )

  def call(self, features):
    user_embeddings = self.user_embeddings(features["user_id"])
    movie_embeddings = self.movie_embeddings(features["movie_title"])

    list_length = features["movie_title"].shape[1]
    user_embedding_repeated = tf.repeat(
        tf.expand_dims(user_embeddings, 1), [list_length], axis=1)

    concatenated_embeddings = tf.concat(
        [user_embedding_repeated, movie_embeddings], 2)

    return self.score_model(concatenated_embeddings)

  def compute_loss(self, features, training=False):
    labels = features.pop("user_rating")

    scores = self(features)

    return self.task(
        labels=labels,
        predictions=tf.squeeze(scores, axis=-1),
    )
epochs = 30

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()
hinge_model = RankingModel(tfr.keras.losses.PairwiseHingeLoss())
hinge_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

hinge_model.fit(cached_train, epochs=epochs, verbose=False)
hinge_model_result = hinge_model.evaluate(cached_test, return_dict=True)
print("{:.4f}".format(hinge_model_result["ndcg_metric"]))