import pandas as pd
import numpy as np
import json
import tensorflow as tf
from time import time
from itertools import chain


def clean(text):
    return tf.keras.preprocessing.text.text_to_word_sequence(
                                text,
                                filters='!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n',
                                lower=True,
                                split=' ')


def get_list_of_dicts(fname): return [json.loads(i) for i in open(fname, "rt")]


def clean_review(review):
    review["reviewText"] = clean(review["reviewText"])
    return review


def word_sequence_to_something(map):
    return lambda x: [map.get(word) for word in x]


def add_reviews(user_review,
                user_indexed_reviews,
                product_indexed_reviews,
                user_pad_len,
                prod_pad_len,
                pad_value):
    asin = user_review["asin"]
    reviewerID = user_review["reviewerID"]

    all_usr_reviews = user_indexed_reviews.loc[reviewerID]\
        .drop(asin)["reviewText"]
    user_reviews = tf.keras.preprocessing.sequence.pad_sequences(
        [list(chain(*all_usr_reviews))],
        maxlen=user_pad_len,
        dtype='float32',
        padding='post',
        truncating='post',
        value=pad_value)
    user_review["userReviews"] = user_reviews[0]

    all_prod_reviews = product_indexed_reviews.loc[asin]\
        .drop(reviewerID)["reviewText"]
    prod_reviews = tf.keras.preprocessing.sequence.pad_sequences(
        [list(chain(*all_prod_reviews))],
        maxlen=prod_pad_len,
        dtype='float32',
        padding='post',
        truncating='post',
        value=pad_value)
    user_review["prodReviews"] = prod_reviews
    return user_review


def get_batch(user_product,
              user_indexed_reviews,
              product_indexed_reviews,
              batch_size,
              max_user_review_len,
              max_prod_review_len,
              pad_value):
    ind = np.random.choice(user_product.shape[0],
                           size=batch_size,
                           replace=False)
    user_prods = user_product.iloc[ind, :]
    return user_prods.apply(add_reviews,
                            args=(user_indexed_reviews,
                                  product_indexed_reviews,
                                  max_user_review_len,
                                  max_prod_review_len,
                                  pad_value),
                            axis=1)


def get_timestamp():
    return pd.Timestamp(int(time()), unit="s")


def prep_data(embedding_size, user_pad_len, prod_pad_len):
    raw_data = pd.DataFrame(
        get_list_of_dicts("../data/reviews_Amazon_Instant_Video_5.json"))
    start = get_timestamp()
    data = raw_data\
        .drop("helpful", axis=1)\
        .drop("reviewTime", axis=1)\
        .drop("reviewerName", axis=1)\
        .drop("summary", axis=1)\
        .drop("unixReviewTime", axis=1)\
        .apply(clean_review, axis=1)
    end = get_timestamp()
    print("took {} to clean reviews".format(end - start))

    user_product = data.loc[:, ["reviewerID", "asin", "overall"]]

    word_sequences = data["reviewText"]
    unique_words = set([val for sublist in word_sequences for val in sublist])

    embedding_map = {word: np.random.normal(size=embedding_size).tolist()
                     for word in unique_words}

    start = get_timestamp()
    data["reviewText"] = word_sequences.apply(
        word_sequence_to_something(embedding_map))
    end = get_timestamp()
    print("took {} to convert to word embeddings".format(end - start))
    user_reviews = pd.pivot_table(data,
                                  index=["reviewerID", "asin"],
                                  aggfunc=lambda x: x)
    user_reviews.apply(lambda x: x, axis=1)
    product_reviews = pd.pivot_table(data,
                                     index=["asin", "reviewerID"],
                                     aggfunc=lambda x: x)
    return user_product, user_reviews, product_reviews


def train(values,
          user_product,
          user_reviews,
          product_reviews,
          user_pad_len,
          prod_pad_len,
          n_epochs,
          batch_size,
          embedding_size):
    n_iters = user_product.shape[0] // batch_size
    print("running for {} iterations".format(n_iters))

    optimizer, loss, y, u_input, i_input = values

    training_op = optimizer.minimize(loss)

    batch_collection_times = []
    batch_train_times = []

    pad_value = np.zeros(embedding_size)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for e in range(n_epochs):
            start_epoch = get_timestamp()
            for _ in range(n_iters):
                get_batch_timer_s = get_timestamp()
                batch = get_batch(user_product,
                                user_reviews,
                                product_reviews,
                                batch_size,
                                user_pad_len,
                                prod_pad_len,
                                pad_value)
                get_batch_timer_e = get_timestamp()
                batch_collection_times.append(get_batch_timer_e - get_batch_timer_s)

                s_train_timer = get_timestamp()
                user_reviews_for_other_movies = np.vstack(batch["userReviews"])\
                    .reshape((batch_size,
                              user_pad_len,
                              embedding_size))
                prod_reviews_for_other_movies = np.vstack(batch["prodReviews"])\
                    .reshape((batch_size,
                              user_pad_len,
                              embedding_size))
                y_inp = batch["overall"].values.reshape((batch_size, 1))
                sess.run(training_op, {y: y_inp,
                                       u_input: user_reviews_for_other_movies,
                                       i_input: prod_reviews_for_other_movies
                                       })
                e_train_timer = get_timestamp()
                batch_train_times.append(e_train_timer - s_train_timer)

                if _ % 250 == 0 and _ > 0:
                    avg_batch_collection_time = pd.Series(batch_collection_times)\
                        .mean()
                    avg_batch_train_time = pd.Series(batch_train_times).mean()
                    train_loss = sess.run(loss, {y: y_inp,
                                                u_input:
                                                 user_reviews_for_other_movies,
                                                i_input:
                                                 prod_reviews_for_other_movies
                                                })
                    print("train loss {:.2f} for iter {}".format(train_loss, _))
                    print("test loss xxx for iter {}".format(_))
                    print("spent an average of {} collecting batches".format(
                        avg_batch_collection_time))
                    print("took {} to train {} iterations in {}th epoch"\
                          .format(
                              avg_batch_train_time,
                              _,
                              e
                          ))
                    print("took {} since start of epoch"\
                          .format(get_timestamp() - start_epoch))
            end_epoch = get_timestamp()
            print("took {} to run for {} iterations".format(end_epoch - start_epoch,
                                                            n_iters))


def build_graph(batch_size, user_pad_len, prod_pad_len, embedding_size):
    print("building inputs graph")
    with tf.name_scope("inputs"):
        u_input = tf.placeholder(tf.float32, [batch_size,
                                              user_pad_len,
                                              embedding_size])
        i_input = tf.placeholder(tf.float32, [batch_size,
                                              prod_pad_len,
                                              embedding_size])
        y = tf.placeholder(tf.float32, [batch_size, 1])

    print("building towers")
    with tf.name_scope("tower"):
        u_tower = get_tower(u_input, batch_size)
        i_tower = get_tower(i_input, batch_size)

    print("building dot products")
    with tf.name_scope("dot"):
        mult = u_tower * i_tower
        sum = tf.reduce_sum(mult, axis=1, keep_dims=True)

    print("building concatenation")
    with tf.name_scope("concat"):
        cat = tf.concat([u_tower, i_tower], axis=1)

    print("building final dense layer")
    with tf.name_scope("dense"):
        dense = tf.layers.dense(cat, 1, activation=tf.nn.relu)

    with tf.name_scope("prediction"):
        prediction = dense + sum

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001,
                                         beta1=0.9,
                                         beta2=0.999,
                                         epsilon=1e-08)
    loss = tf.losses.mean_squared_error(y, prediction)

    return optimizer, loss, y, u_input, i_input



def get_tower(inp, batch_size):
    conv = tf.layers.conv1d(inputs=inp,
                            filters=50,
                            kernel_size=[5],
                            padding="same")
    pool = tf.layers.max_pooling1d(inputs=conv,
                                   pool_size=5,
                                   strides=5)
    flat = tf.reshape(pool, (batch_size, pool.shape[1] * pool.shape[2]))
    dense = tf.layers.dense(flat, 128, activation=tf.nn.relu)
    return dense

if __name__ == "__main__":
    EMBEDDING_SIZE = 50
    USER_PAD_LEN = 600
    PROD_PAD_LEN = 600
    BATCH_SIZE = 38
    user_product_map, user_reviews, product_reviews = prep_data(EMBEDDING_SIZE,
                                                                USER_PAD_LEN,
                                                                PROD_PAD_LEN)

    values = build_graph(BATCH_SIZE,
                         USER_PAD_LEN,
                         PROD_PAD_LEN,
                         EMBEDDING_SIZE)

    train(values,
          user_product_map,
          user_reviews,
          product_reviews,
          USER_PAD_LEN,
          PROD_PAD_LEN,
          1, # n_epoch
          BATCH_SIZE,
          EMBEDDING_SIZE)
