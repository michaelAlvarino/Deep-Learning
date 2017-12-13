import pandas as pd
from keras.preprocessing.text import text_to_word_sequence


def clean(text):
    return text_to_word_sequence(text,
                                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                 lower=True, split=" ")


def clean_each(x):
    return list(map(clean, x[2:-2].split()))


def clean_users_items(x):
    x["userReviews"] = clean_each(x["userReviews"])
    x["movieReviews"] = clean_each(x["movieReviews"])
    return x


data = pd.read_csv("data/unembedded_grouped_reviews.csv")

clean_split_reviews = data.apply(clean_users_items, axis=1)

pd.to_csv("data/unembedded_cleaned_grouped_reviews.csv")
