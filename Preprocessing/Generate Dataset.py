import json
import pandas as pd


def get_list_of_dicts(fname): return [json.loads(i) for i in open(fname, "rt")]


def add_user_reviews(x):
    ur = user_reviews.loc[x["reviewerID"]].drop(x["asin"])
    mr = movie_reviews.loc[x["asin"]].drop(x["reviewerID"])
    x["userReviews"] = ur["reviewText"].tolist()
    x["movieReviews"] = mr["reviewText"].tolist()
    return x


raw_data = get_list_of_dicts("../data/reviews_Amazon_Instant_Video_5.json")

data = pd.DataFrame(raw_data).loc[:,
                                  ["reviewerID",
                                   "reviewText",
                                   "asin",
                                   "overall"]]

user_item_revew = data.drop("reviewText", axis=1)

user_reviews = pd.pivot_table(data,
                              index=["reviewerID", "asin"],
                              aggfunc=lambda x: x)\
    .drop("overall", axis=1)

movie_reviews = pd.pivot_table(data,
                               index=["asin", "reviewerID"],
                               aggfunc=lambda x: x)\
    .drop("overall", axis=1)

dat = user_item_revew.apply(add_user_reviews, axis=1)

dat.loc[0, ["userReviews"]][0]

dat.to_csv(path_or_buf="../data/unembedded_grouped_reviews.csv")
