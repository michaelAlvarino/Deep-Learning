# Deep Learning Final Project Repo
#### Colby Wise | Mike Alvarino | Richard Dewey @ Columbia.edu

### Project Overview:
In this research paper we apply the methodology outlined in the arXiv working
paper: ”Joint Deep Modeling of Users and Items Using Reviews for
Recommendation” for rating prediction of movies using the Amazon Instant Video
data set and GloVe.6B 50 dimensional word embeddings. Of the data set there
are only 18,000  text reviews. The approach used in this paper models users
and items jointly using review text in two cooperative neural networks.

Before attempting to train the networks as provided in this repository, the
user must preprocess the amazon instant video dataset. The goal of this
process is for one data point to contain all of the users reviews (excluding
the review for the current movie), all of the movie's reviews (excluding that
written by the current user), and the associated rating. We have provided some
notebooks and examples in the `Preprocessing` directory that may be useful.

Because one of the primary goals of our project was to explore the
effectiveness of different sequential data modeling neural network layers, it
was natural to split the code base into three different source files
corresponding with the three different layers we analyzed.

Before training any of these models the data is split into training and
testing sets. In this case, the sets are not a simple shuffle of the dataset
because we do not want our network to ever see users or items that appear in
the test set prior to test. We therefore use the following approach:

1. get all unique reviewers
1. extract the unique reviewers' data points to a test set
1. place all remaining data points in a training set
1. from the test set get a set of unique movies
1. remove all entries with these movies from the training set

Following this procedure requires us to discard some data, but means that our
testing set is entirely independent of the training set.

### Data Utilized:
1. Amazon Instant Video 5-core via Julian McAuley @ UCSD.
   Available as of 11/27/17
   URL: http://jmcauley.ucsd.edu/data/amazon/

1. Global Vectors for Word Representation (GloVe) version: 6B.50d.txt
   via J.Pennington, R. Socher, C.Manning @ Stanford
   Available as of 11/27/17
   URL: https://nlp.stanford.edu/projects/glove/

### Environment:
1. requirements.txt included for reference of packages used.

### Source Code:
1. DeepCoNN-CNN.ipynb - re-implementation of the paper
1. DeepCoNN-GRU.ipynb - joint model with GRU instead of CNN
1. DeepCoNN-LSTM.ipynb - joint model with LSTM instead of CNN
1. Custom Functions.py - utility functions implemented
