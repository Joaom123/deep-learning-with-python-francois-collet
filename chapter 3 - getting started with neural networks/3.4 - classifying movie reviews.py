# classify movie reviews as
# positive or negative, based on the text content of the reviews

from keras.datasets import imdb

# train_data and test_data are lists of reviews.
# each review is a list of word indices
# train_labels and test_labels are a list of 0's and 1's
# 0 -> negative review | 1 -> positive review
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# preparing the data
# encoding the integer sequences into a binary matrix
