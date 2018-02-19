from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd

yelp_train = pd.read_csv('./data/yelp-train.txt', sep='\t')
yelp_train_freq = np.genfromtxt('./data/yelp-train-frequency-bow.csv', delimiter=',')
yelp_train_binary = np.genfromtxt('./data/yelp-train-binary-bow.csv', delimiter=',')

clf = MultinomialNB().fit(yelp_train_binary, yelp_train["Rating"])
print("hello")
