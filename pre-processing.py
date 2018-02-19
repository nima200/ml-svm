from collections import Counter
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import numpy as np
import pandas as pd


def pre_process_binary(data: pd.DataFrame, input_col: str):
    data[input_col] = data[input_col].str.lower()
    data[input_col] = data[input_col].str.replace('[^\w\s]', '')
    tokens = data[input_col].str.split(expand=True)
    words = tokens.stack().value_counts().index[:10000]
    out = np.zeros(shape=(tokens.shape[0], words.shape[0]))
    for i, row in tokens.iterrows():
        cnt = Counter(row)
        del cnt[None]
        for j, word in enumerate(words):
            if cnt[word] > 0:
                out[i][j] = 1
            else:
                out[i][j] = 0
    return out


def pre_process_frequency(data: pd.DataFrame, input_col: str):
    data[input_col] = data[input_col].str.lower()
    data[input_col] = data[input_col].str.replace('[^\w\s]', '')
    tokens = data[input_col].str.split(expand=True)
    words = pd.Series(tokens.stack().value_counts().index[:10000], index=[i for i in range(10000)])
    out = np.zeros(shape=(tokens.shape[0], words.shape[0]))
    for i, row in tokens.iterrows():
        row_trimmed = np.array([item for item in row if item is not None])
        cnt = Counter(row_trimmed)
        for j, word in enumerate(words):
            out[i][j] = round(cnt[word] / row_trimmed.shape[0], 3)
    return out


yelp_train = pd.read_csv('./data/yelp-train.txt', sep='\t', names=["Comments", "Rating"])
yelp_valid = pd.read_csv('./data/yelp-valid.txt', sep='\t', names=["Comments", "Rating"])
yelp_test = pd.read_csv('./data/yelp-test.txt', sep='\t', names=["Comments", "Rating"])
yelp_train_frequency = pre_process_frequency(yelp_train, input_col="Comments")
yelp_train_binary = pre_process_binary(yelp_train, input_col="Comments")
np.savetxt('./data/yelp-train-frequency-bow.csv', yelp_train_frequency, delimiter=',', fmt='%.3f',)
np.savetxt('./data/yelp-train-binary-bow.csv', yelp_train_binary, delimiter=',', fmt='%d')

yelp_clf_binary = BernoulliNB()
yelp_clf_binary.fit(yelp_train_binary, yelp_train["Rating"])
predicted = yelp_clf_binary.predict(yelp_test["Comments"])
np.mean(predicted == yelp_test["Comments"])