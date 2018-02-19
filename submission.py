
# coding: utf-8

# In[1]:


from collections import Counter
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import numpy as np
import pandas as pd


# In[2]:
def get_vocab(data: pd.DataFrame, input_col: str):
    data[input_col] = data[input_col].str.lower()
    data[input_col] = data[input_col].str.replace('[^\w\s]', '')
    tokens = data[input_col].str.split(expand=True)
    return tokens.stack().value_counts()


def pre_process_binary(data, input_col, vocab):
    data[input_col] = data[input_col].str.lower()
    data[input_col] = data[input_col].str.replace('[^\w\s]', '')
    tokens = data[input_col].str.split(expand=True)
    out = np.zeros(shape=(tokens.shape[0], vocab.shape[0]))
    for i, row in tokens.iterrows():
        cnt = Counter(row)
        del cnt[None]
        for j, word in enumerate(vocab):
            if cnt[word] > 0:
                out[i][j] = 1
            else:
                out[i][j] = 0
    return out


# In[3]:


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


# In[4]:


yelp_train = pd.read_csv('./data/yelp-train.txt', sep='\t', names=["Comments", "Rating"])
yelp_valid = pd.read_csv('./data/yelp-valid.txt', sep='\t', names=["Comments", "Rating"])
yelp_test = pd.read_csv('./data/yelp-test.txt', sep='\t', names=["Comments", "Rating"])


# In[5]:

vocab = get_vocab(data=yelp_train, input_col="Comments")
yelp_valid_x = pre_process_binary(yelp_valid, "Comments", vocab.index[0:10000])
yelp_train_x = pre_process_binary(yelp_train, "Comments", vocab.index[0:10000])
yelp_test_x = pre_process_binary(yelp_test, "Comments", vocab.index[0:10000])

yelp_train_y = np.array(yelp_train["Rating"])
yelp_valid_y = np.array(yelp_valid["Rating"])
yelp_test_y = np.array(yelp_test["Rating"])


# def nb_classification(x, y, a):
#     nb_clf = BernoulliNB(alpha=a)
#     nb_clf.fit(x, y)
#     return nb_clf
#
#
# def compute_f1(clf, train_x, train_y, valid_x, valid_y, test_x, test_y, ave):
#     f1_train = f1_score(train_y, clf.predict(train_x), average=ave)
#     f1_valid = f1_score(valid_y, clf.predict(valid_x), average=ave)
#     f1_test = f1_score(test_y, clf.predict(test_x), average=ave)
#     return f1_train, f1_valid, f1_test


f1s = np.zeros(shape=(99, 2))
alphas = np.arange(1e-2, 1, 0.01)
for i, alpha in enumerate(alphas):
    nb_clf = BernoulliNB(alpha=alpha)
    nb_clf.fit(yelp_train_x, yelp_train_y)
    f1s[i][0] = alpha
    f1s[i][1] = f1_score(yelp_valid_y, nb_clf.predict(yelp_valid_x), average='weighted')


print(f1s)
