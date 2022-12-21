import numpy as np
import pandas as pd
import pickle
from Param import *


def critic(X):
    n, p = X.shape
    # X[:, 2] = min_best(X[:, 2])  # 自己的数据根据实际情况
    Z = standard(X)  # 标准化X，去量纲
    R = np.array(pd.DataFrame(Z).corr(method='pearson'))
    delta = np.zeros(p)
    c = np.zeros(p)
    for j in range(p):
        delta[j] = Z[:, j].std()
        c[j] = R.shape[0] - R[:, j].sum()
    C = delta * c
    w = C / sum(C)
    return np.round(w, p)


def min_best(X):
    for i in range(len(X)):
        X[i] = max(X) - X[i]
    return X


def standard(X):
    xmin = X.min()
    xmax = X.max()
    xmaxmin = xmax - xmin
    n, p = X.shape
    for i in range(n):
        for j in range(p):
            X[i, j] = (X[i, j] - xmin) / xmaxmin
    return X


if __name__ == '__main__':
    print("----------------use critic method determine weight--------------------")
    # index_list = []
    # index_list.append(0.9)
    # index_list.append(0.8)
    # index_list.append(0.7)
    # index_list.append(0.78)
    # pickle.dump(index_list, open("test.pkl", "wb"))
    #

    print("start load index data....")
    bert_index = pickle.load(open(BASIC_BERT_UNIT_MODEL_SAVE_PATH + BASIC_BERT_UNIT_MODEL_SAVE_PREFIX + 'bert_index.pkl','rb'))
    gcn_index = pickle.load(open(GRAPH_EMB_PATH +'_graph_index.pkl','rb'))
    predicate_index = pickle.load(open(PREDICATE_EMB_PATH + '_predicate_index.pkl','rb'))
    print("bert index :[hit @ 1 hit @ 10 hit @ 50 MRR]",bert_index)
    print("gcn index :[hit @ 1 hit @ 10 hit @ 50 MRR]", gcn_index)
    print("predicate index :[hit @ 1 hit @ 10 hit @ 50 MRR]", predicate_index)
    print("start use critic method determine weight....")
    temp_list = []
    temp_list.append(bert_index)
    temp_list.append(gcn_index)
    temp_list.append(predicate_index)
    X = np.array(temp_list)

    # X = np.array([[0.7802, 0.9552, 0.9856, 0.8745],
    #               [0.5580, 0.8516, 0.9328, 0.6756],
    #               [0.6580, 0.7516, 0.8328, 0.7756]])

    print('each index weight:',critic(X))
    W = critic(X)
    n, p = X.shape
    Y = np.zeros(n)

    for i in range(n):
        for j in range(p):
            Y[i] += X[i, j] * W[j]

    Y[2] = 0
    Y[1] = Y[1] * 4

    # 去除gcn
    # Y[1] = 0

    # Y[0] = Y[0] * 4
    final_w = Y / sum(Y)
    print('each embedding weight:',np.round(final_w, n))
    final_w_list = final_w.tolist()
    pickle.dump(final_w_list,open(CRITIC_WEIGHT_PATH, 'wb'))

    print("save weight....")
