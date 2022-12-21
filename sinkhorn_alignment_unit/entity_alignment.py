import pickle
import time
from tqdm import tqdm
from scipy import optimize
import tensorflow as tf
from utils import *
import os
import numpy as np
from Param import *

seed = 12346
# seed = 12345
np.random.seed(seed)
import torch
import torch.nn as nn
import torch.nn.functional as F

print('----------------use sinkhorn method to entity alignment----------------')
# choose the GPU, "-1" represents using the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print('loading entity embedding from:', ENT_EMB_PATH)
ent_emb = pickle.load(open(ENT_EMB_PATH, "rb"))
# w/o PE
new_ent_emb = []
for temp_list in ent_emb:
    new_ent_emb.append(temp_list[:500])

ent_emb = new_ent_emb
# load KGs and test set

# file_path = "../data/dbp15k/fr_en/"
file_path = DATA_PATH
print('loading other data from:', file_path)
all_triples, node_size, rel_size = load_triples(file_path, True)
train_pair, test_pair = load_aligned_pair(file_path, ratio=0)


ent_vec = np.zeros((node_size, len(ent_emb[0])))
i = 0
for emb in ent_emb:
    ent_vec[i] += emb
    ent_vec[i] = ent_vec[i] / np.linalg.norm(ent_vec[i])
    i += 1
print('ent_vec shape:',ent_vec.shape)
# build the relational adjacency matrix

dr = {}
for x, r, y in all_triples:
    if r not in dr:
        dr[r] = 0
    dr[r] += 1

sparse_rel_matrix = []
for i in range(node_size):
    sparse_rel_matrix.append([i, i, np.log(len(all_triples) / node_size)])
for h, r, t in all_triples:
    sparse_rel_matrix.append([h, t, np.log(len(all_triples) / dr[r])])

sparse_rel_matrix = np.array(sorted(sparse_rel_matrix, key=lambda x: x[0]))
sparse_rel_matrix = tf.SparseTensor(indices=sparse_rel_matrix[:, :2], values=sparse_rel_matrix[:, 2], dense_shape=(node_size, node_size))

# feature selection
feature = ent_vec
feature = tf.nn.l2_normalize(feature, axis=-1)
print('feature lenth is:', len(feature))


def cal_sims(test_pair,feature):
    if len(test_pair)<=50000:
        feature_a = tf.gather(indices=test_pair[0:int(len(test_pair)),0],params=feature)
        feature_b = tf.gather(indices=test_pair[0:int(len(test_pair)),1],params=feature)
    else:
        feature_a = tf.gather(indices=test_pair[0:int(len(test_pair)/2),0],params=feature)
        feature_b = tf.gather(indices=test_pair[0:int(len(test_pair)/2),1],params=feature)
    # feature_a = tf.gather(indices=test_pair[0:int(len(test_pair)), 0], params=feature)
    # feature_b = tf.gather(indices=test_pair[0:int(len(test_pair)), 1], params=feature)
    # feature_a = feature_a.numpy()
    # feature_b = feature_b.numpy()
    # feature_a = feature_a.tolist()
    # feature_b = feature_b.tolist()
    # res_mat = cos_sim_mat_generate(feature_a,feature_b,bs=4096,cuda_num=0)
    # return res_mat
    return tf.matmul(feature_a,tf.transpose(feature_b,[1,0]))
print('start calculate entity similarity')

# choose the graph depth L and feature propagation
start_time = time.perf_counter()
depth = 1
print('depth L:',depth)
sims = cal_sims(test_pair, feature)

for i in range(depth):
    feature = tf.sparse.sparse_dense_matmul(sparse_rel_matrix, feature)
    feature = tf.nn.l2_normalize(feature, axis=-1)
    sims += cal_sims(test_pair, feature)
sims /= depth + 1

end_time = time.perf_counter()
print(end_time - start_time)

# #solving by Hungarian algorithm, only for the CPU
# start_time = time.perf_counter()
# print('start entity alignment use hungarian method')
# result = optimize.linear_sum_assignment(sims,maximize=True)
# test(result,"hungarian")
# end_time = time.perf_counter()
# print(end_time - start_time)

# solving by Sinkhorn operation
start_time = time.perf_counter()
print('start entity alignment use sinkhorn method')
temperature = 0.002
print('temperature τ:',temperature)
sims = tf.exp(sims * (1/temperature))
for k in range(1,21):
    print("iterate k:",k)
    sims = sims / tf.reduce_sum(sims, axis=1, keepdims=True)
    sims = sims / tf.reduce_sum(sims, axis=0, keepdims=True)
    test(sims, "sinkhorn")

end_time = time.perf_counter()
print(end_time - start_time)
