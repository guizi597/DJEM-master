import pickle
import time
from tqdm import tqdm
from scipy import optimize
import tensorflow as tf
from utils import *
import os
import numpy as np
from Param import *
from Basic_Bert_Unit_model import Basic_Bert_Unit_model

seed = 12346
# seed = 12345
np.random.seed(seed)
import torch
import torch.nn as nn
import torch.nn.functional as F

print("----------------don't use sinkhorn method to entity alignment----------------")
# choose the GPU, "-1" represents using the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print('loading entity embedding from:', ENT_EMB_PATH)
ent_emb = pickle.load(open(ENT_EMB_PATH, "rb"))

# load KGs and test set

# file_path = "../data/dbp15k/fr_en/"
file_path = DATA_PATH
print('loading other data from:', file_path)
all_triples, node_size, rel_size = load_triples(file_path, True)
train_pair, test_pair = load_aligned_pair(file_path, ratio=0)
bert_model_other_data_path = BASIC_BERT_UNIT_MODEL_SAVE_PATH + BASIC_BERT_UNIT_MODEL_SAVE_PREFIX + 'other_data.pkl'
train_ill, test_ill, eid2data = pickle.load(open(bert_model_other_data_path, "rb"))

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

print('start calculate entity similarity')

def cal_sims_mat(test_pair,feature):
    feature_a = tf.gather(indices=test_pair[0:int(len(test_pair)), 0], params=feature)
    feature_b = tf.gather(indices=test_pair[0:int(len(test_pair)), 1], params=feature)
    feature_a = feature_a.numpy()
    feature_b = feature_b.numpy()
    feature_a = feature_a.tolist()
    feature_b = feature_b.tolist()
    res_mat = cos_sim_mat_generate(feature_a,feature_b,bs=4096,cuda_num=0)
    return res_mat
def entlist2emb(Model,entids,entid2data,cuda_num):
    """
    return basic bert unit output embedding of entities
    """
    batch_token_ids = []
    batch_mask_ids = []
    for eid in entids:
        temp_token_ids = entid2data[eid][0]
        temp_mask_ids = entid2data[eid][1]

        batch_token_ids.append(temp_token_ids)
        batch_mask_ids.append(temp_mask_ids)

    batch_token_ids = torch.LongTensor(batch_token_ids).cuda(cuda_num)
    batch_mask_ids = torch.FloatTensor(batch_mask_ids).cuda(cuda_num)

    batch_emb = Model(batch_token_ids,batch_mask_ids)
    del batch_token_ids
    del batch_mask_ids
    return batch_emb
def test(Model,ent_ill,entid2data,batch_size,context = ""):
    print("-----test start-----")
    start_time = time.time()
    print(context)
    Model.eval()
    Model = Model.cuda(CUDA_NUM)
    with torch.no_grad():
        ents_1 = [e1 for e1,e2 in ent_ill]
        ents_2 = [e2 for e1,e2 in ent_ill]

        emb1 = []
        for i in range(0,len(ents_1),batch_size):
            batch_ents_1 = ents_1[i: i+batch_size]
            batch_emb_1 = entlist2emb(Model,batch_ents_1,entid2data,CUDA_NUM).detach().cpu().tolist()
            emb1.extend(batch_emb_1)
            del batch_emb_1

        emb2 = []
        for i in range(0,len(ents_2),batch_size):
            batch_ents_2 = ents_2[i: i+batch_size]
            batch_emb_2 = entlist2emb(Model,batch_ents_2,entid2data,CUDA_NUM).detach().cpu().tolist()
            emb2.extend(batch_emb_2)
            del batch_emb_2

        print("Cosine similarity of basic bert unit embedding res:")
        res_mat = cal_sims_mat(test_pair, feature)
        score,top_index = batch_topk(res_mat,batch_size,topn = TOPK,largest=True,cuda_num=CUDA_NUM)
        hit_res(top_index)
    print("test using time: {:.3f}".format(time.time()-start_time))
    print("--------------------")
# not solving by Sinkhorn operation
start_time = time.perf_counter()
print("start entity alignment don't use sinkhorn method")

bert_model_path = BASIC_BERT_UNIT_MODEL_SAVE_PATH + BASIC_BERT_UNIT_MODEL_SAVE_PREFIX + "model_epoch_" \
                  + str(LOAD_BASIC_BERT_UNIT_MODEL_EPOCH_NUM) + '.p'
Model = Basic_Bert_Unit_model(768, 300)
Model.load_state_dict(torch.load(bert_model_path, map_location='cpu'))
test(Model,test_pair,eid2data,32)
print(time.perf_counter() - start_time)
