from warnings import simplefilter
import tensorflow as tf
simplefilter(action='ignore', category=FutureWarning)
import logging

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import pickle
import time
from Param import *
from Basic_Bert_Unit_model import Basic_Bert_Unit_model
import os

def fixed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main():
    print("----------------get entity embedding--------------------")
    cuda_num = CUDA_NUM
    batch_size = 16
    print("GPU NUM:", cuda_num)

    # load basic bert unit model
    bert_model_path = BASIC_BERT_UNIT_MODEL_SAVE_PATH + BASIC_BERT_UNIT_MODEL_SAVE_PREFIX + "model_epoch_" \
                      + str(LOAD_BASIC_BERT_UNIT_MODEL_EPOCH_NUM) + '.p'
    Model = Basic_Bert_Unit_model(768, BASIC_BERT_UNIT_MODEL_OUTPUT_DIM)  # 768->300
    Model.load_state_dict(torch.load(bert_model_path, map_location='cpu'))
    print("loading basic bert unit model from:  {}".format(bert_model_path))
    Model.eval()
    for name, v in Model.named_parameters():
        v.requires_grad = False
    Model = Model.cuda(cuda_num)

    # read other data from bert unit model(train ill/test ill/eid2data)
    # (These files were saved during the training of basic bert unit)
    bert_model_other_data_path = BASIC_BERT_UNIT_MODEL_SAVE_PATH + BASIC_BERT_UNIT_MODEL_SAVE_PREFIX + 'other_data.pkl'
    train_ill, test_ill, eid2data = pickle.load(open(bert_model_other_data_path, "rb"))

    print("train_ill num: {} /test_ill num: {} / train_ill & test_ill num: {}".format(len(train_ill), len(test_ill),
                                                                                      len(set(train_ill) & set(
                                                                                          test_ill))))


    # generate entity embedding by basic bert unit
    start_time = time.time()
    bert_emb = []
    for eid in range(0, len(eid2data.keys()), batch_size):  # eid == [0,n)
        token_inputs = []
        mask_inputs = []
        for i in range(eid, min(eid + batch_size, len(eid2data.keys()))):
            token_input = eid2data[i][0]
            mask_input = eid2data[i][1]
            token_inputs.append(token_input)
            mask_inputs.append(mask_input)
        vec = Model(torch.LongTensor(token_inputs).cuda(cuda_num),
                    torch.FloatTensor(mask_inputs).cuda(cuda_num))
        bert_emb.extend(vec.detach().cpu().tolist())
    print("get bert embedding using time {:.3f}".format(time.time() - start_time))
    print("bert embedding shape: ", np.array(bert_emb).shape)

    gcn_emb_data_path = GRAPH_EMB_PATH + "_graph_embd.pkl"
    gcn_emb = pickle.load(open(gcn_emb_data_path, "rb"))
    print("gcn embedding shape: ", np.array(gcn_emb).shape)

    predicate_emb_data_path = PREDICATE_EMB_PATH + "_predicate_embd.pkl"
    predicate_emb = pickle.load(open(predicate_emb_data_path, "rb"))
    print("predicate embedding shape: ", np.array(predicate_emb).shape)

    # graph_emb = graph_emb.tolist()
    # for i in range(len(graph_emb)):
    #     temp_emb = ent_emb[i]
    #     temp_emb.extend(graph_emb[i])
    #     ent_emb[i] = temp_emb


    weight_list = pickle.load(open(CRITIC_WEIGHT_PATH, 'rb'))
    bert_weight = weight_list[0]
    gcn_weight = weight_list[1]
    predicate_weight = weight_list[2]

    # gcn_emb_torch = torch.from_numpy(gcn_emb)
    # gcn_emb_torch = gcn_emb_torch.tanh(gcn_emb_torch)
    # gcn_emb = gcn_emb_torch.numpy()

    bert_emb = np.array(bert_emb)
    # bert_emb_tensor = tf.convert_to_tensor(bert_emb)
    # bert_emb_tensor = tf.nn.l2_normalize(bert_emb_tensor)
    # bert_emb = bert_emb_tensor.numpy()

    ent_emb = np.concatenate([bert_weight*bert_emb,gcn_weight*gcn_emb,predicate_weight*predicate_emb], axis=1)

    # w/o DE
    # ent_emb = gcn_emb

    # w/o SE
    # ent_emb = np.concatenate([bert_weight*bert_emb,predicate_weight*predicate_emb], axis=1)

    # w/o PE
    # gcn_emb_list = gcn_emb.tolist()
    # new_gcn_emb_list = []
    # for temp_list in gcn_emb_list:
    #     new_gcn_emb_list.append(temp_list[:200])
    # new_gcn_emb = np.array(new_gcn_emb_list)
    # ent_emb = np.concatenate([bert_weight * bert_emb, gcn_weight * new_gcn_emb], axis=1)



    print("connect bert,gcn,predicate entity embedding shape: ", np.array(ent_emb).shape)
    # save entity embedding.
    ent_emb = ent_emb.tolist()
    pickle.dump(ent_emb, open(ENT_EMB_PATH, "wb"))
    print("save entity embedding....")



if __name__ == '__main__':
    fixed(SEED_NUM)
    main()
