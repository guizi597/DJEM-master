import tensorflow as tf
from gcn_model import *
from test_func import *
from load_data import *
import os
import pickle
import argparse

p = argparse.ArgumentParser()
p.add_argument("--lang", help="specify the language pair. (option: zh_en, ja_en, fr_en)", default="zh_en")
p.add_argument("--gpu", help="specify the gpu id. (default=0)", default="0")
p.add_argument("--hybrid", help="specify 1=HMAN/0=MAN. (default=1)", default="1")
args = p.parse_args()

LANG = args.lang
GPU = args.gpu
HYBRID = int(args.hybrid)

os.environ["CUDA_VISIBLE_DEVICES"]=GPU
seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

class Config:
    data_path = '../data/srprs15k-mo/'
    language = LANG # zh_en | ja_en | fr_en
    e1 = data_path + language + '/ent_ids_1'
    e2 = data_path + language + '/ent_ids_2'
    r1 = data_path + language + '/rel_ids_1'
    r2 = data_path + language + '/rel_ids_2'
    a1 = data_path + language + '/training_attrs_1'
    a2 = data_path + language + '/training_attrs_2'
    ill = data_path + language + '/ref_pairs'
    tr = data_path + language + '/train'
    te = data_path + language + '/test'
    dev = data_path + language + '/dev'
    kg1 = data_path + language + '/triples_1'
    kg2 = data_path + language + '/triples_2'
    epochs = 50000 if HYBRID else 2000
    se_dim = 200
    ae_dim = 100
    attr_num = 1000
    rel_dim = 100
    rel_num = 1000
    act_func = tf.nn.relu
    gamma = 3.0  # margin based loss
    k = 25  # number of negative samples for each positive one
    ckpt = "../graph_ckpt"

    print('se_dim:',se_dim)
    print('ae_dim:', ae_dim)
    print('rel_dim:', rel_dim)

if __name__ == '__main__':
    print("start load data....")
    e = len(set(loadfile(Config.e1, 1)) | set(loadfile(Config.e2, 1)))
    print(e)
    ILL = loadfile(Config.ill, 2)
    illL = len(ILL)
    train = loadfile(Config.tr, 2)
    dev = loadfile(Config.dev, 2)
    np.random.shuffle(train)
    np.random.shuffle(dev)
    train = np.array(train + dev)
    test = loadfile(Config.te, 2)
    KG1 = loadfile(Config.kg1, 3)
    KG2 = loadfile(Config.kg2, 3)
    ent2id = get_ent2id([Config.e1, Config.e2]) # attr
    attr = load_attr([Config.a1, Config.a2], e, ent2id, Config.attr_num) # attr
    rel = load_relation(e, KG1+KG2, Config.rel_num)

    if HYBRID:
        print("training GCN embedding...")
        output_layer, loss = build_HMAN(Config.se_dim, Config.act_func, Config.gamma, Config.k, \
                                        e, train, KG1 + KG2, attr, Config.ae_dim, rel, Config.rel_dim)

    graph_embd, J = training(output_layer, loss, 25, Config.epochs, train, e, Config.k, test)
    get_hits(graph_embd, test)
    with open(Config.ckpt+"/%s_graph_embd.pkl"%Config.language, "wb") as f:
        pickle.dump(graph_embd, f)


