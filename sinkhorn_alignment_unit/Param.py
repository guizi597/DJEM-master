"""
hyper-parameters:
"""
CUDA_NUM = 0  # GPU num
DATASET_TYPE = 'dbp15k'  # dataset type 'dbp15k'/'dwy-nb'/'dwy100k'/'srprs15k'

ENTITY_NEIGH_MAX_NUM = 50  # max sampling neighbor num of entity
ENTITY_ATTVALUE_MAX_NUM = 50  # max sampling attributeValue num of entity
KERNEL_NUM = 21
SEED_NUM = 11037
CANDIDATE_NUM = 50  # candidate number

BATCH_SIZE = 32  # train batch size
NEG_NUM = 5  # negative sampling num
LEARNING_RATE = 5e-4  # learning rate
MARGIN = 1  # margin
EPOCH_NUM = 200  # train epoch num
TOPK = 50

LOAD_BASIC_BERT_UNIT_MODEL_EPOCH_NUM = 4
BASIC_BERT_UNIT_MODEL_OUTPUT_DIM = 300

if DATASET_TYPE == 'dbp15k':
    LANG = 'zh'  # language 'zh'/'ja'/'fr'

    # load model(base_bert_unit_model) path
    BASIC_BERT_UNIT_MODEL_SAVE_PATH = "../Save_model/"
    BASIC_BERT_UNIT_MODEL_SAVE_PREFIX = "DBP15K_{}en".format(LANG)

    # load data path
    DATA_PATH = r"../data/dbp15k/{}_en/".format(LANG)

    # load graph embedding data
    GRAPH_EMB_PATH = r'../graph_ckpt/{}_en'.format(LANG)
    PREDICATE_EMB_PATH = r'../predicate_ckpt/{}_en'.format(LANG)

if DATASET_TYPE == 'dwy-nb':
    TYPE = 'dy'  # dataset type 'dw'/'dy'

    # load model(base_bert_unit_model) path
    BASIC_BERT_UNIT_MODEL_SAVE_PATH = "../Save_model/"
    BASIC_BERT_UNIT_MODEL_SAVE_PREFIX = "DWY-NB_{}nb".format(TYPE)

    # load data path
    DATA_PATH = r"../data/dwy-nb/{}_nb/".format(TYPE)

    # load graph embedding data
    GRAPH_EMB_PATH = r'../graph_ckpt/{}_nb'.format(TYPE)
    PREDICATE_EMB_PATH = r'../predicate_ckpt/{}_nb'.format(TYPE)

if DATASET_TYPE == 'dwy100k':
    TYPE = 'wd'  # dataset type 'wd'/'yg'

    # load model(base_bert_unit_model) path
    BASIC_BERT_UNIT_MODEL_SAVE_PATH = "../Save_model/"
    BASIC_BERT_UNIT_MODEL_SAVE_PREFIX = "DWY100K_dbp{}".format(TYPE)

    # load data path
    DATA_PATH = r"../data/dwy100k/dbp_{}/".format(TYPE)

    # load graph embedding data
    GRAPH_EMB_PATH = r'../graph_ckpt/dbp_{}'.format(TYPE)
    PREDICATE_EMB_PATH = r'../predicate_ckpt/dbp_{}'.format(TYPE)

if DATASET_TYPE == 'srprs15k':
    TYPE = 'fr'  # dataset type 'de'/'fr'

    # load model(base_bert_unit_model) path
    BASIC_BERT_UNIT_MODEL_SAVE_PATH = "../Save_model/"
    BASIC_BERT_UNIT_MODEL_SAVE_PREFIX = "SRPRS15K_en{}".format(TYPE)

    # load data path
    DATA_PATH = r"../data/srprs15k/en_{}/".format(TYPE)

    # load graph embedding data
    GRAPH_EMB_PATH = r'../graph_ckpt/en_{}'.format(TYPE)
    PREDICATE_EMB_PATH = r'../predicate_ckpt/en_{}'.format(TYPE)

if DATASET_TYPE == 'srprs15k-mo':
    TYPE = 'yg'  # dataset type 'wd'/'yg'

    # load model(base_bert_unit_model) path
    BASIC_BERT_UNIT_MODEL_SAVE_PATH = "../Save_model/"
    BASIC_BERT_UNIT_MODEL_SAVE_PREFIX = "SRPRS15K-MO_dbp{}".format(TYPE)

    # load data path
    DATA_PATH = r"../data/srprs15k-mo/dbp_{}/".format(TYPE)

    # load graph embedding data
    GRAPH_EMB_PATH = r'../graph_ckpt/dbp_{}'.format(TYPE)
    PREDICATE_EMB_PATH = r'../predicate_ckpt/dbp_{}'.format(TYPE)

# critic weight save path
CRITIC_WEIGHT_PATH = '../critic_weight/' + BASIC_BERT_UNIT_MODEL_SAVE_PREFIX + '_weight.pkl'

# candidata_save_path
TRAIN_CANDIDATES_PATH = DATA_PATH + 'train_candidates.pkl'
TEST_CANDIDATES_PATH = DATA_PATH + 'test_candidates.pkl'

# entity embedding and attributeValue embedding save path.
ENT_EMB_PATH = DATA_PATH + '{}_emb_{}.pkl'.format(BASIC_BERT_UNIT_MODEL_SAVE_PREFIX, LOAD_BASIC_BERT_UNIT_MODEL_EPOCH_NUM)
ATTRIBUTEVALUE_EMB_PATH = DATA_PATH + 'attributeValue_embedding.pkl'
ATTRIBUTEVALUE_LIST_PATH = DATA_PATH + 'attributeValue_list.pkl'  # 1-1 match to attributeValue embedding.

# (candidate) entity_pairs save path.
ENT_PAIRS_PATH = DATA_PATH + 'ent_pairs.pkl'  # [(e1,ea),(e1,eb)...]

# interaction feature save filepath name
NEIGHBORVIEW_SIMILARITY_FEATURE_PATH = DATA_PATH + 'neighbor_view_similarity_feature.pkl'  # 1-1 match to entity_pairs
ATTRIBUTEVIEW_SIMILARITY_FEATURE_PATH = DATA_PATH + 'attribute_similarity_feature.pkl'  # 1-1 match to entity_pairs
DESVIEW_SIMILARITY_FEATURE_PATH = DATA_PATH + 'des_view_similarity_feature.pkl'  # 1-1 match to entity_pairs
