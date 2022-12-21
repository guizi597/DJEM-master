print("In params:")

DATASET_TYPE = 'dbp15k' #dataset type 'dbp15k'/'dwy-nb'/'dwy100k'/'srprs15k'

CUDA_NUM = 0 # used GPU num
MODEL_INPUT_DIM  = 768
MODEL_OUTPUT_DIM = 300 # dimension of basic bert unit output embedding
RANDOM_DIVIDE_ILL = False #if is True: get train/test_ILLs by random divide all entity ILLs, else: get train/test ILLs from file.
TRAIN_ILL_RATE = 0.3 # (only work when RANDOM_DIVIDE_ILL == True) training data rate. Example: train ILL number: 15000 * 0.3 = 4500.

SEED_NUM = 11037

EPOCH_NUM = 5 #training epoch num

NEAREST_SAMPLE_NUM = 128
CANDIDATE_GENERATOR_BATCH_SIZE = 64

TOPK = 50
NEG_NUM = 2 # negative sample num
MARGIN = 3 # margin
LEARNING_RATE = 1e-5 # learning rate
TRAIN_BATCH_SIZE = 12
TEST_BATCH_SIZE = 64

DES_LIMIT_LENGTH = 128 # max length of description/name.

DES_DICT_PATH = r"../data/dbp15k/2016-10-des_dict"  # description data path
if DATASET_TYPE == 'dbp15k':
    LANG = 'zh'  # language 'zh'/'ja'/'fr'
    DATA_PATH = r"../data/dbp15k/{}_en/".format(LANG)  #data path
    MODEL_SAVE_PREFIX = "DBP15K_{}en".format(LANG)
if DATASET_TYPE == 'dwy100k':
    TYPE = 'wd'  # dataset type 'wd'/'yg'
    DATA_PATH = r"../data/dwy100k/dbp_{}/".format(TYPE)  #data path
    MODEL_SAVE_PREFIX = "DWY100K_dbp{}".format(TYPE)
if DATASET_TYPE == 'dwy-nb':
    TYPE = 'dy'  # dataset type 'dw'/'dy'
    DATA_PATH = r"../data/dwy-nb/{}_nb/".format(TYPE)  # data path
    MODEL_SAVE_PREFIX = "DWY-NB_{}nb".format(TYPE)
if DATASET_TYPE == 'srprs15k':
    TYPE = 'de'  # dataset type 'de'/'fr'
    DATA_PATH = r"../data/srprs15k/en_{}/".format(TYPE)  # data path
    MODEL_SAVE_PREFIX = "SRPRS15K_en{}".format(TYPE)
if DATASET_TYPE == 'srprs15k-mo':
    TYPE = 'wd'  # dataset type 'wd'/'yg'
    DATA_PATH = r"../data/srprs15k-mo/dbp_{}/".format(TYPE)  # data path
    MODEL_SAVE_PREFIX = "SRPRS15K-MO_dbp{}".format(TYPE)



MODEL_SAVE_PATH = "../Save_model/"                 #model save path
import os
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)


print("NEG_NUM:",NEG_NUM)
print("MARGIN:",MARGIN)
print("LEARNING RATE:",LEARNING_RATE)
print("TRAIN_BATCH_SIZE:",TRAIN_BATCH_SIZE)
print("TEST_BATCH_SIZE",TEST_BATCH_SIZE)
print("DES_LIMIT_LENGTH:",DES_LIMIT_LENGTH)
print("RANDOM_DIVIDE_ILL:",RANDOM_DIVIDE_ILL)
print("")
print("")
