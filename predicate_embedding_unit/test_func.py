import numpy as np
import scipy.spatial
import pickle
from main import Config

def get_hits(vec, test_pair, top_k=(1, 10, 50, 100, 200)):
    Lvec = np.array([vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([vec[e2] for e1, e2 in test_pair])

    LMRR = 0
    RMRR = 0

    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    ent1_num, ent2_num = sim.shape

    top_lr = [0] * len(top_k)
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        LMRR += (1 / (rank_index + 1))
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1

    top_rl = [0] * len(top_k)
    for i in range(Rvec.shape[0]):
        rank = sim[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]
        RMRR += (1 / (rank_index + 1))
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1

    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print("")
    LMRR /= ent1_num
    print("MRR:{:.5f}".format(LMRR))

    print('For each right:')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
    print("")
    RMRR /= ent2_num
    print("MRR:{:.5f}".format(RMRR))

    index_list = []
    index_list.append(top_lr[0] / len(test_pair))
    index_list.append(top_lr[1] / len(test_pair))
    index_list.append(top_lr[2] / len(test_pair))
    index_list.append(LMRR)
    pickle.dump(index_list, open(Config.ckpt+"/%s_predicate_index.pkl"%Config.language, "wb"))
