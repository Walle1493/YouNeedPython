import numpy as np


def pageRank(M, damping=0.85, norm=100, delta=0.00001):
    assert M.shape[0] == M.shape[1]
    page_num = M.shape[0]
    
    e = np.ones(page_num)
    r = e * (1 - damping) / page_num
    P = np.random.random(page_num).reshape((page_num, -1))

    while norm > delta:
        P0 = P
        P = damping * np.matmul(M, P) + r
        norm = 0
        for i in range(page_num):
            norm += np.abs(P[i][0] - P0[i][0])
    
    return P
