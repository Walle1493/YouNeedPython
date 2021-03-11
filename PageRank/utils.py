import numpy as np


def probTransMatrix(M):
    """求转移概率矩阵"""
    assert M.shape[0] == M.shape[1]

    page_num = M.shape[0]
    m = np.sum(M, axis=1).reshape((page_num, -1))
    return np.transpose(M / m)
