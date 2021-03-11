import argparse
from utils import probTransMatrix
from pagerank import pageRank
import numpy as np


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--page_num", default=8, type=int)
    parser.add_argument("--damping_factor", default=0.85, type=float)
    parser.add_argument("--iter_delta", default=0.00001, type=float)
    parser.add_argument("--norm", default=100, type=int)
    
    args = parser.parse_args()

    # M = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 0, 0]])
    M = np.random.randint(2, size=args.page_num*args.page_num).reshape((args.page_num, -1))
    M = probTransMatrix(M)
    PR = pageRank(M, args.damping_factor, args.norm, args.iter_delta)

    print(M)
    print(PR)


if __name__ == "__main__":
    main()
