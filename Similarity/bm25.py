"""
BM25
score(q,d) = \Sigma_i^{n} W_i R(q_i,d)
R(q_i,d)为词与标准问题的相关性分数（类似TF）
W_i为词的权重（类似IDF）
"""


import numpy as np
from collections import Counter
import jieba


class BM25_Model(object):
    def __init__(self, documents_list, k1=2, k2=1, b=0.5):
        self.documents_list = documents_list
        self.documents_number = len(documents_list)
        self.avg_documents_len = sum([len(document) for document in documents_list]) / self.documents_number
        self.f = []     # 每个词在文档中出现的次数
        self.idf = {}   # 逆文档频率
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.init()

    def init(self):
        df = {}
        for document in self.documents_list:
            temp = {}
            for word in document:
                temp[word] = temp.get(word, 0) + 1
            self.f.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            self.idf[key] = np.log((self.documents_number - value + 0.5) / (value + 0.5))

    def get_score(self, index, query):
        score = 0.0
        document_len = len(self.f[index])
        qf = Counter(query)
        for q in query:
            if q not in self.f[index]:
                continue
            score += self.idf[q] * (self.f[index][q] * (self.k1 + 1) / (
                        self.f[index][q] + self.k1 * (1 - self.b + self.b * document_len / self.avg_documents_len))) * (
                                 qf[q] * (self.k2 + 1) / (qf[q] + self.k2))

        return score

    def get_documents_score(self, query):
        score_list = []
        for i in range(self.documents_number):
            score_list.append(self.get_score(i, query))
        return score_list


if __name__ == "__main__":
    # document_list = ["行政机关强行解除行政协议造成损失，如何索取赔偿？",
    #              "借钱给朋友到期不还得什么时候可以起诉？怎么起诉？",
    #              "我在微信上被骗了，请问被骗多少钱才可以立案？",
    #              "公民对于选举委员会对选民的资格申诉的处理决定不服，能不能去法院起诉吗？",
    #              "有人走私两万元，怎么处置他？",
    #              "法律上餐具、饮具集中消毒服务单位的责任是不是对消毒餐具、饮具进行检验？"]
    with open("data/text_demo.txt", encoding="utf-8") as f:
        document_list = f.readlines()
        document_list = [document.strip() for document in document_list]
    # 获取语料库列表，并分词
    document_list = [list(jieba.cut(doc)) for doc in document_list]
    model = BM25_Model(document_list)
    
    # 获取问题
    query = "走私了两万元，在法律上应该怎么量刑？"
    query = list(jieba.cut(query))
    scores = model.get_documents_score(query)
    best_doc = document_list[np.argmax(scores)]

    print("query:", "".join(query))
    print("best_doc:", "".join(best_doc))
