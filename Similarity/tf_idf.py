"""
词频-逆文档频率
tf = 词汇在文本中出现的次数/文本词汇的总个数
idf = log((语料库中文本的总个数)/(包含该词汇的文本个数+1))
tf-idf = tf * idf
问答机器人：现在系统中配置好一些经常使用的标准问题，将用户的提问与标准问题进行相似度计算，获取与用户问题最相思的标准问题，返回其答案
"""

import numpy as np
import jieba


class TF_IDF_Model(object):
    def __init__(self, documents_list):
        self.documents_list = documents_list
        # 文本总个数
        self.documents_number = len(documents_list)
        # 存储每个文本中每个词的词频
        self.tf = []    # [{word1: tf1, word2: tf2}, {}, ...]
        # 存储每个词汇的逆文档频率
        self.idf = {}   # {word1: idf1, word2: idf2, ...}
        # 类初始化tf和idf
        self.init()

    def init(self):
        # 存储每个词汇在多少个文本中出现了
        df = {}
        for document in self.documents_list:
            # 存储每个文档中每个词的词频
            temp = {}
            for word in document:
                # 存储每个文档中每个词的词频
                temp[word] = temp.get(word, 0) + 1/len(document)
            self.tf.append(temp)
            for key in temp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            # 每个词的逆文档频率
            self.idf[key] = np.log(self.documents_number / (value + 1))

    def get_score(self, index, query):
        """检索query与语料库中第index个document的相关性"""
        score = 0.0
        for q in query:
            if q not in self.tf[index]:
                continue
            # 累加q的tf-idf值
            score += self.tf[index][q] * self.idf[q]
        return score

    def get_documents_score(self, query):
        """检索query与整个语料库的相关性"""
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
    model = TF_IDF_Model(document_list)
    
    # 获取问题
    query = "走私了两万元，在法律上应该怎么量刑？"
    query = list(jieba.cut(query))
    scores = model.get_documents_score(query)
    best_doc = document_list[np.argmax(scores)]

    print("query:", "".join(query))
    print("best_doc:", "".join(best_doc))
