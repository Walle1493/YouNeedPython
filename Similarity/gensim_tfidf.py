from gensim import corpora, models, similarities
from pprint import pprint

from numpy.testing._private.utils import print_assert_equal


# dictionary/corpus
def GenDictandCorpus():
    documents = ["Human machine interface for lab abc computer applications",
                 "A survey of user opinion of computer system response time",
                 "The EPS user interface management system",
                 "System and human system engineering testing of EPS",
                 "Relation of user perceived response time to error measurement",
                 "The generation of random binary unordered trees",
                 "The intersection graph of paths in trees",
                 "Graph minors IV Widths of trees and well quasi ordering",
                 "Graph minors A survey"]

    texts = [[word for word in document.lower().split()] for document in documents]

    # 词典
    dictionary = corpora.Dictionary(texts)
    # 词库，以(词索引，词频)方式存贮
    corpus = [dictionary.doc2bow(text) for text in texts]
    # print(dictionary)
    # print(corpus)
    return dictionary, corpus


# TF-IDF
def Tfidf():
    dictionary, corpus = GenDictandCorpus()

    # initialize a model
    tfidf = models.TfidfModel(corpus)
    # print(tfidf)

    # Transforming vectors
    # 此时，tfidf被视为一个只读对象，可以用于将任何向量从旧表示（词频）转换为新表示（TfIdf实值权重）
    # doc_bow = [(0, 1), (1, 1)]
    # 使用模型tfidf，将doc_bow(由词,词频)表示转换成(词,tfidf)表示
    # print(tfidf[doc_bow])

    # 转换整个词库
    corpus_tfidf = tfidf[corpus]
    # for doc in corpus_tfidf:
    #     print(doc)

    return corpus_tfidf


# LDA
def LDA():
    dictionary, corpus = GenDictandCorpus()
    corpus_tfidf = Tfidf()
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=2)

    return lda.print_topics()
    # pprint(lda.print_topics())


if __name__ == "__main__":
    pprint(LDA())
