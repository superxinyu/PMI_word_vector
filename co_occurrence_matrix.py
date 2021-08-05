import jieba
import numpy as np


def get_matrix(sentence_list, return_word_id=False):
    """
    获取词频共现矩阵
    :param sentence_list: 句子的list, return_word_id:返回共现矩阵还是返回每个词的表示
    :return: 返回共现矩阵和id2word，id2word是用来活得每个词对应的位置
    """
    matrix_dict = {}
    word2id = {}
    # 先转化为共现字典，并记录word2id
    for s in sentence_list:
        word_list = jieba.lcut(s)
        for word_key in word_list:
            if word_key not in word2id.keys():
                word2id[word_key] = len(word2id.keys())
            if word2id[word_key] not in matrix_dict.keys():
                matrix_dict[word2id[word_key]] = {}
            for word_value in word_list:
                if word_value not in word2id.keys():
                    word2id[word_value] = len(word2id.keys())
                if word_value == word_key:
                    continue
                if word2id[word_value] not in matrix_dict[word2id[word_key]].keys():
                    matrix_dict[word2id[word_key]][word2id[word_value]] = 1
                else:
                    matrix_dict[word2id[word_key]][word2id[word_value]] += 1
    # 根据共现字典得到共现矩阵
    word_num = len(matrix_dict.keys())
    matrix = np.zeros((word_num, word_num))
    for word_key, word_value_dict in matrix_dict.items():
        for word_value_key, word_value_value in word_value_dict.items():
            matrix[word_key][word_value_key] = word_value_value
    if not return_word_id:
        return matrix
    id2word = {v: k for k, v in word2id.items()}
    return matrix,id2word
